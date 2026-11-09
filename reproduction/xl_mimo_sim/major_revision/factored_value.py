"""Causal known-cost interface and bounded unknown-return approximation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.action_value import observation,observation_size,action_size,baseline_command
from major_revision.action_value_report import numeric_rows
from major_revision.action_value_study import ROOT as BASE_ROOT
from major_revision.full_domain import FullDomainRollout
from major_revision.gram_control import representation_size,subarray_grams,rates_from_gram
from major_revision.pilot import trajectory,SCENARIOS
from major_revision.rollout_value_data import ROOT as VALUE_ROOT

ROOT=Path('results/major_revision_20260921_factored_value')


def current_features(run,score):
    """Use freshly observed ACTIVE CSI; never future or sleeping true CSI."""
    cfg=run.cfg
    if cfg.wake_delay_slots<=0:
        raise ValueError('This current-CSI identity requires positive recovery delay')
    masks=np.asarray([select_top(score,n) for n in range(1,cfg.subarrays+1)])
    available=masks&run.hw.active[None,:]
    assert not np.any(available&~run.observed[None,:])
    grams=np.einsum('aq,qkl->akl',available,subarray_grams(cfg,run.tracker.channel_estimate))
    rates=rates_from_gram(cfg,grams,available.sum(1)*cfg.antennas_per_subarray)
    # Preserve threshold outcome separately, before float32 and ratio clipping.
    outage=(rates<run.req).mean(1)
    return np.concatenate((np.clip(rates/run.req,0.,4.),outage[:,None]),axis=1).astype(np.float32)


def extend_observation(run,setting):
    original,score=observation(run)
    x=np.concatenate((original,np.asarray([setting['margin'],setting['minimum']/run.cfg.subarrays],
        dtype=np.float32),current_features(run,score).ravel()))
    return x,score


class FactoredNetwork(nn.Module):
    def __init__(self,cfg,structured=True):
        super().__init__()
        self.cfg,self.structured=cfg,structured
        self.tail_weight=sum(cfg.rl_gamma**i for i in (1,2,3))
        self.common=nn.Sequential(nn.Linear(representation_size(cfg)+2,64),nn.LayerNorm(64),nn.SiLU())
        self.action=nn.Sequential(nn.Linear(action_size(cfg)+cfg.users+1,32),nn.LayerNorm(32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,x):
        cfg=self.cfg
        common_end=representation_size(cfg)
        legacy_end=observation_size(cfg)
        common=self.common(torch.cat((x[:,:common_end],x[:,legacy_end:legacy_end+2]),dim=-1))
        actions=x[:,common_end:legacy_end].reshape(-1,cfg.subarrays,action_size(cfg))
        current=x[:,legacy_end+2:].reshape(-1,cfg.subarrays,cfg.users+1)
        encoding=self.action(torch.cat((actions,current),dim=-1))
        raw=self.head(torch.cat((common[:,None].expand(-1,cfg.subarrays,-1),encoding),dim=-1))
        e0=actions[:,:,cfg.subarrays:cfg.subarrays+4].mean(-1) if self.structured else nn.functional.softplus(raw[:,:,0])
        p0=torch.sigmoid(raw[:,:,1])
        o0=current[:,:,-1]/4+.75*p0 if self.structured else p0
        tail_e=nn.functional.softplus(raw[:,:,2])
        tail_o=self.tail_weight*torch.sigmoid(raw[:,:,3])
        return torch.stack((torch.stack((e0,o0),dim=-1),torch.stack((e0+tail_e,o0+tail_o),dim=-1)),dim=-2)

    def initialize_from_training(self,x,costs):
        """Training-only constant component priors; never a fitted endpoint choice."""
        cfg=self.cfg
        current=x[:,observation_size(cfg)+2:].reshape(-1,cfg.subarrays,cfg.users+1)[:,:,-1]
        root_outage=(costs[:,:,0,1]-current/4)/.75 if self.structured else costs[:,:,0,1]
        tail=costs[:,:,1]-costs[:,:,0]
        assert tail[:,:,0].min()>-1e-7 and tail[:,:,1].min()>-1e-7
        assert root_outage.min()>-1e-6 and root_outage.max()<1+1e-6
        means=[costs[:,:,0,0].mean(),root_outage.mean(),tail[:,:,0].mean(),tail[:,:,1].mean()/self.tail_weight]
        bias=[]
        for i,value in enumerate(means):
            if i in (0,2):
                bias.append(float(np.log(np.expm1(max(float(value),1e-3)))))
            else:
                probability=np.clip(value,.01,.99)
                bias.append(float(np.log(probability/(1-probability))))
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor(bias))


def relative(costs,reference):
    return costs-costs[torch.arange(len(costs),device=costs.device),reference][:,None]


def collect(split):
    assert (VALUE_ROOT/'data_audit/COMPLETE').exists()
    old_audit=json.loads((VALUE_ROOT/'data_audit/audit.json').read_text())
    for path,digest in old_audit['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    cfg=make_config('validation')
    source=VALUE_ROOT/split/'labels.npz'
    old=np.load(source,allow_pickle=False)
    settings=numeric_rows(BASE_ROOT/'baselines/selected.csv')
    stream,count=(1010000000,24) if split=='train' else (1020000000,12)
    folder=ROOT/split
    folder.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('FACTORED_VALUE_PLAN.md'),
        Path(__file__).with_name('test_factored_value.py'),source,VALUE_ROOT/'data_audit/audit.json',
        BASE_ROOT/'baselines/selected.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (folder/'manifest.json').write_text(json.dumps(dict(config=cfg.as_dict(),split=split,
        root_states=len(old['features']),input_dimension=1500,threads=1,hashes=hashes,
        role='causal feature extension only; archived future labels unchanged'),indent=2))
    key_to_index={tuple(key):i for i,key in enumerate(old['keys'])}
    features=np.empty((len(key_to_index),1500),dtype=np.float32)
    seen=set()
    energy_error=0.
    current_error=0.
    for j,sc in enumerate(SCENARIOS):
        setting=next(r for r in settings if r['method']=='cost' and r['scenario']==j)
        for i in range(count):
            run=FullDomainRollout(cfg,trajectory(cfg,stream+j*10000+i,sc))
            while run.t<cfg.episode_steps:
                if run.t%4:
                    run.step()
                    continue
                key=(j,i,run.t)
                if key in key_to_index:
                    index=key_to_index[key]
                    x,score=extend_observation(run,setting)
                    np.testing.assert_array_equal(x[:1436],old['features'][index])
                    features[index]=x
                    seen.add(index)
                    rows=x[representation_size(cfg):observation_size(cfg)].reshape(cfg.subarrays,action_size(cfg))
                    e0=rows[:,cfg.subarrays:cfg.subarrays+4].astype(float).mean(-1)
                    energy_error=max(energy_error,float(np.max(np.abs(e0-old['costs'][index,:,0,0]))))
                    current=x[1436:].reshape(cfg.subarrays,4)
                    for a in range(cfg.subarrays):
                        mask=select_top(score,a+1)&run.hw.active
                        direct=run.system.achievable_rates(run.tracker.channel_estimate,mask)
                        np.testing.assert_allclose(current[a,:3],np.clip(direct/run.req,0,4),atol=3e-7,rtol=1e-7)
                        current_error=max(current_error,abs(current[a,-1]-float(np.mean(direct<run.req))))
                command=baseline_command(run,'cost',setting['margin'],setting['minimum'])
                if key in key_to_index:
                    assert int(command.sum())-1==old['reference'][key_to_index[key]]
                run.step(command)
            print('factored causal data',split,j,i+1,'/',count,flush=True)
    assert len(seen)==len(features) and energy_error<5e-7 and current_error<3e-8
    np.savez_compressed(folder/'features.npz',features=features,keys=old['keys'],reference=old['reference'])
    for path,digest in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    (folder/'audit.json').write_text(json.dumps(dict(roots=len(features),energy_max_error=energy_error,
        current_outage_float32_error=current_error,old_features_exact=True,reference_exact=True,
        feature_hash=hashlib.sha256((folder/'features.npz').read_bytes()).hexdigest(),hashes_match=True),indent=2))
    (folder/'COMPLETE').write_text('Causal features replayed; no new future labels or policy result.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=('train','validation'),required=True)
    torch.set_num_threads(1)
    collect(parser.parse_args().split)
