"""Frozen-policy audit: one causal PRS action, then return control to the same RL."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN, select_top
from future_vr.experiment import _write_rows
from future_vr.rate_search import rate_search_command
from major_revision.audit_rollout import AuditRollout
from major_revision.pilot import SCENARIOS, trajectory
from major_revision.ready_report import read

ROOT = Path('results/major_revision_20260920_policy_intervention')
TRAINING = Path('results/major_revision_20260920_time_limit')
SLOTS = (0,32,64,96)


def realized_return(run, start):
    cfg = run.cfg
    full = run.system.evaluate(run.tr.channel[0],np.ones(cfg.subarrays,bool),
        np.zeros(cfg.subarrays,bool),0.,False,run.tr.requirements_mbps[0],
        observed_count=cfg.subarrays).energy_j
    e = np.asarray(run.slot_energy[start:])/full
    c = np.asarray(run.outages[start:])
    discount = cfg.rl_gamma**(np.arange(len(e))//4)
    return float(np.sum(discount*(-e-4*c))/4)


def run():
    torch.set_num_threads(1)
    cfg = make_config('validation')
    ROOT.mkdir(parents=True,exist_ok=False)
    selection = Path('results/major_revision_20260920_constraint_screen/prs/selected.csv')
    margins = {int(r['scenario']):float(r['margin']) for r in read(selection)}
    sources = [Path(__file__),Path(__file__).with_name('audit_rollout.py'),
               Path(__file__).with_name('pilot.py'),Path(__file__).with_name('rate_context.py'),
               Path(__file__).with_name('ready_rate_context.py'),*Path('future_vr').glob('*.py')]
    inputs = [selection]
    models, references = {},{}
    for seed in (11,22):
        folder = TRAINING/f'seed{seed}'
        assert (folder/'COMPLETE').exists()
        cp = folder/'selected.pt'
        inputs += [cp,folder/'development_raw.csv']
        saved = torch.load(cp,map_location='cpu',weights_only=True)
        model = ConstrainedDQN(cfg,217,seed)
        model.online.load_state_dict(saved['online'])
        model.online.eval()
        models[seed] = model
        references[seed] = {(int(r['scenario']),int(r['trajectory'])):r
                           for r in read(folder/'development_raw.csv')}
    manifest = dict(config=cfg.as_dict(),seeds=[11,22],trajectories_per_scenario=4,
        decision_slots=SLOTS,scenarios=SCENARIOS,development_stream=973000000,
        margins=margins,expected_pairs=96,
        scope='viewed development data; offline single-action diagnostic, not a deployable policy',
        future_access='exogenous traces used for scoring only; both actions use causal information',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows,episodes = [],[]
    max_replay_error = 0.
    for seed,model in models.items():
        for j,scenario in enumerate(SCENARIOS):
            for i in range(4):
                tr = trajectory(cfg,973000000+j*10000+i,scenario)
                base = AuditRollout(cfg,tr)
                factual_metrics = []
                while base.t < cfg.episode_steps:
                    if base.t % 4:
                        base.step()
                        continue
                    observation,score = base.observation()
                    with torch.no_grad():
                        values = model.online(torch.as_tensor(observation)[None])[0].numpy()
                    action = int(np.argmax(values))
                    count = cfg.users+1+action
                    command = select_top(score,count)
                    if base.t in SLOTS:
                        t = base.t
                        prs_command = rate_search_command(cfg,base.tracker,base.hw,
                                                         base.req*margins[j],score)
                        prs_count = int(prs_command.sum())
                        factual,alternative = base.clone(),base.clone()
                        a = factual.finish(model,command)
                        b = alternative.finish(model,prs_command)
                        factual_metrics.append(a)
                        delta_e,delta_o = b['energy_j']-a['energy_j'],b['outage']-a['outage']
                        first_e = sum(alternative.slot_energy[t:t+4])-sum(factual.slot_energy[t:t+4])
                        first_o = float(np.mean(alternative.outages[t:t+4])-
                                        np.mean(factual.outages[t:t+4]))
                        rows.append(dict(seed=seed,scenario=j,trajectory=i,slot=t,
                            rl_count=count,prs_count=prs_count,same_action=np.array_equal(command,prs_command),
                            rl_q=float(values[action]),prs_q=float(values[prs_count-cfg.users-1]),
                            learned_rl_preference=float(values[action]-values[prs_count-cfg.users-1]),
                            rl_energy_j=a['energy_j'],intervention_energy_j=b['energy_j'],
                            rl_outage=a['outage'],intervention_outage=b['outage'],
                            energy_delta_j=delta_e,outage_delta=delta_o,
                            immediate_energy_delta_j=first_e,immediate_outage_delta=first_o,
                            realized_return_delta=realized_return(alternative,t)-realized_return(factual,t),
                            dominance_on_realized_trace=bool(delta_e < -1e-8 and delta_o <= 1e-12),
                            **{f'q_count_{n}':float(values[n-cfg.users-1])
                               for n in range(cfg.users+1,cfg.subarrays+1)}))
                        if np.array_equal(command,prs_command):
                            assert a == b
                    base.step(command)
                actual = base.metrics()
                for metrics in factual_metrics:
                    assert actual == metrics, 'Factual replay differs from uninterrupted policy'
                for key,value in actual.items():
                    if key != 'method':
                        err = abs(value-float(references[seed][j,i][key]))
                        max_replay_error = max(max_replay_error,err)
                        assert err < 1e-9, (seed,j,i,key,err)
                episodes.append(dict(seed=seed,scenario=j,trajectory=i,**actual))
                _write_rows(ROOT/'pairs.csv',rows)
                _write_rows(ROOT/'factual_episodes.csv',episodes)
                print('intervention',seed,j,i,'pairs',len(rows),flush=True)
    summary = []
    for j in range(3):
        subset = [r for r in rows if r['scenario']==j]
        differing = [r for r in subset if not r['same_action']]
        summary.append(dict(scenario=j,paired_starts=len(subset),different_actions=len(differing),
            rl_more_count=sum(r['rl_count']>r['prs_count'] for r in subset),
            prs_action_realized_dominance=sum(r['dominance_on_realized_trace'] for r in subset),
            realized_return_better=sum(r['realized_return_delta']>1e-9 for r in differing),
            **{key:float(np.mean([r[key] for r in subset])) for key in
               ('energy_delta_j','outage_delta','immediate_energy_delta_j',
                'immediate_outage_delta','learned_rl_preference','realized_return_delta')}))
    keys = [(r['seed'],r['scenario'],r['trajectory'],r['slot']) for r in rows]
    assert len(keys)==len(set(keys))==96
    _write_rows(ROOT/'summary.csv',summary)
    audit = dict(pairs=len(rows),factual_episodes=len(episodes),max_reference_error=max_replay_error,
        unique_keys=len(set(keys)),sources_match=all(hashlib.sha256(p.read_bytes()).hexdigest()==
        manifest['sources'][str(p)] for p in sources),
        warning='Correlated starts; do not add intervention gains or report this as online-policy performance.')
    assert audit['sources_match']
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'COMPLETE').write_text('Frozen-policy intervention diagnostic complete; no training.\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    run()
