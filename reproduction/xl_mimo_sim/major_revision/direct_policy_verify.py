"""Independent reconstruction of optimizer updates and validation accounting."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from future_vr.config import make_config
from major_revision.absolute_return_screen import episode as search_episode
from major_revision.action_value import full_slot_energy
from major_revision.direct_policy import ROOT, VAL_STREAM, initialize, finish, verify, episode
from major_revision.pilot import SCENARIOS, trajectory


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        result = list(csv.DictReader(f))
    return [{k: v if k in ('method','variant') else (None if v=='' else float(v))
             for k,v in r.items()} for r in result]


def independent_frontier(points, target):
    # Recompute finite two-policy interpolation without production helper.
    candidates = [(p['energy_j'],p['outage']) for p in points if p['outage']<=target+1e-12]
    for a in range(len(points)):
        for b in range(a+1,len(points)):
            x,y = points[a],points[b]
            if not min(x['outage'],y['outage']) < target < max(x['outage'],y['outage']):
                continue
            fraction = (target-y['outage'])/(x['outage']-y['outage'])
            candidates.append((fraction*x['energy_j']+(1-fraction)*y['energy_j'],target))
    return min(candidates) if candidates else None


def main():
    cfg = make_config('validation')
    full_energy = cfg.episode_steps*full_slot_energy(cfg)
    errors = dict(reward=0.,update=0.,summary=0.,frontier=0.,paired=0.,replay=0.)
    for seed in (11,22):
        source = ROOT/f'train_seed{seed}'
        verify(source)
        raw = rows(source/'training_raw.csv')
        assert len(raw)==1152
        assert len({(r['variant'],r['iteration'],r['direction'],r['sign'],r['scenario'],r['trajectory']) for r in raw})==1152
        for variant, width in (('state',9),('bias',1)):
            with np.load(source/f'{variant}_history.npz',allow_pickle=False) as z:
                weights,directions,rewards = (z[k].copy() for k in ('weights','directions','rewards'))
            assert weights.shape==(13,width) and rewards.shape==(12,4,2)
            np.testing.assert_array_equal(weights[0],np.zeros(width))
            rng = np.random.default_rng(seed)
            for iteration in range(12):
                np.testing.assert_array_equal(directions[iteration],rng.normal(size=(4,width)))
                reconstructed = np.zeros((4,2))
                for k in range(4):
                    for col, sign in enumerate((1,-1)):
                        batch = [r for r in raw if r['variant']==variant and r['iteration']==iteration
                                 and r['direction']==k and r['sign']==sign]
                        assert len(batch)==6
                        assert {(r['scenario'],r['trajectory']) for r in batch} == {
                            (j,2*iteration+i) for j in range(3) for i in range(2)}
                        direct = np.array([-r['energy_j']/full_energy-4*r['outage'] for r in batch])
                        errors['reward'] = max(errors['reward'],float(np.max(np.abs(direct-[r['reward'] for r in batch]))))
                        reconstructed[k,col] = direct.mean()
                errors['reward'] = max(errors['reward'],float(np.max(np.abs(reconstructed-rewards[iteration]))))
                chosen = sorted(range(4),key=lambda k:(-max(rewards[iteration,k]),k))[:2]
                sd = float(np.std(rewards[iteration,chosen]))
                update = np.zeros(width)
                if sd>1e-12:
                    for k in chosen:
                        update += .04/(2*sd)*(rewards[iteration,k,0]-rewards[iteration,k,1])*directions[iteration,k]
                errors['update'] = max(errors['update'],float(np.max(np.abs(weights[iteration]+update-weights[iteration+1]))))
    for part in ('validation_learned','validation_baselines','audit'):
        verify(ROOT/part)
    learned = rows(ROOT/'validation_learned/raw.csv')
    search = rows(ROOT/'validation_baselines/raw.csv')
    summary = rows(ROOT/'audit/summary.csv')
    selected = rows(ROOT/'audit/selected.csv')
    comparisons = rows(ROOT/'audit/comparisons.csv')
    paired = rows(ROOT/'audit/paired.csv')
    assert len(summary)==291 and len(selected)==12 and len(comparisons)==6
    for p in summary:
        source = learned if p['seed'] else search
        group = [r for r in source if r['method']==p['method'] and r['scenario']==p['scenario'] and
                 r['margin']==p['margin'] and (r['seed']==p['seed'] if p['seed'] else r['minimum']==p['minimum'])]
        assert len(group)==8
        for key in ('energy_j','outage'):
            errors['summary'] = max(errors['summary'],abs(p[key]-np.mean([r[key] for r in group])))
    for p in selected:
        choices = [r for r in summary if (r['method'],r['seed'],r['scenario'])==(p['method'],p['seed'],p['scenario'])]
        feasible = [r for r in choices if r['outage']<=.05]
        if feasible:
            choice = sorted(feasible,key=lambda r:(r['energy_j'],r['outage'],r['margin']))[0]
        else:
            choice = sorted(choices,key=lambda r:(r['outage'],r['energy_j'],r['margin']))[0]
        assert p==choice
    for p in comparisons:
        j,seed = p['scenario'],p['seed']
        for kind in ('search','bias'):
            points = [r for r in summary if r['scenario']==j and
                      (r['seed']==0 if kind=='search' else ((r['method']=='bias' and r['seed']==seed) or r['method']=='full'))]
            result = independent_frontier(points,p['outage'])
            value = p[f'matched_{kind}_energy_j']
            assert (result is None)==(value is None)
            if result is not None:
                errors['frontier'] = max(errors['frontier'],abs(result[0]-value))
        group = [r for r in paired if r['scenario']==j and r['seed']==seed]
        if group:
            assert len(group)==8
            differences = np.array([r['energy_j']-r['expected_search_j'] for r in group])
            errors['paired'] = max(errors['paired'],abs(p['difference_j']-differences.mean()),
                                   abs(p['paired_sd_j']-differences.std(ddof=1)))
    # Replay every selected learned setting on the first trajectory, not cherry-
    # picked good cases. Also replay every scenario's best feasible pure search.
    for p in selected:
        j,seed = int(p['scenario']),int(p['seed'])
        with np.load(ROOT/f'train_seed{seed}/{p["method"]}_history.npz',allow_pickle=False) as z:
            theta = z['weights'][-1].copy()
        actual = episode(cfg,trajectory(cfg,VAL_STREAM+j*10000,SCENARIOS[j]),theta,p['margin'])
        stored = next(r for r in learned if r['scenario']==j and r['trajectory']==0 and r['seed']==seed
                      and r['method']==p['method'] and r['margin']==p['margin'])
        for key in ('energy_j','outage','switching_j','pilot_j','operating_j','transitions','mean_count','mean_margin'):
            errors['replay'] = max(errors['replay'],abs(actual[key]-stored[key]))
    for j in range(3):
        feasible = [p for p in summary if p['scenario']==j and p['seed']==0 and p['method']!='full' and p['outage']<=.05]
        if not feasible:
            continue
        p = min(feasible,key=lambda r:(r['energy_j'],r['outage']))
        actual = search_episode(cfg,trajectory(cfg,VAL_STREAM+j*10000,SCENARIOS[j]),p['method'],
                                dict(margin=p['margin'],minimum=int(p['minimum'])))
        stored = next(r for r in search if r['scenario']==j and r['trajectory']==0 and
                      (r['method'],r['minimum'],r['margin'])==(p['method'],p['minimum'],p['margin']))
        for key in ('energy_j','outage','switching_j','pilot_j','operating_j','transitions','mean_count'):
            errors['replay'] = max(errors['replay'],abs(actual[key]-stored[key]))
    assert max(errors.values())<1e-10,errors
    gate = json.loads((ROOT/'audit/gate.json').read_text())
    independent_gates = []
    for p in comparisons:
        if p['scenario']!=1:
            continue
        all_ok = all(r['outage']<=.05 for r in selected if r['seed']==p['seed'] and r['method']=='state')
        expected = (all_ok and p['pure_energy_j'] is not None and p['matched_search_energy_j'] is not None
                    and p['matched_bias_energy_j'] is not None and p['outage']<=.05
                    and p['energy_j']<=.98*p['pure_energy_j'] and p['outage']<=p['pure_outage']+.005
                    and p['energy_j']<=.98*p['matched_search_energy_j']
                    and p['energy_j']<p['matched_bias_energy_j'])
        original = next(r for r in gate['rows'] if r['scenario']==1 and r['seed']==p['seed'])
        assert bool(expected)==original['passed']
        independent_gates.append(bool(expected))
    assert all(independent_gates)==gate['permit_confirmation_protocol']
    files = [p for folder in ('train_seed11','train_seed22','validation_learned','validation_baselines','audit')
             for p in (ROOT/folder).iterdir() if p.is_file()]
    folder,hashes = initialize('verified',files)
    report = dict(errors=errors,training_episodes=2304,training_external_trajectories=72,
        validation_episodes=2328,validation_external_trajectories=24,replayed_selected_settings=15,
        optimizer_updates_reconstructed=48,summary_points_verified=291,
        independent_confirmation=False,scientific_completion=False,
        permit_confirmation_protocol=all(independent_gates))
    (folder/'audit.json').write_text(json.dumps(report,indent=2))
    finish(folder,hashes)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
