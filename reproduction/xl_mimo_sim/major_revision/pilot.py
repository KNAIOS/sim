"""Development pilot: richer causal state versus switch-aware search.

No changes to the channel, power, reward, or original methods. Development data
are not publication test data. All search methods retain active-only CSI.
"""
import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from future_vr.belief import VisibilityBelief
from future_vr.channel import generate_trajectory
from future_vr.config import make_config
from future_vr.controllers import ConstrainedDQN, Transition, build_state, select_top
from future_vr.experiment import _prediction, _write_rows, simulate_episode
from future_vr.rate_search import rate_search_command
from future_vr.system import HardwareState, XLSystem

SCENARIOS = ((.5,.85), (2.,1.), (3.,1.15))


def state(cfg, tracker, hw, req, prev_req, age, last_outage, enriched):
    score, _ = _prediction(cfg, tracker, hw, req)
    base = build_state(cfg, score, hw.active, hw.command, hw.readiness(),
                       req, tracker.velocity, last_outage)
    if not enriched:
        return base, score
    noise_db = -174 + 10*np.log10(cfg.bandwidth_mhz*1e6) + cfg.noise_figure_db - 30
    # Absolute per-branch gain relative to thermal noise; never future true CSI.
    snr_proxy = np.clip((tracker.gain_db-noise_db)/20, -4, 4).ravel()
    extra = np.concatenate((snr_proxy, np.minimum(age, 4*cfg.horizon)/(4*cfg.horizon),
                            (req-prev_req)/max(cfg.demand_levels_mbps)))
    return np.concatenate((base, extra)).astype(np.float32), score


def cost_search(cfg, tracker, hw, req, score):
    """Explicit hold-command rollout including pilots, transition and RF energy.

    Forecasts cover t,...,t+H-1 so the current unavailable wake slot is included.
    Same 13 ranking prefixes as RL, plus retaining the existing mask.
    """
    gains, _, _ = tracker.predict(cfg.horizon)
    gains = np.concatenate((tracker.gain_db[None], gains[:-1]))
    estimate = tracker.channel_estimate
    power = np.mean(abs(estimate.reshape(cfg.users,cfg.subarrays,cfg.antennas_per_subarray))**2, axis=2).T
    ratio = np.sqrt(10**(gains/10)/np.maximum(power[None],1e-20))
    forecasts = estimate[None]*np.repeat(ratio.transpose(0,2,1),cfg.antennas_per_subarray,axis=2)
    candidates = [select_top(score,n) for n in range(cfg.users+1,cfg.subarrays+1)] + [hw.command.copy()]
    system = XLSystem(cfg)
    results = []
    for mask in candidates:
        trial = copy.deepcopy(hw)
        energy, outage, shortfall = 0., 0., 0.
        for h in range(cfg.horizon):
            observed = int(trial.active.sum())
            active, transition, _ = trial.apply(mask,1.)
            result = system.evaluate(forecasts[h], active, trial.countdown>0, transition,
                                     False,req,observed_count=observed)
            energy += result.energy_j
            outage += result.outage/cfg.horizon
            shortfall += result.shortfall/cfg.horizon
            trial.tick()
        results.append(((outage>cfg.outage_target,
                         energy if outage<=cfg.outage_target else shortfall, energy),mask))
    return min(results,key=lambda x:x[0])[1]


def episode(cfg,tr,method,controller=None,training=False,epsilon=0.,rate_margin=1.,
            physical_channel=None,physical_system=None,state_features=None):
    if not np.isfinite(rate_margin) or rate_margin < 1:
        raise ValueError('rate_margin must be finite and at least one')
    if rate_margin != 1 and method not in ('PRS','Hysteresis PRS','Switch-aware PRS'):
        raise ValueError('rate_margin is a search-policy parameter only')
    system = XLSystem(cfg) if physical_system is None else physical_system
    physical_channel = tr.channel if physical_channel is None else physical_channel
    if len(physical_channel) != cfg.episode_steps:
        raise ValueError('Physical channel length must match episode')
    hw = HardwareState(cfg)
    tracker = VisibilityBelief(cfg,tr.gain_db[0])
    tracker.channel_estimate = tr.channel[0].copy()
    age = np.zeros(cfg.subarrays)
    on_age = np.full(cfg.subarrays,float(cfg.horizon))
    last_off = np.full(cfg.subarrays,-10000)
    prev_req = tr.requirements_mbps[0].copy()
    last_outage, pending = 0., None
    energy = switching = pilot = 0.
    outages, counts, transitions, churn = [], [], 0, 0
    full_e = system.evaluate(physical_channel[0],np.ones(cfg.subarrays,bool),
                            np.zeros(cfg.subarrays,bool),0.,False,prev_req,
                            observed_count=cfg.subarrays).energy_j
    for t in range(cfg.episode_steps):
        observed = hw.active.copy()
        age = np.where(observed,0.,age+1)
        tracker.update(tr.gain_db[t],observed)
        antennas = np.repeat(observed,cfg.antennas_per_subarray)
        tracker.channel_estimate[:,antennas] = tr.channel[t][:,antennas]
        req = tr.requirements_mbps[t]
        action = None
        if t%4 == 0:
            s, score = state(cfg,tracker,hw,req,prev_req,age,last_outage,method=='Enriched-DQN')
            if state_features is not None:
                s=np.concatenate((s,state_features(cfg,tracker,score,req))).astype(np.float32)
            if training and pending is not None:
                ps,pa,pe,po,n = pending
                controller.remember(Transition(ps,pa,pe/n,po/n,s,False))
                controller.learn()
                pending = None
            if method in ('Original-DQN','Enriched-DQN'):
                action,count = controller.choose(s,epsilon if training else 0.)
                command = select_top(score,count)
            elif method == 'Switch-aware PRS':
                command = cost_search(cfg,tracker,hw,req*rate_margin,score)
            elif method in ('PRS','Hysteresis PRS'):
                command = rate_search_command(cfg,tracker,hw,req*rate_margin,score)
                if method == 'Hysteresis PRS':
                    # Allow activation immediately; hold new commands for H physical slots.
                    command |= hw.command & (on_age<cfg.horizon)
            elif method == 'Full array':
                command = np.ones(cfg.subarrays,bool)
            else:
                raise ValueError(method)
            prev_req = req.copy()
        else:
            command = hw.command.copy()
        changed_on = command & ~hw.command
        changed_off = ~command & hw.command
        churn += int(np.sum(changed_on & ((t-last_off)<=cfg.horizon)))
        last_off[changed_off] = t
        on_age = np.where(changed_on,0,on_age)
        active, trans_e, ntrans = hw.apply(command,1.)
        result = system.evaluate(physical_channel[t],active,hw.countdown>0,trans_e,False,
                                 req,observed_count=int(observed.sum()))
        hw.tick()
        on_age += 1
        energy += result.energy_j
        switching += trans_e
        pilot += cfg.scan_energy_j_per_subarray*observed.sum()
        outages.append(result.outage)
        counts.append(command.sum())
        transitions += ntrans
        if training:
            if action is not None:
                pending = (s,action,result.energy_j/full_e,result.outage,1)
            else:
                ps,pa,pe,po,n = pending
                pending = (ps,pa,pe+result.energy_j/full_e,po+result.outage,n+1)
        last_outage = result.outage
    if training and pending is not None:
        ps,pa,pe,po,n = pending
        controller.remember(Transition(ps,pa,pe/n,po/n,ps,True))
        controller.learn()
    return dict(method=method,energy_j=energy,outage=float(np.mean(outages)),
                switching_j=switching,pilot_j=float(pilot),
                operating_j=energy-switching-pilot,transitions=transitions,
                short_sleep_reactivations=churn,mean_count=float(np.mean(counts)))


def trajectory(cfg,seed,scenario):
    speed,load = scenario
    tr = generate_trajectory(cfg,seed,speed)
    return replace(tr,requirements_mbps=tr.requirements_mbps*load)


def run(args):
    torch.set_num_threads(1)
    cfg = replace(make_config('validation'),episode_steps=args.slots)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    manifest = dict(config=cfg.as_dict(),episodes=args.episodes,seeds=args.seeds,
                    scenarios=SCENARIOS,validation_per_scenario=3,development_per_scenario=args.trajectories,
                    train_stream=921000000,validation_stream=922000000,development_stream=923000000,
                    publication_test='NOT USED: all outcomes are development-only',
                    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__),*Path('future_vr').glob('*.py')]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    val = [(scenario,trajectory(cfg,922000000+j*100+i,scenario))
           for j,scenario in enumerate(SCENARIOS) for i in range(3)]
    controllers = {}
    histories, selections = [], []
    for method in ('Original-DQN','Enriched-DQN'):
        for seed in args.seeds:
            dim = 4*cfg.subarrays+2*cfg.users+2
            if method == 'Enriched-DQN':
                dim += cfg.subarrays*cfg.users+cfg.subarrays+cfg.users
            controller = ConstrainedDQN(cfg,dim,seed)
            rng = np.random.default_rng(seed)
            bestkey, best, window = (float('inf'),)*2, None, []
            for ep in range(args.episodes):
                sc = SCENARIOS[int(rng.integers(len(SCENARIOS)))]
                tr = trajectory(cfg,921000000+seed*10000+ep,sc)
                eps = .04+.76*np.exp(-4*ep/max(args.episodes-1,1))
                result = episode(cfg,tr,method,controller,True,eps)
                window.append(result['outage'])
                histories.append(dict(seed=seed,episode=ep+1,dual=controller.dual,**result))
                if len(window)==10:
                    controller.dual=float(np.clip(controller.dual+50*(np.mean(window)-.05),0,80))
                    window=[]
                if (ep+1)%20==0 or ep+1==args.episodes:
                    metrics = [episode(cfg,tr,method,controller) for _,tr in val]
                    worst = max(np.mean([m['outage'] for m in metrics[j*3:j*3+3]]) for j in range(3))
                    avg_energy = float(np.mean([m['energy_j'] for m in metrics]))
                    # Prefer feasible checkpoints; otherwise rank worst-scenario violation.
                    key=(max(0.,worst-.05),avg_energy)
                    if key<bestkey:
                        bestkey=key
                        best=copy.deepcopy(controller.online.state_dict())
                        selected=ep+1
                    selections.append(dict(method=method,seed=seed,episode=ep+1,worst_outage=worst,energy_j=avg_energy))
                    print(method,seed,ep+1,'validation worst outage',round(worst,4),flush=True)
            controller.online.load_state_dict(best)
            controllers[method,seed]=controller
            torch.save(dict(online=best,selected_episode=selected,validation_key=bestkey,
                            state_dimension=dim),out/f'{method}_{seed}.pt')
            _write_rows(out/'training.csv',histories)
            _write_rows(out/'validation.csv',selections)
    rows=[]
    for j,sc in enumerate(SCENARIOS):
        for i in range(args.trajectories):
            tr=trajectory(cfg,923000000+j*100+i,sc)
            specs=[(m,s,c) for (m,s),c in controllers.items()]
            specs += [(m,0,None) for m in ('PRS','Hysteresis PRS','Switch-aware PRS','Full array')]
            for method,seed,c in specs:
                r=episode(cfg,tr,method,c)
                rows.append(dict(scenario=j,speed=sc[0],load=sc[1],trajectory=i,seed=seed,**r))
            _write_rows(out/'development_raw.csv',rows)
            print('development',j,i+1,'/',args.trajectories,flush=True)
    summary=[]
    for j in range(len(SCENARIOS)):
        for method in sorted({r['method'] for r in rows}):
            group=[r for r in rows if r['scenario']==j and r['method']==method]
            summary.append(dict(scenario=j,method=method,**{k:float(np.mean([r[k] for r in group]))
                for k in ('energy_j','outage','switching_j','operating_j','pilot_j','short_sleep_reactivations')}))
    _write_rows(out/'development_summary.csv',summary)
    (out/'COMPLETE').write_text('Development pilot completed; not publication evidence.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=80)
    p.add_argument('--slots',type=int,default=140)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22])
    p.add_argument('--trajectories',type=int,default=8)
    run(p.parse_args())
