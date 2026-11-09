"""DQN-only episode adapter for a controlled readiness-feature ablation.

Physical dynamics and accounting reproduce pilot.episode. The separate adapter
keeps all previously hashed experiment sources intact; tests compare full replay
transitions with the old engine when the added feature is disabled.
"""
import numpy as np
from future_vr.belief import VisibilityBelief
from future_vr.controllers import Transition, select_top
from future_vr.system import HardwareState, XLSystem
from major_revision.pilot import state
from major_revision.rate_context import candidate_rate_context
from major_revision.ready_rate_context import ready_rate_context


def ready_episode(cfg, tr, controller, training=False, epsilon=0., include_ready=True):
    system, hw = XLSystem(cfg), HardwareState(cfg)
    tracker = VisibilityBelief(cfg, tr.gain_db[0])
    tracker.channel_estimate = tr.channel[0].copy()
    age = np.zeros(cfg.subarrays)
    last_off = np.full(cfg.subarrays, -10000)
    previous_req = tr.requirements_mbps[0].copy()
    last_outage, pending = 0., None
    energy = switching = pilot = 0.
    outages, counts, transitions, churn = [], [], 0, 0
    full_energy = system.evaluate(tr.channel[0], np.ones(cfg.subarrays,bool),
        np.zeros(cfg.subarrays,bool), 0., False, previous_req,
        observed_count=cfg.subarrays).energy_j
    for t in range(cfg.episode_steps):
        observed = hw.active.copy()
        age = np.where(observed, 0., age+1)
        tracker.update(tr.gain_db[t], observed)
        antennas = np.repeat(observed, cfg.antennas_per_subarray)
        tracker.channel_estimate[:,antennas] = tr.channel[t][:,antennas]
        req = tr.requirements_mbps[t]
        action = None
        if t % 4 == 0:
            base, score = state(cfg, tracker, hw, req, previous_req, age, last_outage, True)
            features = [base, candidate_rate_context(cfg,tracker,score,req)]
            if include_ready:
                features.append(ready_rate_context(cfg,tracker,hw,score,req))
            observation = np.concatenate(features).astype(np.float32)
            if training and pending is not None:
                old_state, old_action, e, c, n = pending
                controller.remember(Transition(old_state,old_action,e/n,c/n,observation,False))
                controller.learn()
                pending = None
            action, count = controller.choose(observation, epsilon if training else 0.)
            command = select_top(score, count)
            previous_req = req.copy()
        else:
            command = hw.command.copy()
        changed_on = command & ~hw.command
        changed_off = ~command & hw.command
        churn += int(np.sum(changed_on & ((t-last_off) <= cfg.horizon)))
        last_off[changed_off] = t
        active, transition_energy, number_changed = hw.apply(command, 1.)
        result = system.evaluate(tr.channel[t],active,hw.countdown>0,transition_energy,
                                 False,req,observed_count=int(observed.sum()))
        hw.tick()
        energy += result.energy_j
        switching += transition_energy
        pilot += cfg.scan_energy_j_per_subarray*observed.sum()
        outages.append(result.outage)
        counts.append(command.sum())
        transitions += number_changed
        if training:
            if action is not None:
                pending = (observation,action,result.energy_j/full_energy,result.outage,1)
            else:
                old_state,old_action,e,c,n = pending
                pending = (old_state,old_action,e+result.energy_j/full_energy,c+result.outage,n+1)
        last_outage = result.outage
    if training and pending is not None:
        old_state,old_action,e,c,n = pending
        controller.remember(Transition(old_state,old_action,e/n,c/n,old_state,True))
        controller.learn()
    return dict(method='Enriched-DQN',energy_j=energy,outage=float(np.mean(outages)),
        switching_j=switching,pilot_j=float(pilot),operating_j=energy-switching-pilot,
        transitions=transitions,short_sleep_reactivations=churn,mean_count=float(np.mean(counts)))
