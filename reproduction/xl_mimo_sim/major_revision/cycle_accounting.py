"""Exact hardware-cycle accounting; prescribed schedules, never an online policy."""
import numpy as np

from future_vr.system import HardwareState, XLSystem


def processing_per_module(cfg):
    return 8*cfg.antennas_per_subarray*cfg.users**2*(cfg.bandwidth_mhz*1e6/
        cfg.symbols_per_processing_block)/(cfg.compute_efficiency_gflop_w*1e9)


def cycle_difference(cfg, sleep_slots):
    """E(always active)-E(sleep then fully recover); positive favors cycling."""
    if sleep_slots < 1:
        raise ValueError('Positive integer sleep duration required')
    if int(sleep_slots) != sleep_slots:
        raise ValueError('Sleep duration must be integer slots')
    l, d, pc = sleep_slots, cfg.wake_delay_slots, processing_per_module(cfg)
    return (cfg.slot_s*(l*(cfg.active_subarray_w+pc-cfg.sleeping_subarray_w)
        + d*(cfg.active_subarray_w+pc-cfg.waking_subarray_w))
        + (l+d-1+int(d == 0))*cfg.scan_energy_j_per_subarray-cfg.sleep_energy_j-cfg.wake_energy_j)


def break_even_slots(cfg):
    """Real-valued algebraic crossing; negative does not imply negative time."""
    pc, d = processing_per_module(cfg), cfg.wake_delay_slots
    slope = cfg.slot_s*(cfg.active_subarray_w+pc-cfg.sleeping_subarray_w)+cfg.scan_energy_j_per_subarray
    if slope <= 0:
        raise ValueError('Positive active-versus-sleep slope required')
    intercept = (cfg.slot_s*d*(cfg.active_subarray_w+pc-cfg.waking_subarray_w)
        +(d-1+int(d == 0))*cfg.scan_energy_j_per_subarray-cfg.sleep_energy_j-cfg.wake_energy_j)
    return -intercept/slope


def replay(cfg, tr, decisions, block=4):
    commands = np.repeat(np.asarray(decisions, bool), block, axis=0)[:cfg.episode_steps]
    if commands.shape != (cfg.episode_steps, cfg.subarrays):
        raise ValueError('Decision schedule does not cover this episode')
    hw, system = HardwareState(cfg), XLSystem(cfg)
    values = {k: [] for k in ('pre_active', 'pre_countdown', 'pre_command',
        'active', 'waking', 'energy', 'outage', 'failures', 'switching', 'pilot')}
    for t, command in enumerate(commands):
        values['pre_active'].append(hw.active.copy())
        values['pre_countdown'].append(hw.countdown.copy())
        values['pre_command'].append(hw.command.copy())
        observed = int(hw.active.sum())
        active, switching, _ = hw.apply(command, 1.)
        values['active'].append(active.copy())
        values['waking'].append((hw.countdown > 0).copy())
        result = system.evaluate(tr.channel[t], active, hw.countdown > 0, switching,
            False, tr.requirements_mbps[t], observed_count=observed)
        values['energy'].append(result.energy_j)
        values['outage'].append(result.outage)
        values['failures'].append(result.rates_mbps < tr.requirements_mbps[t])
        values['switching'].append(switching)
        values['pilot'].append(cfg.scan_energy_j_per_subarray*observed)
        hw.tick()
    for k, value in (('pre_active', hw.active), ('pre_countdown', hw.countdown), ('pre_command', hw.command)):
        values[k].append(value.copy())
    return dict(commands=commands, **{k: np.array(v) for k, v in values.items()})


def completed_cycles(cfg, trace):
    """Enumerate every off-to-restored event; separately expose terminal censoring."""
    commands = trace['commands']
    complete, censored = [], []
    for q in range(cfg.subarrays):
        previous, off = True, None
        for t, value in enumerate(commands[:, q]):
            if previous and not value:
                if off is not None or not trace['pre_active'][t, q]:
                    raise ValueError('Aborted recovery or non-ready shutdown outside this identity')
                off = t
            elif not previous and value:
                if off is None:
                    raise ValueError('Missing shutdown event')
                ready = t+cfg.wake_delay_slots
                cycle = dict(module=q, off=off, on=t, ready=ready, sleep_slots=t-off)
                if ready > len(commands):
                    censored.append(dict(**cycle, reason='recovery_beyond_episode'))
                else:
                    restored = (trace['active'][ready, q] if cfg.wake_delay_slots == 0
                                else trace['pre_active'][ready, q])
                    if not np.all(commands[t:ready, q]) or not restored:
                        raise ValueError('Aborted recovery outside this identity')
                    complete.append(cycle)
                off = None
            previous = bool(value)
        if off is not None:
            censored.append(dict(module=q, off=off, on=-1, ready=-1,
                sleep_slots=len(commands)-off, reason='no_wake_before_end'))
    return complete, censored


def bridge(cfg, tr, trace, cycle):
    """Extra module on, fixed other availability; offline counterfactual only."""
    q, off, on, ready = (cycle[k] for k in ('module', 'off', 'on', 'ready'))
    energy, failures, system = 0., 0, XLSystem(cfg)
    for t in range(off, ready):
        active, waking = trace['active'][t].copy(), trace['waking'][t].copy()
        observed = trace['pre_active'][t].copy()
        active[q], waking[q], observed[q] = True, False, True
        switch = trace['switching'][t]
        if t == off:
            switch -= cfg.sleep_energy_j
        if t == on:
            switch -= cfg.wake_energy_j
        # At d=0 the cycle interval ends at 'on'. Its activation transition
        # still differs, and is accounted below without changing availability.
        result = system.evaluate(tr.channel[t], active, waking, switch, False,
            tr.requirements_mbps[t], observed_count=int(observed.sum()))
        energy += result.energy_j
        failures += int(np.sum(result.rates_mbps < tr.requirements_mbps[t]))
    difference = energy-float(trace['energy'][off:ready].sum())
    if cfg.wake_delay_slots == 0:
        difference -= cfg.wake_energy_j
        # At the immediate wake slot the original pilot was unobserved.
        difference += cfg.scan_energy_j_per_subarray
    original_failures = int(trace['failures'][off:ready].sum())
    return dict(energy_difference_j=difference, analytic_difference_j=cycle_difference(cfg, on-off),
        original_failed_user_slots=original_failures, bridge_failed_user_slots=failures,
        failed_user_slot_difference=failures-original_failures)


def scalar_bridge(cfg, tr, trace, cycle):
    """Independent HardwareState replay, including explicit recovery completion."""
    q, off, on, ready = (cycle[k] for k in ('module', 'off', 'on', 'ready'))
    hw, system = HardwareState(cfg), XLSystem(cfg)
    hw.active = trace['pre_active'][off].copy()
    hw.countdown = trace['pre_countdown'][off].copy()
    hw.command = trace['pre_command'][off].copy()
    energy, failed = 0., 0
    # With immediate recovery, include the wake slot to count its pilot/transition.
    end = ready+1 if cfg.wake_delay_slots == 0 else ready
    if end > cfg.episode_steps:
        raise ValueError('Scalar comparison requires its full accounting window')
    for t in range(off, end):
        command = trace['commands'][t].copy()
        command[q] = True
        observed = int(hw.active.sum())
        active, switching, _ = hw.apply(command, 1.)
        result = system.evaluate(tr.channel[t], active, hw.countdown > 0, switching,
            False, tr.requirements_mbps[t], observed_count=observed)
        energy += result.energy_j
        failed += int(np.sum(result.rates_mbps < tr.requirements_mbps[t]))
        hw.tick()
    np.testing.assert_array_equal(hw.active, trace['pre_active'][end])
    np.testing.assert_array_equal(hw.countdown, trace['pre_countdown'][end])
    np.testing.assert_array_equal(hw.command, trace['pre_command'][end])
    return dict(energy_difference_j=energy-float(trace['energy'][off:end].sum()),
        failed_user_slot_difference=failed-int(trace['failures'][off:end].sum()))
