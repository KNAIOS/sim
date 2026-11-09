"""Independent antenna-domain, per-bin replay of logged powers and hardware."""
from collections import defaultdict

import numpy as np


def direct_rates(cfg, channel, power):
    noise = 10**((-174+10*np.log10(cfg.bandwidth_mhz*1e6)+cfg.noise_figure_db-30)/10)
    adjoint = channel.conj().swapaxes(-2, -1)
    beam = adjoint@np.linalg.inv(channel@adjoint+cfg.users*noise/cfg.tx_power_w*np.eye(cfg.users))
    beam /= np.maximum(np.linalg.norm(beam, axis=-2, keepdims=True), 1e-15)
    received = abs(channel@(beam*np.sqrt(power)[None, None]))**2
    desired = np.diagonal(received, axis1=-2, axis2=-1)
    interference = (received*(1-np.eye(cfg.users))).sum(-1)
    return cfg.bandwidth_mhz*(1-cfg.pilot_fraction)*np.log2(1+desired/(noise+interference)).mean(0)


def independent(cfg, records, physical_channel, requirements):
    active, countdown = np.ones(cfg.subarrays, bool), np.zeros(cfg.subarrays, int)
    errors = defaultdict(float)
    for t, row in enumerate(records):
        assert row['slot'] == t
        observed = int(active.sum())
        command = np.array([bool(row['command_bits'] & (1 << q)) for q in range(cfg.subarrays)])
        off = ~command & (active | (countdown > 0))
        on = command & ~active & (countdown == 0)
        active[off], countdown[off], countdown[on] = False, 0, cfg.wake_delay_slots
        assert row['active_bits'] == sum(int(b) << q for q, b in enumerate(active))
        power = np.array([row[f'power_{k}'] for k in range(cfg.users)])
        assert (power >= 0).all() and power.sum() <= cfg.tx_power_w+1e-10
        h = physical_channel[t][:, :, np.repeat(active, cfg.antennas_per_subarray)]
        rate = direct_rates(cfg, h, power)
        errors['rate_mbps'] = max(errors['rate_mbps'],
            float(abs(rate-[row[f'rate_{k}'] for k in range(cfg.users)]).max()))
        assert float(np.mean(rate < requirements[t])) == row['outage']
        na, nw = int(active.sum()), int((countdown > 0).sum())
        assert (na, nw, observed) == (row['active_count'], row['waking_count'], row['observed_count'])
        processing = (8*na*cfg.antennas_per_subarray*cfg.users**2+8*cfg.users**3/3)
        processing *= cfg.bandwidth_mhz*1e6/cfg.symbols_per_processing_block/(cfg.compute_efficiency_gflop_w*1e9)
        circuit = cfg.slot_s*(cfg.fixed_power_w+cfg.active_subarray_w*na+cfg.waking_subarray_w*nw
            +cfg.sleeping_subarray_w*(cfg.subarrays-na-nw)+processing)
        pa = cfg.slot_s*power.sum()/cfg.pa_efficiency
        switching = cfg.wake_energy_j*on.sum()+cfg.sleep_energy_j*off.sum()
        pilot = cfg.scan_energy_j_per_subarray*observed
        for name, value in (('circuit_j', circuit), ('pa_j', pa), ('switching_j', switching),
                            ('pilot_j', pilot), ('energy_j', circuit+pa+switching+pilot)):
            errors[name] = max(errors[name], abs(row[name]-value))
        ready = countdown == 1
        countdown[countdown > 0] -= 1
        active[ready] = True
    return dict(errors)
