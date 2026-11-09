"""Numerical checks on cached UMi examples, not new policy performance data."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from future_vr.config import make_config
from joint_power.model import effective_gains
from joint_power.study import digest, write_rows
from joint_power.wideband_power import allocate


def direct_rates(cfg, channel, power):
    noise = 10**((-174+10*np.log10(40e6)+7-30)/10)
    beam = channel.conj().swapaxes(-2, -1) @ np.linalg.inv(
        channel@channel.conj().swapaxes(-2, -1)+3*noise/1.2*np.eye(3))
    beam /= np.maximum(np.linalg.norm(beam, axis=-2, keepdims=True), 1e-15)
    received = abs(channel@(beam*np.sqrt(power)[None, None]))**2
    desired = np.diagonal(received, axis1=-2, axis2=-1)
    interference = (received*(1-np.eye(3))).sum(-1)
    return 39.4*np.log2(1+desired/(noise+interference)).mean(0)


def main():
    cfg = make_config('validation')
    source = Path('../standard_channel_20260920/results/manuscript_validation_v1')
    output = Path('results/joint_wideband_kernel_checks_20260922')
    output.mkdir(parents=True, exist_ok=False)
    files = [source/'LOS_example.npz', source/'NLOS_example.npz',
             Path('joint_power/wideband_power.py'), Path(__file__)]
    hashes = {str(p): digest(p) for p in files}
    rows, max_error = [], 0.
    for label in ('LOS', 'NLOS'):
        with np.load(source/f'{label}_example.npz') as payload:
            channels = payload['channel'][[0, 53]]
            demands = payload['requirements_mbps'][[0, 53]]
        for index, slot in enumerate((0, 53)):
            for count in (4, 8, 16):
                h = channels[index, :, :, :count*8]
                gain = effective_gains(cfg, h@h.conj().swapaxes(-2, -1))
                before = direct_rates(cfg, h, np.full(3, .4))
                for load in (.5, 1.):
                    target = demands[index]*load
                    power, calculated, metadata = allocate(cfg, gain, target)
                    independent = direct_rates(cfg, h, power)
                    error = float(abs(calculated-independent).max())
                    max_error = max(max_error, error)
                    assert error < 1e-6
                    assert (power >= 0).all() and power.sum() <= 1.2+1e-10
                    assert np.all(independent >= np.minimum(before, target)-1e-7)
                    rows.append(dict(state=label, slot=slot, subarrays=count, load=load,
                        power_w=float(power.sum()), rate_error_mbps=error,
                        achieved_outage=float(np.mean(independent < target)),
                        equal_power_outage=float(np.mean(before < target)),
                        mode=metadata['mode'], iterations=metadata['iterations']))
    write_rows(output/'cases.csv', rows)
    assert all(digest(p) == h for p, h in hashes.items())
    report = dict(cases=len(rows), frequency_bins=99, maximum_rate_error_mbps=max_error,
        modes=dict(Counter(r['mode'] for r in rows)), input_hashes=hashes,
        used_for_training=False, scope='Numerical consistency on previously seen UMi examples; not new R3 policy evidence')
    (output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
