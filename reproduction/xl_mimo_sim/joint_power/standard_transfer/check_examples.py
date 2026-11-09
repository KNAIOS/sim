"""Pre-existing UMi fixtures: hardware/floor/PHY checks, not policy results."""
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from future_vr.channel import Trajectory
from future_vr.config import make_config
from joint_power.standard_transfer.environment import WidebandEnvironment, failure_floor
from joint_power.standard_transfer.verify import independent
from joint_power.study import digest, write_rows


def main():
    source = Path('../standard_channel_20260920/results/manuscript_validation_v1')
    output = Path('results/joint_wideband_environment_checks_20260922')
    output.mkdir(exist_ok=False)
    cfg = replace(make_config('validation'), array_aperture_m=15*.5*299792458/14e9)
    paths = [source/'LOS_example.npz', source/'NLOS_example.npz',
             *Path('joint_power/standard_transfer').glob('*.py'),
             Path('joint_power/standard_transfer/README.md'), Path('joint_power/wideband_power.py'),
             Path('joint_power/model.py'), Path('future_vr/system.py')]
    hashes = {str(p): digest(p) for p in paths}
    (output/'manifest.json').write_text(json.dumps(dict(input_hashes=hashes,
        scope='Previously seen numerical fixtures, fixed schedules, no policy performance claim',
        config=cfg.as_dict(), host_environment_changed=False), indent=2)+'\n')
    errors, modes, slots = defaultdict(float), Counter(), []
    for state in ('LOS', 'NLOS'):
        with np.load(source/f'{state}_example.npz') as payload:
            h, demands = payload['channel'], payload['requirements_mbps']
            xy, velocity = payload['ut_positions_m'][:, :2], payload['ut_velocity_mps'][:, :2]
        central = h[:, h.shape[1]//2]
        gain = 10*np.log10(np.maximum((abs(central.reshape(140, 3, 16, 8))**2).mean(3), 1e-30)).transpose(0, 2, 1)
        for load in (.5, 1.):
            tr = Trajectory(central, gain, np.ones_like(gain),
                np.repeat(xy[None], 140, axis=0), np.repeat(velocity[None], 140, axis=0), demands*load)
            env = WidebandEnvironment(cfg, tr, h)
            commands = (np.arange(16) < 4, np.arange(16) >= 12, np.ones(16, bool))
            while env.t < cfg.episode_steps:
                command = commands[(env.t//4) % 3]
                bound = failure_floor(env, command[None])[0]
                start = len(env.records)
                env.block(command)
                assert bound <= sum(round(3*r['outage']) for r in env.records[start:])
            for key, error in independent(cfg, env.records, h, tr.requirements_mbps).items():
                errors[key] = max(errors[key], error)
            slots.extend(dict(state=state, load=load, **r) for r in env.records)
            modes.update(r['allocator_mode'] for r in env.records)
    assert len(slots) == 560 and errors['rate_mbps'] < 1e-6
    assert max(v for k, v in errors.items() if k != 'rate_mbps') < 1e-8
    write_rows(output/'fixed_schedule_slots.csv', slots)
    assert all(digest(p) == h for p, h in hashes.items())
    report = dict(fixed_schedule_episodes=4, verified_slots=560, frequency_bins=99,
        maximum_errors=dict(errors), allocator_modes=dict(modes), causal_lower_bound_checks=140,
        fresh_test_data=False, learned_policy_evaluated=False, R3_closed=False,
        scope='Numerical/causal adapter audit only; not standardized-channel policy evidence')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'output_hashes.json').write_text(json.dumps({str(p): digest(p) for p in output.iterdir()
        if p.is_file()}, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
