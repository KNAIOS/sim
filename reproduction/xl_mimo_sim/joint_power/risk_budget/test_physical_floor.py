import copy
import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.risk_budget.physical_floor import failure_floor
from joint_power.study import make_trace


class PhysicalFloorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(make_config('validation'), episode_steps=20)
        self.trace = make_trace(self.cfg, 4299220003, (2., 1.))

    def test_known_current_and_blackout_failures_bound_actual_block(self):
        env = Environment(self.cfg, self.trace)
        env.block(np.arange(16) < 2)
        candidates = env.candidates()
        empty_current = np.arange(16) >= 2
        masks = np.vstack((candidates.masks, empty_current))
        bound = failure_floor(env, masks)
        self.assertEqual(bound[-1], 12)
        for mask, lower in zip(masks, bound):
            branch = copy.deepcopy(env)
            start = len(branch.records)
            branch.block(mask)
            actual = sum(round(3*r['outage']) for r in branch.records[start:])
            self.assertLessEqual(lower, actual)
            if lower < 12:
                self.assertGreaterEqual(lower, round(3*branch.records[start]['outage']))

    def test_hidden_current_and_future_csi_cannot_change_floor(self):
        env = Environment(self.cfg, self.trace)
        env.block(np.arange(16) < 2)
        changed = copy.deepcopy(env)
        channel, gain = self.trace.channel.copy(), self.trace.gain_db.copy()
        channel[env.t+1:] *= 1000
        channel[env.t, :, np.repeat(~env.hw.active, 8)] *= 1000
        gain[env.t+1:] += 60
        gain[env.t, ~env.hw.active] += 60
        changed.tr = replace(self.trace, channel=channel, gain_db=gain)
        masks = env.candidates().masks
        np.testing.assert_array_equal(failure_floor(env, masks), failure_floor(changed, masks))


if __name__ == '__main__':
    unittest.main()
