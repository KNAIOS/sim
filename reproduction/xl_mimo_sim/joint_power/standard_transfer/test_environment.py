"""Synthetic unit fixtures, never reported as experimental performance."""
import copy
import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from joint_power.model import Environment, effective_gains, rates
from joint_power.risk_budget.physical_floor import failure_floor as narrow_floor
from joint_power.standard_transfer.environment import WidebandEnvironment, failure_floor
from joint_power.standard_transfer.verify import independent
from joint_power.study import make_trace


class WidebandEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(make_config('validation'), episode_steps=12)
        cls.trace = make_trace(cls.cfg, 5899220001, (2., 1.))
        cls.physical = cls.trace.channel[:, None]*np.array([.05, 1., 2.5])[None, :, None, None]

    def test_flat_channel_matches_narrowband_schedule(self):
        flat = np.repeat(self.trace.channel[:, None], 3, axis=1)
        narrow = Environment(self.cfg, self.trace)
        wide = WidebandEnvironment(self.cfg, self.trace, flat)
        for command in (np.arange(16) < 4, np.arange(16) >= 12, np.ones(16, bool)):
            narrow.block(command)
            wide.block(command)
        for a, b in zip(narrow.records, wide.records):
            for key in ('command_bits', 'active_bits', 'active_count', 'waking_count', 'outage'):
                self.assertEqual(a[key], b[key])
            for key in ('energy_j', 'circuit_j', 'pa_j', 'switching_j', 'pilot_j'):
                self.assertAlmostEqual(a[key], b[key], places=7)
            np.testing.assert_allclose([a[f'rate_{k}'] for k in range(3)],
                                       [b[f'rate_{k}'] for k in range(3)], atol=1e-5)

    def test_wideband_floor_bounds_every_executed_block(self):
        env = WidebandEnvironment(self.cfg, self.trace, self.physical)
        env.block(np.arange(16) < 2)
        masks = np.vstack((env.candidates().masks, np.arange(16) >= 2))
        lower = failure_floor(env, masks)
        self.assertEqual(lower[-1], 12)
        for mask, bound in zip(masks, lower):
            branch = copy.deepcopy(env)
            start = len(branch.records)
            branch.block(mask)
            self.assertLessEqual(bound, sum(round(3*r['outage']) for r in branch.records[start:]))

    def test_hidden_current_and_future_bins_do_not_change_floor(self):
        env = WidebandEnvironment(self.cfg, self.trace, self.physical.copy())
        env.block(np.arange(16) < 2)
        other = copy.deepcopy(env)
        other.physical[env.t+1:] *= 1000
        other.physical[env.t][:, :, np.repeat(~env.hw.active, 8)] *= 1000
        masks = env.candidates().masks
        np.testing.assert_array_equal(failure_floor(env, masks), failure_floor(other, masks))

    def test_centre_frequency_failure_is_not_wideband_failure(self):
        h = self.trace.channel[0]
        equal_rates = rates(self.cfg, effective_gains(self.cfg, h@h.conj().T), np.full(3, .4))
        trace = replace(self.trace, requirements_mbps=np.tile(.9*equal_rates, (12, 1)))
        physical = np.zeros((12, 3, 3, 128), dtype=complex)
        physical[:, 1] = trace.channel
        narrow, wide = Environment(self.cfg, trace), WidebandEnvironment(self.cfg, trace, physical)
        command = np.ones((1, 16), bool)
        self.assertEqual(narrow_floor(narrow, command)[0], 0)
        self.assertGreater(failure_floor(wide, command)[0], 0)

    def test_independent_physical_replay_and_cache_isolation(self):
        env = WidebandEnvironment(self.cfg, self.trace, self.physical)
        power, _, _ = env.current_phy(np.ones(16, bool))
        power[:] = -100
        self.assertTrue((env.current_phy(np.ones(16, bool))[0] >= 0).all())
        for command in (np.arange(16) < 2, np.arange(16) >= 12, np.ones(16, bool)):
            env.block(command)
        errors = independent(self.cfg, env.records, self.physical, self.trace.requirements_mbps)
        self.assertLess(errors['rate_mbps'], 1e-6)
        self.assertLess(max(v for k, v in errors.items() if k != 'rate_mbps'), 1e-9)

    def test_current_phy_rejects_unobserved_subarrays(self):
        env = WidebandEnvironment(self.cfg, self.trace, self.physical)
        env.block(np.arange(16) < 2)
        with self.assertRaises(ValueError):
            env.current_phy(np.ones(16, bool))

    def test_invalid_physical_input_rejected(self):
        with self.assertRaises(ValueError):
            WidebandEnvironment(self.cfg, self.trace, self.physical[:, :2])
        with self.assertRaises(ValueError):
            WidebandEnvironment(self.cfg, self.trace, self.physical*2)


if __name__ == '__main__':
    unittest.main()
