"""Exact subset equivalence, actual kernel work and causal observation tests."""
import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from future_vr.config import make_config
from joint_power.budgeted import evaluate
from joint_power.model import Environment, effective_gains, search_action
from joint_power.study import make_trace


class BudgetedTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config('validation')
        self.trace = make_trace(self.cfg, 3509220000, (2., 1.))

    def test_exact_full_and_subset(self):
        env = Environment(self.cfg, self.trace)
        for count in (4, 10, 6, 16):
            reference = env.candidates()
            active, countdown = env.hw.active.copy(), env.hw.countdown.copy()
            for stride in (1, 2, 3, 4, 8):
                result = evaluate(env, stride)
                np.testing.assert_array_equal(result.masks, reference.masks[result.ids])
                for metric in ('energy', 'outage', 'shortfall'):
                    np.testing.assert_allclose(getattr(result, metric),
                        getattr(reference, metric)[result.ids], atol=1e-10)
                self.assertEqual(result.ids[result.choose()], search_action(reference, stride))
                np.testing.assert_array_equal(env.hw.active, active)
                np.testing.assert_array_equal(env.hw.countdown, countdown)
            env.block(np.arange(16) < count)

    def test_real_candidate_work_not_posthoc_slicing(self):
        env = Environment(self.cfg, self.trace)
        with patch('joint_power.budgeted.effective_gains', wraps=effective_gains) as kernel:
            result = evaluate(env, 3)
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(kernel.call_args.args[1].shape, (7, 8, 3, 3))
        self.assertEqual(result.candidate_slots, 56)
        self.assertEqual(evaluate(env, 1).candidate_slots, 136)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                evaluate(env, invalid)

    def test_no_hidden_or_future_channel_leak(self):
        env = Environment(self.cfg, self.trace)
        env.block(np.arange(16) < 4)
        altered = copy.deepcopy(env)
        channel, gain = self.trace.channel.copy(), self.trace.gain_db.copy()
        channel[env.t, :, np.repeat(~env.hw.active, 8)] *= 1000
        channel[env.t+1:] *= 1000
        gain[env.t, ~env.hw.active] += 60
        gain[env.t+1:] += 60
        altered.tr = replace(self.trace, channel=channel, gain_db=gain)
        first, second = evaluate(env, 3), evaluate(altered, 3)
        for field in ('ids', 'masks', 'energy', 'outage', 'shortfall'):
            np.testing.assert_allclose(getattr(first, field), getattr(second, field), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
