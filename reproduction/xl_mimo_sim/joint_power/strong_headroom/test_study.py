import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.strong_headroom.study import advance, reference_command, select, state_labels, totals
from joint_power.study import make_trace


class HeadroomTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(make_config('validation'), episode_steps=20)
        self.trace = make_trace(self.cfg, 5099220003, (2., 1.))

    def test_suffix_identity_and_parent_preserved(self):
        for adaptive in (False, True):
            env, past = Environment(self.cfg, self.trace), 0
            past = advance(env, reference_command(env, adaptive, past), past)
            env.prepare()
            before = copy.deepcopy(env)
            labels, _ = state_labels(env, adaptive, past)
            self.assertEqual(labels['y'].shape, (17, 3, 6))
            self.assertEqual(env.t, before.t)
            self.assertEqual(env.records, before.records)
            for a, b in ((env.age, before.age), (env.hw.active, before.hw.active),
                         (env.hw.countdown, before.hw.countdown),
                         (env.tracker.channel_estimate, before.tracker.channel_estimate)):
                np.testing.assert_array_equal(a, b)
            start = len(env.records)
            while env.t < self.cfg.episode_steps:
                past = advance(env, reference_command(env, adaptive, past), past)
            np.testing.assert_allclose(labels['y'][labels['reference'], -1],
                                       totals(env.records[start:], self.cfg.users), atol=1e-9)

    def test_duplicate_masks_and_component_sums(self):
        env = Environment(self.cfg, self.trace)
        labels, _ = state_labels(env, False, 0)
        np.testing.assert_array_equal(labels['y'][15], labels['y'][16])
        np.testing.assert_allclose(labels['y'][..., 0], labels['y'][..., 2:].sum(-1), atol=1e-9)

    def test_causal_features_do_not_include_future(self):
        env = Environment(self.cfg, self.trace)
        env.block(np.arange(16) < 3)
        altered = copy.deepcopy(env)
        channel, gain = self.trace.channel.copy(), self.trace.gain_db.copy()
        channel[env.t+1:] *= .01
        gain[env.t+1:] -= 40
        altered.tr = replace(self.trace, channel=channel, gain_db=gain)
        first, _ = state_labels(env, True, 0)
        second, _ = state_labels(altered, True, 0)
        np.testing.assert_array_equal(first['x'], second['x'])
        np.testing.assert_array_equal(first['masks'], second['masks'])
        self.assertEqual(first['reference'], second['reference'])
        self.assertFalse(np.allclose(first['y'], second['y']))

    def test_budget_uses_past_and_remaining_horizon(self):
        env = Environment(self.cfg, self.trace)
        env.block(np.ones(16, bool))
        with patch('joint_power.strong_headroom.study.command_at', return_value=np.ones(16, bool)) as call:
            reference_command(env, True, 1)
            self.assertAlmostEqual(call.call_args.args[0].outage_target, (3-1)/(3*16))
            reference_command(env, True, 9)
            self.assertEqual(call.call_args.args[0].outage_target, 0)
            reference_command(env, False, 9)
            self.assertEqual(call.call_args.args[0].outage_target, .05)

    def test_shared_full_tail_risk_filter(self):
        y = np.zeros((3, 3, 6))
        y[:, :, 0] = [[1, 10, 10], [2, 8, 8], [.5, 3, 3]]
        y[:, -1, 1] = [2, 1, 3]
        self.assertEqual(select(y, 0, 0), 0)
        self.assertEqual(select(y, 0, 2), 1)


if __name__ == '__main__':
    unittest.main()
