import copy
import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.model import Environment
from joint_power.postdecision.encoding import decision
from joint_power.rollout.collect import base_command, rollout, state_labels
from joint_power.study import make_trace


class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(make_config('validation'), episode_steps=20)
        self.trace = make_trace(self.cfg, 3599220000, (2., 1.))

    def test_reference_suffix_and_no_mutation(self):
        env = Environment(self.cfg, self.trace)
        env.block(base_command(env))
        env.prepare()
        hardware, age, current_time = copy.deepcopy(env.hw), env.age.copy(), env.t
        labels = state_labels(env, load_predictor())
        self.assertEqual(labels['y'].shape, (17, 3, 2))
        self.assertEqual(env.t, current_time)
        np.testing.assert_array_equal(env.hw.active, hardware.active)
        np.testing.assert_array_equal(env.hw.countdown, hardware.countdown)
        np.testing.assert_array_equal(env.age, age)
        previous = len(env.records)
        while env.t < self.cfg.episode_steps:
            env.block(base_command(env))
        suffix = env.records[previous:]
        expected = [sum(r['energy_j'] for r in suffix),
                    sum(round(3*r['outage']) for r in suffix)]
        np.testing.assert_allclose(labels['y'][labels['reference'], -1], expected, atol=1e-10)

    def test_immediate_label_and_duplicate_commands(self):
        env = Environment(self.cfg, self.trace)
        labels = state_labels(env, load_predictor())
        np.testing.assert_array_equal(labels['y'][15], labels['y'][16])
        for a in (0, 3, 8, 15):
            alternative = copy.deepcopy(env)
            alternative.block(labels['masks'][a])
            expected = [alternative.metrics()['energy_j'],
                        sum(round(3*r['outage']) for r in alternative.records)]
            np.testing.assert_allclose(labels['y'][a, 0], expected, atol=1e-10)

    def test_current_features_independent_of_future_outcomes(self):
        env = Environment(self.cfg, self.trace)
        env.block(np.arange(16) < 4)
        altered = copy.deepcopy(env)
        channel, gains = self.trace.channel.copy(), self.trace.gain_db.copy()
        channel[env.t+1:] *= .001
        gains[env.t+1:] -= 60
        altered.tr = replace(self.trace, channel=channel, gain_db=gains)
        predictor = load_predictor()
        first = decision(env, env.candidates(), predictor)
        second = decision(altered, altered.candidates(), predictor)
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(base_command(env), base_command(altered))
        # Outcome labels are supposed to change; they are not decision features.
        self.assertFalse(np.allclose(rollout(env, base_command(env)),
                                    rollout(altered, base_command(altered))))


if __name__ == '__main__':
    unittest.main()
