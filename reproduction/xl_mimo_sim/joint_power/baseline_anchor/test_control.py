import copy
import unittest
from unittest.mock import patch

import numpy as np

from future_vr.config import make_config
from joint_power.baseline_anchor.control import action_at, admission
from joint_power.model import Environment
from joint_power.rollout.collect import base_command
from joint_power.study import make_trace


class AdmissionTests(unittest.TestCase):
    def test_exact_rules_and_numerical_tolerance(self):
        self.assertEqual(admission(0, -2., 'Empty'), (False, 'empty'))
        self.assertEqual(admission(2, 1., 'Empty'), (True, 'accepted'))
        self.assertEqual(admission(2, 1., 'Energy'), (False, 'no_predicted_saving'))
        self.assertEqual(admission(2, -1e-10, 'Energy'), (False, 'no_predicted_saving'))
        self.assertEqual(admission(2, -.01, 'Energy'), (True, 'accepted'))
        with self.assertRaises(ValueError):
            admission(2, np.nan, 'Energy')

    def test_fallback_is_actual_reference_not_previous_mask(self):
        cfg = make_config('validation')
        env = Environment(cfg, make_trace(cfg, 4299220006, (2., 1.)))
        reference = base_command(env)
        proposal = np.ones(16, bool)
        info = dict(feasible_candidates=0, predicted_relative_energy=2., residual_budget=21.,
                    applied_bound=2.1, selected_projected_count=10.)
        before = copy.deepcopy(env.hw)
        with patch('joint_power.baseline_anchor.control.proposal_at', return_value=(proposal, info)):
            command, ledger = action_at(env, None, None, 'Value', 'Energy', .1, 0)
        np.testing.assert_array_equal(command, reference)
        np.testing.assert_array_equal(env.hw.active, before.active)
        np.testing.assert_array_equal(env.hw.countdown, before.countdown)
        self.assertEqual(ledger['executed_bits'], ledger['reference_bits'])
        self.assertEqual(ledger['accepted'], 0)


if __name__ == '__main__':
    unittest.main()
