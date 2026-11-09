import unittest
from unittest.mock import patch

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.floor_projection.control import action_at, project_counts
from joint_power.model import Environment
from joint_power.risk_budget.control import action_at as original
from joint_power.risk_budget.study import controllers
from joint_power.study import make_trace


class ProjectionTests(unittest.TestCase):
    def test_projection_only_raises_inconsistent_count(self):
        raw = np.array([.2, 3., 4.])
        floor = np.array([1, 0, 4])
        projected = project_counts(raw, floor)
        np.testing.assert_array_equal(projected, [1, 3, 4])
        np.testing.assert_array_equal(project_counts(projected, floor), projected)
        np.testing.assert_array_equal(raw, [.2, 3, 4])

    def test_invalid_counts_rejected(self):
        for raw, lower in (([1], [1, 2]), ([-1], [0]), ([1], [np.nan])):
            with self.assertRaises(ValueError):
                project_counts(raw, lower)

    def test_zero_floor_reproduces_existing_control(self):
        torch.set_num_threads(1)
        cfg, predictor = make_config('validation'), load_predictor()
        env = Environment(cfg, make_trace(cfg, 4299220003, (2., 1.)))
        loaded, _ = controllers()
        model = loaded['Value Visitation-added', 11][1]
        with patch('joint_power.floor_projection.control.failure_floor', return_value=np.zeros(17)):
            for rule in ('Value', 'Myopic', 'Raw budget'):
                expected, old = original(env, predictor, model, rule, .1, 0)
                actual, new = action_at(env, predictor, model, rule, .1, 0)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(new['selected_raw_count'], old['selected_count_prediction'])
                self.assertEqual(new['changed_command'], 0)


if __name__ == '__main__':
    unittest.main()
