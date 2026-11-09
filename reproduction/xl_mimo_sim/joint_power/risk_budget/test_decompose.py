import unittest
from dataclasses import replace

import torch

from future_vr.config import make_config
from joint_power.aggregation.control import models
from joint_power.calibrated.histogram import load_predictor
from joint_power.risk_budget.decompose import decomposition
from joint_power.study import make_trace


class DecompositionTests(unittest.TestCase):
    def test_complete_visited_state_advantages_telescope(self):
        torch.set_num_threads(1)
        cfg = replace(make_config('validation'), episode_steps=20)
        trace = make_trace(cfg, 4299220002, (2., 1.))
        loaded, _ = models()
        _, states, audit = decomposition(cfg, trace, loaded['Visitation-added', 11], load_predictor(), .1)
        self.assertEqual(len(states), 5)
        self.assertLess(audit['energy_identity_error'], 1e-8)
        self.assertEqual(audit['failed_identity_error'], 0)


if __name__ == '__main__':
    unittest.main()
