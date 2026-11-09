import unittest

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.calibrated.histogram import load_predictor
from joint_power.calibrated.study import episode as previous_episode
from joint_power.rollout.absolute import AbsoluteValue
from joint_power.rollout.control import episode
from joint_power.study import make_trace


class ControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.trace = make_trace(make_config('validation'), 3899220000, (2., 1.))
        self.predictor = load_predictor()

    def test_initial_absolute_policy_is_exact_prs(self):
        model = AbsoluteValue(np.zeros(343), np.ones(343))
        actual, records = episode(self.trace, 'MC absolute', 8., self.predictor, model)
        baseline, reference = episode(self.trace, 'PRS original', .05, self.predictor)
        self.assertEqual(records, reference)
        self.assertEqual(actual, baseline)

    def test_baselines_preserved_at_original_threshold(self):
        for method, previous in (('PRS original', 'PRS power'), ('Raw full', 'Raw full'), ('Raw coarse', 'Raw coarse')):
            actual, records = episode(self.trace, method, .05, self.predictor)
            baseline, reference = previous_episode(self.trace, previous, self.predictor)
            self.assertEqual(records, reference)
            for metric in actual:
                self.assertEqual(actual[metric], baseline[metric])


if __name__ == '__main__':
    unittest.main()
