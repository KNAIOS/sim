import unittest

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.study import make_trace
from joint_power.calibrated.calibration import Predictor, branch, scores


class CalibrationTests(unittest.TestCase):
    def test_branch_does_not_modify_source(self):
        cfg = make_config('validation')
        env = Environment(cfg, make_trace(cfg, 2026092204, (2., 1.)))
        c = env.candidates()
        saved = c.features.copy()
        alternative = branch(env)
        alternative.block(c.masks[0])
        self.assertEqual(env.t, 0)
        self.assertFalse(env.records)
        self.assertTrue(env.hw.active.all())
        self.assertIs(alternative.tr, env.tr)
        np.testing.assert_array_equal(env.candidates().features, saved)

    def test_initial_prediction_preserves_analytic_energy(self):
        cfg = make_config('validation')
        c = Environment(cfg, make_trace(cfg, 2026092205, (2., 1.))).candidates()
        net = Predictor(np.zeros(229), np.ones(229))
        prediction = net.predict(c.features)
        np.testing.assert_allclose(prediction[:, 0], c.features[:, -13], atol=1e-7)
        np.testing.assert_allclose(prediction[:, 1], np.clip(c.features[:, -11], .01, .99), atol=1e-7)
        self.assertTrue(torch.isfinite(net(torch.from_numpy(c.features))).all())

    def test_scores_detect_miscalibration(self):
        y = np.array([[.5, 0], [.5, 1.]])
        correct = scores(y, y)
        wrong = scores(np.array([[.5, 1], [.5, 0.]]), y)
        self.assertEqual(correct['brier'], 0)
        self.assertEqual(wrong['brier'], 1)
        self.assertLess(correct['bce'], wrong['bce'])


if __name__ == '__main__':
    unittest.main()
