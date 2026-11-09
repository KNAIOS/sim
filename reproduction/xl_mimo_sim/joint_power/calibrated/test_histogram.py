import unittest

import numpy as np
import torch

from joint_power.calibrated.histogram import HistogramPredictor, fit, groups


class HistogramTests(unittest.TestCase):
    def test_named_group_boundaries(self):
        x = torch.zeros(4, 229)
        x[:, -2] = torch.tensor([1, 2, 4, 8])/16
        x[:, -11] = .25
        ids, risks = groups(x)
        np.testing.assert_array_equal(ids.numpy(), [3, 29, 55, 81])
        x[:, -4] = 1/16
        awake_ids, _ = groups(x)
        np.testing.assert_array_equal((awake_ids-ids).numpy(), [13]*4)

    def test_training_averages_and_unseen_group_are_bounded(self):
        x = np.zeros((12, 229), np.float32)
        x[:, -2] = 4/16
        x[:, -13] = .4
        train = dict(x=x, y=np.tile([.4, .1], (12, 1)).astype(np.float32))
        model, _ = fit(train, 32)
        prediction = model.predict(x)
        np.testing.assert_allclose(prediction[:, 0], .4)
        self.assertTrue(((prediction[:, 1] > 0)&(prediction[:, 1] < 1)).all())
        rebuilt = HistogramPredictor(model.mean, model.std, model.table)
        np.testing.assert_array_equal(prediction, rebuilt.predict(x))


if __name__ == '__main__':
    unittest.main()
