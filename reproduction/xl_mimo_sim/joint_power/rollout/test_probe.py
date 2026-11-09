import unittest

import numpy as np
import torch

from joint_power.rollout.probe import DifferenceValue, FULL_SLOT, labels, select


class ProbeTests(unittest.TestCase):
    def test_exact_anchor_and_initial_baseline(self):
        torch.manual_seed(11)
        model = DifferenceValue(np.zeros(9), np.ones(9))
        x, refs = torch.randn(4, 17, 9), torch.tensor([3, 8, 6, 15])
        prediction = model(x, refs)
        np.testing.assert_array_equal(select(prediction.detach().numpy(), refs.numpy(), 8), refs.numpy())
        with torch.no_grad():
            model.net[-1].weight.normal_()
        prediction = model(x, refs)
        np.testing.assert_array_equal(prediction.detach().numpy()[np.arange(4), refs], np.zeros((4, 2)))

    def test_cost_units_and_action(self):
        y = np.zeros((2, 17, 3, 2))
        y[0, :, -1] = [10., 3.]
        y[0, 0, -1] = [10.-FULL_SLOT, 6.]
        data = dict(y=y, reference=np.array([3, 3]))
        target = labels(data)
        np.testing.assert_allclose(target[0, 0], [-1., 1.], atol=1e-6)
        self.assertEqual(select(target, data['reference'], 8.)[0], 3)
        self.assertEqual(select(target, data['reference'], .1)[0], 0)


if __name__ == '__main__':
    unittest.main()
