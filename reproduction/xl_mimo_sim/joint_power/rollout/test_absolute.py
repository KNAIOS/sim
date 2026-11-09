import unittest

import numpy as np
import torch

from joint_power.rollout.absolute import AbsoluteValue, loss
from joint_power.rollout.probe import select


class AbsoluteTests(unittest.TestCase):
    def test_physical_count_bound_and_anchor(self):
        model = AbsoluteValue(np.zeros(9), np.ones(9))
        x, ref, t = torch.randn(3, 17, 9), torch.tensor([3, 8, 2]), torch.tensor([140., 64., 4.])
        predicted = model(x, ref, t)
        np.testing.assert_array_equal(select(predicted.detach().numpy(), ref.numpy(), 8), ref.numpy())
        with torch.no_grad():
            model.net[-1].weight.normal_()
        energy, count, _ = model.components(x, ref, t)
        self.assertTrue(torch.all(count >= 0) and torch.all(count <= 3*t[:, None]))
        np.testing.assert_array_equal(model(x, ref, t).detach().numpy()[np.arange(3), ref], np.zeros((3, 2)))

    def test_count_loss_targets_the_mean_not_the_median(self):
        # Correlation/Poisson independence is not needed for this mean-risk identity.
        counts = torch.tensor([0., 0., 0., 4.])
        estimates = (.1, 1., 2.)
        objectives = []
        for estimate in estimates:
            p = torch.full_like(counts, estimate/10)
            objectives.append(float(10*torch.nn.functional.binary_cross_entropy(p, counts/10)))
        self.assertEqual(int(np.argmin(objectives)), 1)

    def test_finite_gradients(self):
        model = AbsoluteValue(np.zeros(9), np.ones(9))
        x, ref, t = torch.randn(2, 17, 9), torch.tensor([3, 8]), torch.tensor([140., 64.])
        y = torch.zeros(2, 17, 2)
        y[..., 0] = 30.
        y[:, 0, 1] = 5.
        value = loss(model, x, ref, t, y)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
