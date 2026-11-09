import unittest
import numpy as np
import torch
from major_revision.factored_value_fit import loss_scales,objective


class FactoredFitTests(unittest.TestCase):
    def test_loss_uses_absolute_and_paired_targets_and_has_finite_gradient(self):
        truth=np.random.default_rng(42).uniform(size=(3,16,2,2))
        reference=np.array([0,6,15])
        scales=loss_scales(truth,reference)
        target=torch.from_numpy(truth).float()
        refs=torch.from_numpy(reference)
        self.assertEqual(float(objective(target,target,refs,scales)),0.)
        prediction=(target+.1).clone().requires_grad_()
        loss=objective(prediction,target,refs,scales)
        self.assertGreater(float(loss.detach()),0.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        for scale in loss_scales(np.zeros_like(truth),reference):
            np.testing.assert_allclose(scale,.001)


if __name__=='__main__':
    unittest.main()
