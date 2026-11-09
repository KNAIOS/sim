import unittest
from dataclasses import replace

import numpy as np
import torch

from future_vr.config import make_config
from major_revision.bellman_branch_data import FEATURE_SETTING, known_costs
from major_revision.bellman_value import (
    BellmanValue, DEPTH, continuation_target, greedy_cost, tail_weight,
)
from major_revision.factored_value import extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory


class BellmanValueTests(unittest.TestCase):
    def test_scalarized_bellman_identity_uses_same_action_for_both_costs(self):
        costs = torch.tensor([[[1., .4], [2., .1], [3., 0.]],
                              [[1., .4], [2., .1], [3., 0.]]])
        prices = torch.tensor([.5, 8.])
        actions, chosen = greedy_cost(costs, prices)
        torch.testing.assert_close(actions, torch.tensor([0, 1]))
        torch.testing.assert_close(chosen, torch.tensor([[1., .4], [2., .1]]))
        gamma, depth = .98, 3
        target = continuation_target(costs, prices, gamma, depth)
        present = torch.tensor([[.2, .03], [.1, .1]])
        combined = present + tail_weight(gamma, depth)*target
        expected = present+gamma*chosen
        torch.testing.assert_close(combined, expected)
        lhs = combined[:, 0]+prices*combined[:, 1]
        rhs = present[:, 0]+prices*present[:, 1]+gamma*(costs[..., 0]+prices[:, None]*costs[..., 1]).min(1).values
        torch.testing.assert_close(lhs, rhs)

    def test_known_energy_depth_one_and_risk_bounds(self):
        cfg = replace(make_config('validation'), episode_steps=5)
        root = FullDomainRollout(cfg, trajectory(cfg, 1100900101, (2., 1.)))
        x, _ = extend_observation(root, FEATURE_SETTING)
        tx = torch.from_numpy(x[None])
        model = BellmanValue(cfg).eval()
        one_low, one_high = model(tx, .5, 1), model(tx, 16., 1)
        torch.testing.assert_close(one_low, one_high, rtol=0, atol=0)
        e, o = known_costs(cfg, x)
        np.testing.assert_allclose(one_low[0, :, 0].detach().numpy(), e, rtol=0, atol=1e-7)
        self.assertTrue(torch.all(one_low[..., 1] >= torch.tensor(o[None]/4)))
        self.assertTrue(torch.all(one_low[..., 1] <= torch.tensor(o[None]/4+.75)))
        for depth in (2, DEPTH):
            value = model(tx, 2., depth)
            self.assertTrue(torch.all(value >= one_low))
            self.assertLessEqual(float(value[..., 1].max().detach()), 1+tail_weight(cfg.rl_gamma, depth))

    def test_invalid_depth_price_and_training_target_indexing(self):
        with self.assertRaises(ValueError):
            tail_weight(.98, 0)
        with self.assertRaises(ValueError):
            continuation_target(torch.zeros(1, 16, 2), 2., .98, 1)
        model = BellmanValue(make_config('validation'))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 1500), float('nan'))
        target = torch.arange(3*16*6*2).reshape(3, 16, 6, 2)
        roots, prices = np.array([2, 0, 1]), np.array([5, 1, 3])
        selected = target[roots, :, prices, :]
        self.assertEqual(selected.shape, (3, 16, 2))
        for i in range(3):
            torch.testing.assert_close(selected[i], target[roots[i], :, prices[i]])


if __name__ == '__main__':
    unittest.main()
