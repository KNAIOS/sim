import unittest
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.factored_value import extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.rollout_value_data import branch_return
from major_revision.policy_consistent_value import continuation_return, epoch_indices, promotion


class FullCount(torch.nn.Module):
    def forward(self, x):
        costs = torch.zeros((len(x), 16, 2, 2), dtype=x.dtype)
        costs[:, :, :, 0] = torch.arange(16., 0., -1.)[None, :, None]
        return costs


class PolicyConsistentValueTests(unittest.TestCase):
    def test_branch_replay_and_current_block_invariance(self):
        cfg = make_config('validation')
        setting = dict(margin=1.08, minimum=1)
        run = FullDomainRollout(cfg, trajectory(cfg, 1000900030, (2., 1.)))
        before, score = extend_observation(run, setting)
        command = select_top(score, 3)
        old, _ = branch_return(run, command, setting)
        mu, _ = continuation_return(run, command, setting)
        pi, blocks = continuation_return(run, command, setting, FullCount())
        np.testing.assert_array_equal(old, mu)
        np.testing.assert_array_equal(mu[0], pi[0])
        np.testing.assert_allclose(pi[1], (blocks*cfg.rl_gamma**np.arange(4)[:, None]).sum(0))
        self.assertEqual(run.t, 0)
        np.testing.assert_array_equal(extend_observation(run, setting)[0], before)

    def test_factorial_budget_and_state_access(self):
        keys = np.array([[0, i] for i in range(5)]+[[1, i] for i in range(5)])
        for variant in ('mu_mu', 'mu_pi', 'mix_mu', 'mix_pi'):
            order = epoch_indices(keys, variant, np.random.default_rng(11))
            self.assertEqual(len(order), 10)
            if variant.startswith('mu_'):
                np.testing.assert_array_equal(np.bincount(order, minlength=10), [2]*5+[0]*5)
            else:
                self.assertEqual(len(set(order)), 10)

    def test_promotion_requires_risk_matched_gain_and_controls(self):
        candidate = dict(energy_j=97., outage=.04)
        pure = dict(energy_j=100., outage=.039)
        self.assertFalse(promotion(candidate, pure, dict(energy_j=98.), [])['passed'])
        self.assertTrue(promotion(candidate, pure, dict(energy_j=100.), [dict(energy_j=99., outage=.04)])['passed'])
        self.assertFalse(promotion(candidate, pure, dict(energy_j=100.), [dict(energy_j=96., outage=.04)])['passed'])
        self.assertFalse(promotion(candidate, None, None, [])['passed'])


if __name__ == '__main__':
    unittest.main()
