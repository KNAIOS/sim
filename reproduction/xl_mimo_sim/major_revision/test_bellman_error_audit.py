import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.bellman_branch_data import FEATURE_SETTING, all_action_transitions
from major_revision.bellman_error_audit import (
    deployment_return, error_components, oracle_action, policy_values,
)
from major_revision.bellman_value import BellmanValue
from major_revision.factored_value import extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory


class BellmanErrorTests(unittest.TestCase):
    def test_exact_signed_error_decomposition(self):
        p4, p64 = np.array([.3, .1]), np.array([4., .9])
        r4, r64 = np.array([.3, .2]), np.array([3., 1.2])
        parts = error_components(p4, p64, r4, r64, 2.)
        self.assertAlmostEqual(parts.sum(), (p64-r64) @ [1., 2.])
        self.assertEqual(parts[0], 0.)
        self.assertAlmostEqual(parts[1], -.2)

    def test_oracle_changes_only_immediate_risk(self):
        p4 = np.array([[1., .01], [1., .1]])
        p64 = np.array([[3., .2], [3.1, .3]])
        r4 = np.array([[1., .5], [1., .01]])
        self.assertEqual(oracle_action(p4, p64, r4, 2.), 1)
        # Altering energy labels cannot influence the risk-only diagnostic action.
        r4[:, 0] = [1000., 0.]
        self.assertEqual(oracle_action(p4, p64, r4, 2.), 1)

    def test_branch_first_block_matches_and_root_unchanged(self):
        cfg = replace(make_config('validation'), episode_steps=13)
        root = FullDomainRollout(cfg, trajectory(cfg, 1110900041, (2., 1.)))
        model = BellmanValue(cfg).eval()
        before, score, _, _, chosen = policy_values(root, model, 2.)
        _, _, actual4, _, _, _ = all_action_transitions(root)
        value, blocks = deployment_return(root, select_top(score, chosen+1), model, 2., slots=8)
        np.testing.assert_allclose(blocks[0], actual4[chosen], atol=1e-12)
        np.testing.assert_allclose(value, blocks[0]+cfg.rl_gamma*blocks[1], atol=1e-12)
        self.assertEqual(root.t, 0)
        np.testing.assert_array_equal(extend_observation(root, FEATURE_SETTING)[0], before)


if __name__ == '__main__':
    unittest.main()
