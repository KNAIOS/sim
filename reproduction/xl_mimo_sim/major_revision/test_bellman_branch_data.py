import copy
import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.bellman_branch_data import (
    FEATURE_SETTING, all_action_transitions, block_transition, known_costs,
)
from major_revision.factored_value import extend_observation
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory


class BellmanBranchTests(unittest.TestCase):
    def root(self, seed=1100900011):
        cfg = replace(make_config('validation'), episode_steps=13)
        return FullDomainRollout(cfg, trajectory(cfg, seed, (2., 1.)))

    def test_all_actions_match_known_costs_and_leave_root_unchanged(self):
        root = self.root()
        x, score = extend_observation(root, FEATURE_SETTING)
        hardware, tracker = copy.deepcopy(root.hw), copy.deepcopy(root.tracker)
        after, _, costs, successors, ee, ce = all_action_transitions(root)
        self.assertEqual(root.t, 0)
        self.assertEqual(root.energy, 0)
        self.assertEqual(root.outages, [])
        np.testing.assert_array_equal(after, x)
        for field in ('active', 'countdown', 'command'):
            np.testing.assert_array_equal(getattr(hardware, field), getattr(root.hw, field))
        np.testing.assert_array_equal(tracker.channel_estimate, root.tracker.channel_estimate)
        self.assertEqual(costs.shape, (16, 2))
        self.assertEqual(successors.shape, (16, 1500))
        e, o = known_costs(root.cfg, x)
        np.testing.assert_allclose(e, costs[:, 0], atol=5e-7, rtol=0)
        self.assertLessEqual(ee, 5e-7)
        self.assertLessEqual(ce, 5e-7)
        for a in (0, 7, 15):
            trial = root.clone()
            trial.step(select_top(score, a+1))
            self.assertAlmostEqual(trial.outages[-1], float(o[a]), places=6)
            for _ in range(3):
                trial.step()
            np.testing.assert_array_equal(extend_observation(trial, FEATURE_SETTING)[0], successors[a])

    def test_cutoff_uses_real_successor_and_rejects_missing_sample(self):
        root = self.root()
        for _ in range(2):
            _, score = extend_observation(root, FEATURE_SETTING)
            root.step(select_top(score, 8))
            for _ in range(3):
                root.step()
        _, score = extend_observation(root, FEATURE_SETTING)
        _, successor, _ = block_transition(root, select_top(score, 9))
        self.assertEqual(successor.shape, (1500,))
        self.assertEqual(root.t, 8)
        root.cfg = replace(root.cfg, episode_steps=12)
        with self.assertRaises(ValueError):
            block_transition(root, select_top(score, 9))

    def test_no_future_peeking_or_post_successor_suffix_dependence(self):
        root = self.root()
        altered = copy.deepcopy(root.tr)
        # Samples strictly after successor t=4 may not change the transition.
        altered.channel[5:] *= 0.01
        altered.gain_db[5:] -= 40.
        altered.requirements_mbps[5:] *= 3.
        alternate = FullDomainRollout(root.cfg, altered)
        x, score = extend_observation(root, FEATURE_SETTING)
        np.testing.assert_array_equal(x, extend_observation(alternate, FEATURE_SETTING)[0])
        mask = select_top(score, 5)
        c1, nx1, o1 = block_transition(root, mask)
        c2, nx2, o2 = block_transition(alternate, mask)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(nx1, nx2)
        self.assertEqual(o1, o2)
        # Changing the unobserved immediate future also cannot leak into x_t.
        changed = copy.deepcopy(root.tr)
        changed.channel[1:] *= 0.01
        changed.gain_db[1:] -= 40.
        changed.requirements_mbps[1:] *= 3.
        future_changed = FullDomainRollout(root.cfg, changed)
        np.testing.assert_array_equal(x, extend_observation(future_changed, FEATURE_SETTING)[0])


if __name__ == '__main__':
    unittest.main()
