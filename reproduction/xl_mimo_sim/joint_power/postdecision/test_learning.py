import unittest

import numpy as np
import torch

from future_vr.config import make_config
from joint_power.model import Environment
from joint_power.study import make_trace
from joint_power.calibrated.histogram import load_predictor
from joint_power.postdecision.encoding import decision
from joint_power.postdecision.learning import Learner, counterfactuals


class PostDecisionLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.predictor = load_predictor()
        cfg = make_config('validation')
        self.env = Environment(cfg, make_trace(cfg, 2026092212, (2., 1.)))
        self.state = decision(self.env, self.env.candidates(), self.predictor)

    def test_branches_have_action_specific_successors_and_no_mutation(self):
        before = self.state.x.copy()
        transitions = counterfactuals(self.env, self.state, self.predictor)
        self.assertEqual(self.state.x.shape, (17, 343))
        self.assertEqual(len(transitions), 17)
        self.assertEqual(self.env.t, 0)
        self.assertFalse(self.env.records)
        self.assertFalse(np.array_equal(transitions[0][3].x, transitions[15][3].x))
        np.testing.assert_array_equal(before, decision(self.env, self.env.candidates(), self.predictor).x)
        actual = self.env.block(self.state.masks[0])
        following = decision(self.env, self.env.candidates(), self.predictor)
        np.testing.assert_allclose(transitions[0][2], actual)
        np.testing.assert_allclose(transitions[0][3].x, following.x)

    def test_single_step_replay_and_gradients(self):
        learner = Learner(11)
        transitions = counterfactuals(self.env, self.state, self.predictor)
        for _ in range(4):
            loss = learner.learn_transitions(transitions)
        self.assertEqual(learner.transitions_seen, 68)
        self.assertEqual(learner.updates, 4)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(item[4] == .98 for item in learner.memory))

    def test_multistep_price_relabeling_is_not_policy_relabeling(self):
        # At next state: action0 costs(E=0,C=1), action1 costs(E=1,C=0).
        # A behavior price8 chooses action1; target price.1 chooses action0.
        future = np.array([[0., 1.], [1., 0.]])
        behavior = int((future@[1., 8.]).argmin())
        relabeled_sample = .98*float(future[behavior]@[1., .1])
        one_step_optimal_target = .98*float((future@[1., .1]).min())
        self.assertGreater(relabeled_sample, one_step_optimal_target)
        self.assertAlmostEqual(relabeled_sample-one_step_optimal_target, .882)


if __name__ == '__main__':
    unittest.main()
