import unittest

import numpy as np
import torch

from joint_power.calibrated.control import Decision, Learner, ValueNetwork, calibrated_search


class ControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        masks = np.array([np.arange(16) < n for n in range(1, 17)]+[np.ones(16, bool)])
        self.state = Decision(masks, np.zeros((17, 231), np.float32),
                              np.stack((np.linspace(.3, 1., 17), np.linspace(.2, 0, 17)), -1).astype(np.float32), 20.)

    def test_initial_action_ranking_does_not_depend_on_tail(self):
        model = ValueNetwork()
        x, im = torch.from_numpy(self.state.x), torch.from_numpy(self.state.immediate)
        first = model(x, im, torch.tensor(0.), torch.tensor(2.))
        last = model(x, im, torch.tensor(20.), torch.tensor(2.))
        np.testing.assert_allclose((first-first[0]).detach(), (last-last[0]).detach(), atol=2e-6)
        self.assertEqual(int(first.argmin()), int(last.argmin()))

    def test_terminal_nstep_returns_and_flush(self):
        learner = Learner(11)
        for i in range(3):
            learner.remember(self.state, 0, (1., .1), self.state, i == 2)
        self.assertFalse(learner.pending)
        self.assertEqual(len(learner.memory), 3)
        for i, item in enumerate(learner.memory):
            np.testing.assert_allclose(item[2], np.array([1., .1])*sum(.98**j for j in range(3-i)))
            self.assertEqual(item[4], 0.)

    def test_price_conditioned_update_is_finite(self):
        learner = Learner(11)
        for _ in range(70):
            loss = learner.remember(self.state, 0, (.2, 0.), self.state, True)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(learner.updates, 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in learner.online.parameters()))

    def test_calibrated_search_respects_grid_and_common_risk(self):
        a = calibrated_search(self.state, .05, stride=3)
        self.assertIn(a, [0, 3, 6, 9, 12, 15, 16])
        self.assertLessEqual(self.state.immediate[a, 1], .05)


if __name__ == '__main__':
    unittest.main()
