import unittest

import numpy as np
import torch

from joint_power.model import Candidates
from joint_power.study import Learner


class JointLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.features=np.array([[1.,0.,.2],[0.,1.,.8]],dtype=np.float32)
        self.prior=np.array([.6,.4],dtype=np.float32)
        self.c=Candidates(np.eye(2,dtype=bool),self.prior,np.zeros(2),np.zeros(2),self.features,self.prior)

    def test_zero_residual_exact_prior(self):
        for seed in (11,22):
            learner=Learner(3,seed)
            values=learner.online(torch.from_numpy(self.features),torch.from_numpy(self.prior))
            np.testing.assert_array_equal(values.detach().numpy(),self.prior)
            self.assertEqual(learner.choose(self.c),1)

    def test_terminal_targets_learn_and_stay_finite(self):
        learner=Learner(3,11)
        losses=[]
        for i in range(164):
            action=i%2
            result=learner.learn(self.c,action,float(action),self.c,True)
            if result is not None:
                losses.append(result)
        self.assertEqual(learner.updates,101)
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(np.mean(losses[-10:]),np.mean(losses[:10]))
        self.assertEqual(learner.choose(self.c),0)


if __name__=='__main__':
    unittest.main()
