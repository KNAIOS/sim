import unittest
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import Transition
from major_revision.predictive_residual import ResidualDQN


class PricedTrainingTests(unittest.TestCase):
    def test_fixed_price_changes_td_reward_not_initial_policy(self):
        cfg=make_config('validation')
        losses=[]
        for price in (1.,2.):
            model=ResidualDQN(cfg,11)
            model.dual=price
            state=np.zeros(218,dtype=np.float32)
            state[-1]=8/16
            self.assertEqual(model.choose(state,0.),(1,8))
            for _ in range(64):
                model.remember(Transition(state.copy(),1,.5,.1,state.copy(),False))
            loss=model.learn()
            losses.append(loss)
            self.assertAlmostEqual(loss,.5*(.5+.1*price)**2,places=5)
        self.assertGreater(losses[1],losses[0])


if __name__=='__main__':
    unittest.main()
