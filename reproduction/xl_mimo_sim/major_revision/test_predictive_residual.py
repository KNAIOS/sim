import unittest
from dataclasses import replace
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import Transition
from major_revision.pilot import episode,trajectory
from major_revision.predictive_residual import ResidualDQN,FixedResidual,residual_episode


class PredictiveResidualTests(unittest.TestCase):
    def test_masked_network_supports_a_finite_td_update(self):
        cfg=make_config('validation')
        model=ResidualDQN(cfg,11)
        for i in range(72):
            old=np.zeros(218,dtype=np.float32)
            nxt=np.zeros(218,dtype=np.float32)
            old[-1]=(4,8,16)[i%3]/16
            nxt[-1]=(16,4,8)[i%3]/16
            model.remember(Transition(old,1,.5,.01,nxt,False))
        before=model.online.network.layers[-1].bias.detach().clone()
        loss=model.learn()
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss,0.)
        self.assertFalse(torch.equal(before,model.online.network.layers[-1].bias))

    def test_zero_residual_and_initial_network_reproduce_prs(self):
        for delay in (0,2,4,6):
            cfg = replace(make_config('validation'),episode_steps=20,wake_delay_slots=delay)
            tr = trajectory(cfg,979800000+delay,(3.,1.15))
            reference = episode(cfg,tr,'PRS',rate_margin=1.04)
            for policy in (FixedResidual(cfg,0),ResidualDQN(cfg,11)):
                result = residual_episode(cfg,tr,policy,margin=1.04)
                result['method']='PRS'
                self.assertEqual(reference,result)

    def test_boundaries_masked_for_exploration_and_both_networks(self):
        cfg = make_config('validation')
        model = ResidualDQN(cfg,11)
        for anchor in (4,8,16):
            obs = np.zeros(218,dtype=np.float32)
            obs[-1] = anchor/cfg.subarrays
            for network in (model.online,model.target):
                q = network(torch.as_tensor(obs)[None]).detach().numpy()[0]
                self.assertEqual(q[1],0.)
                if anchor==4:
                    self.assertLess(q[0],-1e8)
                if anchor==16:
                    self.assertLess(q[2],-1e8)
            for _ in range(50):
                action,count = model.choose(obs,1.)
                self.assertEqual(count,anchor+action-1)
                self.assertTrue(4<=count<=16)

    def test_training_records_real_continuing_successor(self):
        cfg = replace(make_config('validation'),episode_steps=20)
        tr = trajectory(replace(cfg,episode_steps=21),979800011,(2.,1.))
        model = ResidualDQN(cfg,11)
        # Five transitions are below the batch threshold, so physics must still
        # be that of the unchanged anchor despite enabled training.
        result = residual_episode(cfg,tr,model,training=True)
        limited = replace(tr,**{name:getattr(tr,name)[:20] for name in tr.__dataclass_fields__})
        expected = episode(cfg,limited,'PRS')
        result['method']='PRS'
        self.assertEqual(result,expected)
        self.assertEqual(len(model.memory),5)
        self.assertFalse(any(t.done for t in model.memory))
        self.assertEqual(model.memory[-1].next_state.size,218)
        self.assertFalse(np.array_equal(model.memory[-1].state,model.memory[-1].next_state))
        with self.assertRaises(ValueError):
            residual_episode(cfg,limited,model,training=True)


if __name__=='__main__':
    unittest.main()
