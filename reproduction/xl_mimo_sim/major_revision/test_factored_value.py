import unittest
from dataclasses import replace
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.rollout_value_data import branch_return
from major_revision.factored_value import FactoredNetwork,extend_observation,relative
from major_revision.structural_value_audit import known_costs


class FactoredValueTests(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(make_config('validation'),episode_steps=24)
        self.run=FullDomainRollout(self.cfg,trajectory(self.cfg,1050900001,(2.,1.)))
        self.setting=dict(margin=1.08,minimum=1)

    def test_current_csi_and_known_cost_remain_exact_after_wake(self):
        for t in (0,4):
            if t:
                self.run.step(select_top(score,3))
                for _ in range(3):
                    self.run.step()
            x,score=extend_observation(self.run,self.setting)
            self.assertEqual(x.shape,(1500,))
            energy,outage=known_costs(self.run,score)
            np.testing.assert_allclose(x[-64:].reshape(16,4)[:,-1],outage,atol=3e-8)
            model=FactoredNetwork(self.cfg)
            predicted=model(torch.from_numpy(x[None])).detach().numpy()[0]
            np.testing.assert_allclose(predicted[:,0,0],energy,atol=1e-7)
            self.assertTrue(np.all(predicted[:,0,1]>=outage/4-1e-7))
            self.assertTrue(np.all(predicted[:,0,1]<=outage/4+.75+1e-7))
            self.assertTrue(np.all(predicted[:,1]>=predicted[:,0]))
            self.assertTrue(np.all(predicted[:,1,1]<=1+model.tail_weight))

    def test_paired_reference_zero_and_both_models_can_train(self):
        x,score=extend_observation(self.run,self.setting)
        target=np.array([[branch_return(self.run,select_top(score,a+1),self.setting)[0] for a in range(16)]])
        ref=torch.tensor([5])
        for structured in (True,False):
            model=FactoredNetwork(self.cfg,structured)
            model.initialize_from_training(x[None],target)
            tensor=torch.from_numpy(x[None])
            before=model(tensor)
            np.testing.assert_array_equal(relative(before,ref)[0,5].detach(),0.)
            loss=(before-torch.from_numpy(target).float()).square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            opt=torch.optim.AdamW(model.parameters(),lr=4e-4)
            opt.step()
            self.assertGreater(float((model(tensor)-before).detach().abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
