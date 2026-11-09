"""Physical replay, feature/action dimensions and truncation tests."""
import unittest
from dataclasses import replace
import numpy as np
import torch
from future_vr.config import make_config
from major_revision.audit_rollout import AuditRollout
from major_revision.full_domain import FullDomainDQN, FullDomainRollout, full_domain_episode, observation_size
from major_revision.pilot import episode, trajectory


class FullDomainTests(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(make_config('validation'),episode_steps=12)
        self.tr=trajectory(self.cfg,981001111,(2.,1.))

    def test_four_floor_observation_matches(self):
        old,new=AuditRollout(self.cfg,self.tr),FullDomainRollout(self.cfg,self.tr,4)
        np.testing.assert_array_equal(old.observation()[0],new.observation()[0])
        self.assertEqual(observation_size(self.cfg),235)
        self.assertEqual(observation_size(self.cfg,4),217)

    def test_four_floor_search_and_physics_match(self):
        for delay in (0,4,6):
            cfg=replace(self.cfg,wake_delay_slots=delay)
            for mode,old_name in (('prs','PRS'),('cost','Switch-aware PRS'),('full','Full array')):
                old=episode(cfg,self.tr,old_name)
                new=full_domain_episode(cfg,self.tr,mode=mode,minimum=4)
                for key,value in old.items():
                    if key!='method':
                        self.assertAlmostEqual(value,new[key],places=9)

    def test_new_inputs_extend_old_rate_blocks(self):
        a=FullDomainRollout(self.cfg,self.tr,4).observation()[0]
        b=FullDomainRollout(self.cfg,self.tr,1).observation()[0]
        self.assertEqual(len(b),235)
        np.testing.assert_array_equal(a[:139],b[:139])
        np.testing.assert_array_equal(a[139:178],b[148:187])
        np.testing.assert_array_equal(a[178:],b[196:])

    def test_model_and_successor(self):
        torch.set_num_threads(1)
        model=FullDomainDQN(self.cfg,11)
        obs=FullDomainRollout(self.cfg,self.tr).observation()[0]
        self.assertEqual(tuple(model.online(torch.from_numpy(obs)[None]).shape),(1,16))
        model.choose=lambda observation,epsilon:(0,1)
        model.learn=lambda:0.
        extended=replace(self.cfg,episode_steps=13)
        tr=trajectory(extended,981001111,(2.,1.))
        result=full_domain_episode(self.cfg,tr,model,True)
        self.assertEqual(len(model.memory),3)
        self.assertTrue(all(not row.done for row in model.memory))
        self.assertEqual(model.memory[-1].next_state.shape,(235,))
        self.assertAlmostEqual(result['energy_j'],result['switching_j']+
                               result['operating_j']+result['pilot_j'])
        self.assertEqual(result['below_four_fraction'],1.)
        with self.assertRaises(ValueError):
            full_domain_episode(self.cfg,self.tr,model,True)


if __name__=='__main__':
    unittest.main()
