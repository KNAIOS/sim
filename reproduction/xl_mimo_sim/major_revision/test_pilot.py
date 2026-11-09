import unittest
from dataclasses import replace
import numpy as np
import torch
from unittest.mock import patch
from future_vr.config import make_config
from future_vr.channel import generate_trajectory
from future_vr.controllers import ConstrainedDQN
from future_vr.experiment import simulate_episode
from future_vr.belief import VisibilityBelief
from future_vr.system import HardwareState
from major_revision.pilot import episode, state, cost_search


class PilotTests(unittest.TestCase):
    def test_margin_changes_selection_not_true_requirements(self):
        requests=[]
        def command(cfg, tracker, hw, req, score):
            requests.append(req.copy())
            return np.ones(cfg.subarrays,bool)
        before=self.tr.requirements_mbps.copy()
        with patch('major_revision.pilot.rate_search_command',side_effect=command):
            result=episode(self.cfg,self.tr,'PRS',rate_margin=1.5)
        full=episode(self.cfg,self.tr,'Full array')
        np.testing.assert_array_equal(before,self.tr.requirements_mbps)
        np.testing.assert_allclose(requests[0],1.5*before[0])
        self.assertEqual(result,dict(full,method='PRS'))

    def setUp(self):
        torch.set_num_threads(1)
        self.cfg=replace(make_config('smoke'),episode_steps=12)
        self.tr=generate_trajectory(self.cfg,931,2.)

    def test_original_engine_matches_existing_simulator(self):
        cfg=self.cfg
        c=ConstrainedDQN(cfg,4*cfg.subarrays+2*cfg.users+2,11)
        old,_=simulate_episode(cfg,self.tr,'RA-FVV',c,1.,931,control_period=4)
        new=episode(cfg,self.tr,'Original-DQN',c)
        self.assertAlmostEqual(old.energy_j,new['energy_j'],places=9)
        self.assertAlmostEqual(old.outage_probability,new['outage'],places=12)

    def test_energy_components_sum_to_total(self):
        for m in ('PRS','Hysteresis PRS','Switch-aware PRS','Full array'):
            r=episode(self.cfg,self.tr,m)
            self.assertAlmostEqual(r['energy_j'],r['operating_j']+r['pilot_j']+r['switching_j'])
            self.assertTrue(np.isfinite(r['energy_j']))
            self.assertTrue(0<=r['outage']<=1)

    def test_absolute_gain_feature_distinguishes_uniform_attenuation(self):
        cfg=self.cfg
        hw=HardwareState(cfg)
        req=self.tr.requirements_mbps[0]
        a=VisibilityBelief(cfg,self.tr.gain_db[0])
        b=VisibilityBelief(cfg,self.tr.gain_db[0]-6.)
        x,_=state(cfg,a,hw,req,req,np.zeros(cfg.subarrays),0.,False)
        y,_=state(cfg,b,hw,req,req,np.zeros(cfg.subarrays),0.,False)
        np.testing.assert_allclose(x,y,atol=1e-6)
        x,_=state(cfg,a,hw,req,req,np.zeros(cfg.subarrays),0.,True)
        y,_=state(cfg,b,hw,req,req,np.zeros(cfg.subarrays),0.,True)
        self.assertGreater(np.max(abs(x-y)),.1)

    def test_enriched_training_updates_without_nonfinite_values(self):
        cfg=self.cfg
        dim=4*cfg.subarrays+2*cfg.users+2+cfg.subarrays*cfg.users+cfg.subarrays+cfg.users
        c=ConstrainedDQN(cfg,dim,11)
        for _ in range(22):
            episode(cfg,self.tr,'Enriched-DQN',c,True,.2)
        self.assertEqual(len(c.memory),66)
        self.assertGreater(c.steps,0)
        self.assertTrue(c.memory[-1].done)
        self.assertTrue(all(np.isfinite(t.energy_cost) for t in c.memory))

if __name__=='__main__':
    unittest.main()
