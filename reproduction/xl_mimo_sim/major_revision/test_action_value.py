import copy
import unittest
from dataclasses import replace
import numpy as np
import torch
from future_vr.config import make_config
from future_vr.controllers import Transition,select_top
from major_revision.full_domain import FullDomainRollout
from major_revision.full_domain_baselines import baseline_episode
from major_revision.pilot import trajectory
from major_revision.action_value import (ActionValueDQN,hold_predictions,observation,
    action_episode,baseline_command,full_slot_energy,observation_size,prior_values)


class ActionValueTests(unittest.TestCase):
    def test_prediction_matches_full_channel_and_hardware_accounting(self):
        for delay in (0,4,9):
            cfg=replace(make_config('validation'),episode_steps=12,wake_delay_slots=delay)
            run=FullDomainRollout(cfg,trajectory(cfg,995900100+delay,(2.,1.)))
            run.prepare()
            run.hw.active[5:]=False
            run.hw.command[5:]=False
            if delay:
                run.hw.countdown[5:8]=[1,2,3]
                run.hw.command[5:8]=True
            _,score=run.base_observation()
            before=copy.deepcopy(run.hw)
            prediction=hold_predictions(cfg,run.tracker,run.hw,score)
            gains,_,_=run.tracker.predict(cfg.horizon)
            gains=np.concatenate((run.tracker.gain_db[None],gains[:-1]))
            h=run.tracker.channel_estimate
            power=np.mean(abs(h.reshape(cfg.users,cfg.subarrays,cfg.antennas_per_subarray))**2,axis=2).T
            scale=np.sqrt(10**(gains/10)/np.maximum(power[None],1e-20))
            channels=h[None]*np.repeat(scale.transpose(0,2,1),cfg.antennas_per_subarray,axis=2)
            for action in range(16):
                trial=copy.deepcopy(run.hw)
                mask=select_top(score,action+1)
                for step in range(8):
                    observed=int(trial.active.sum())
                    active,energy,_=trial.apply(mask,1.)
                    result=run.system.evaluate(channels[step],active,trial.countdown>0,energy,False,
                        run.req,observed_count=observed)
                    np.testing.assert_allclose(prediction['rates'][action,step],result.rates_mbps,
                                               atol=1e-5,rtol=1e-7)
                    self.assertAlmostEqual(prediction['energy'][action,step],result.energy_j,places=11)
                    self.assertEqual(prediction['active'][action,step],active.sum())
                    trial.tick()
            for attr in ('active','command','countdown'):
                np.testing.assert_array_equal(getattr(run.hw,attr),getattr(before,attr))
            full=run.system.evaluate(h,np.ones(16,bool),np.zeros(16,bool),0.,False,run.req,observed_count=16)
            self.assertAlmostEqual(full_slot_energy(cfg),full.energy_j,places=12)

    def test_gram_prs_replays_original(self):
        cfg=replace(make_config('validation'),episode_steps=24)
        for j,(minimum,margin) in enumerate(((1,1.04),(4,1.16))):
            tr=trajectory(cfg,995900120+j,(3.,1.15))
            a=action_episode(cfg,tr,mode='prs',minimum=minimum,margin=margin)
            b=baseline_episode(cfg,tr,'prs',margin,minimum)
            for k in ('energy_j','outage','operating_j','switching_j','pilot_j','mean_count'):
                self.assertAlmostEqual(a[k],b[k],places=9)

    def test_zero_residual_matches_nonlearning_prior(self):
        cfg=replace(make_config('validation'),episode_steps=16)
        tr=trajectory(cfg,995900130,(2.,1.))
        expected=action_episode(cfg,tr,mode='prior')
        for architecture in ('shared','flat'):
            model=ActionValueDQN(cfg,11,architecture)
            actual=action_episode(cfg,tr,model)
            for k in ('energy_j','outage','mean_count'):
                self.assertAlmostEqual(actual[k],expected[k],places=9)
        self.assertEqual(observation_size(cfg),1434)

    def test_training_uses_real_successor_and_updates_weights(self):
        cfg=replace(make_config('validation'),episode_steps=12)
        tr=trajectory(replace(cfg,episode_steps=13),995900140,(2.,1.))
        model=ActionValueDQN(cfg,11)
        action_episode(cfg,tr,model,training=True)
        self.assertEqual(len(model.memory),3)
        self.assertTrue(all(not t.done for t in model.memory))
        self.assertFalse(np.array_equal(model.memory[-1].state,model.memory[-1].next_state))
        last=model.memory[-1]
        for _ in range(65):
            model.remember(last)
        before=model.online.head[-1].weight.detach().clone()
        loss=model.learn()
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before,model.online.head[-1].weight))
        with self.assertRaises(ValueError):
            short=trajectory(cfg,995900140,(2.,1.))
            action_episode(cfg,short,model,training=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
