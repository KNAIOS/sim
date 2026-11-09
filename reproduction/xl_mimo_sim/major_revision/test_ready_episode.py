import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from major_revision.pilot import episode, trajectory
from major_revision.rate_context import candidate_rate_context
from major_revision.ready_episode import ready_episode


class RecordingController:
    def __init__(self,cfg):
        self.cfg, self.calls, self.memory, self.observations = cfg, 0, [], []
    def choose(self,observation,epsilon):
        self.observations.append(observation.copy())
        counts = (3,7,4,10,5)
        count = min(self.cfg.subarrays,max(self.cfg.users+1,counts[self.calls%len(counts)]))
        self.calls += 1
        return count-self.cfg.users-1,count
    def remember(self,item):
        self.memory.append(item)
    def learn(self):
        return 0.


class ReadyEpisodeTests(unittest.TestCase):
    def test_longer_channel_generation_preserves_prefix(self):
        cfg = replace(make_config('smoke'),episode_steps=20)
        a = trajectory(cfg,979300000,(2.,1.))
        b = trajectory(replace(cfg,episode_steps=21),979300000,(2.,1.))
        for field in a.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(a,field),getattr(b,field)[:20])

    def test_truncation_requires_a_real_successor_observation(self):
        cfg = replace(make_config('smoke'),episode_steps=20)
        tr = trajectory(cfg,979300001,(2.,1.))
        with self.assertRaises(ValueError):
            ready_episode(cfg,tr,RecordingController(cfg),True,bootstrap_truncation=True)

    def test_truncation_successor_matches_next_actual_decision(self):
        cfg = replace(make_config('smoke'),episode_steps=20)
        extended = replace(cfg,episode_steps=24)
        tr = trajectory(extended,979300002,(2.,1.))
        limited,continued = RecordingController(cfg),RecordingController(extended)
        before = ready_episode(cfg,tr,limited,True,bootstrap_truncation=True)
        ready_episode(extended,tr,continued,True)
        final = limited.memory[-1]
        self.assertFalse(final.done)
        np.testing.assert_array_equal(final.next_state,continued.observations[5])
        self.assertEqual(final.energy_cost,continued.memory[4].energy_cost)
        self.assertEqual(final.outage_cost,continued.memory[4].outage_cost)
        terminal = RecordingController(cfg)
        same_window = ready_episode(cfg,tr,terminal,True)
        self.assertEqual(before,same_window)
        self.assertTrue(terminal.memory[-1].done)
        self.assertEqual(len(limited.observations),5)

    def test_disabled_feature_matches_physics_and_replay(self):
        for delay in (0,2,4,6):
            cfg = replace(make_config('smoke'),episode_steps=23,wake_delay_slots=delay)
            tr = trajectory(cfg, 979100000+delay,(2.,1.))
            old,new = RecordingController(cfg),RecordingController(cfg)
            a = episode(cfg,tr,'Enriched-DQN',old,True,state_features=candidate_rate_context)
            b = ready_episode(cfg,tr,new,True,include_ready=False)
            self.assertEqual(a,b)
            self.assertEqual(len(old.memory),len(new.memory))
            for x,y in zip(old.memory,new.memory):
                np.testing.assert_array_equal(x.state,y.state)
                np.testing.assert_array_equal(x.next_state,y.next_state)
                self.assertEqual((x.action,x.energy_cost,x.outage_cost,x.done),
                                 (y.action,y.energy_cost,y.outage_cost,y.done))

    def test_added_feature_does_not_change_fixed_policy_physics(self):
        cfg = replace(make_config('validation'),episode_steps=19)
        tr = trajectory(cfg,979200000,(3.,1.15))
        plain,ready = RecordingController(cfg),RecordingController(cfg)
        a = ready_episode(cfg,tr,plain,True,include_ready=False)
        b = ready_episode(cfg,tr,ready,True,include_ready=True)
        self.assertEqual(a,b)
        self.assertEqual(plain.memory[0].state.size,178)
        self.assertEqual(ready.memory[0].state.size,217)
        for x,y in zip(plain.memory,ready.memory):
            np.testing.assert_array_equal(x.state,y.state[:178])
            self.assertTrue(np.isfinite(y.state).all())
            self.assertTrue(np.all((y.state[178:]>=0)&(y.state[178:]<=4)))


if __name__ == '__main__':
    unittest.main()
