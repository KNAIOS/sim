import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.pilot import trajectory
from major_revision.ready_episode import ready_episode
from major_revision.audit_rollout import AuditRollout


class ObservedPolicy:
    def __init__(self, cfg):
        self.cfg, self.states = cfg, []
    def choose(self, observation, epsilon):
        self.states.append(observation.copy())
        # State-dependent, not a call-index-dependent policy: forks share the
        # same observation-to-action rule without hidden mutable policy state.
        n = self.cfg.users+1 + int(abs(float(observation[-1]))*17) % (self.cfg.subarrays-self.cfg.users)
        return n-self.cfg.users-1,n


class AuditRolloutTests(unittest.TestCase):
    def test_matches_existing_engine(self):
        for delay in (0,2,4,6):
            cfg = replace(make_config('smoke'), episode_steps=23,wake_delay_slots=delay)
            tr = trajectory(cfg,979600000+delay,(2.,1.))
            old,new = ObservedPolicy(cfg),ObservedPolicy(cfg)
            expected = ready_episode(cfg,tr,old)
            actual = AuditRollout(cfg,tr).finish(new)
            self.assertEqual(expected,actual)
            for a,b in zip(old.states,new.states):
                np.testing.assert_array_equal(a,b)
            self.assertEqual(len(old.states),len(new.states))

    def test_fork_replays_same_action_and_does_not_mutate_parent(self):
        cfg = replace(make_config('validation'),episode_steps=24)
        tr = trajectory(cfg,979600010,(3.,1.15))
        run = AuditRollout(cfg,tr)
        policy = ObservedPolicy(cfg)
        while run.t < 8:
            if run.t % 4 == 0:
                obs,score = run.observation()
                run.step(select_top(score,policy.choose(obs,0.)[1]))
            else:
                run.step()
        obs,score = run.observation()
        command = select_top(score,policy.choose(obs,0.)[1])
        frozen_state = (run.t,run.energy,run.hw.active.copy(),run.tracker.gain_db.copy())
        a,b = run.clone(),run.clone()
        self.assertEqual(a.finish(policy,command),b.finish(policy))
        self.assertEqual((run.t,run.energy),frozen_state[:2])
        np.testing.assert_array_equal(run.hw.active,frozen_state[2])
        np.testing.assert_array_equal(run.tracker.gain_db,frozen_state[3])
        self.assertEqual(run.finish(policy),a.metrics())

    def test_repeated_prepare_does_not_age_or_update_twice(self):
        cfg = replace(make_config('smoke'),episode_steps=8)
        run = AuditRollout(cfg,trajectory(cfg,979600011,(2.,1.)))
        a,s = run.observation()
        b,t = run.observation()
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(s,t)
        with self.assertRaises(ValueError):
            run.step()


if __name__ == '__main__':
    unittest.main()
