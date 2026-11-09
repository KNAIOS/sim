import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from major_revision.full_domain import FullDomainRollout
from major_revision.full_domain_baselines import cost_command,baseline_episode
from major_revision.pilot import trajectory,cost_search,episode


class BaselineCorrectionTests(unittest.TestCase):
    def test_filtered_current_gain_matches_original(self):
        cfg=replace(make_config('validation'),episode_steps=16)
        tr=trajectory(cfg,981123456,(3.,1.15))
        run=FullDomainRollout(cfg,tr,4)
        run.prepare()
        # Deliberately separate filtered gain from last-observed magnitude.
        run.tracker.gain_db+=np.linspace(-12,12,cfg.subarrays)[:,None]
        _,score=run.base_observation()
        for margin in (.9,1.,1.16):
            expected=cost_search(cfg,run.tracker,run.hw,run.req*margin,score)
            np.testing.assert_array_equal(expected,cost_command(run,margin))

    def test_physical_replay(self):
        cfg=replace(make_config('validation'),episode_steps=40)
        for seed in (981222211,981333322):
            tr=trajectory(cfg,seed,(3.,1.15))
            old=episode(cfg,tr,'Switch-aware PRS',rate_margin=1.08)
            new=baseline_episode(cfg,tr,'cost',1.08,minimum=4)
            for key,value in old.items():
                if key!='method':
                    self.assertAlmostEqual(value,new[key],places=9)


if __name__=='__main__':
    unittest.main()
