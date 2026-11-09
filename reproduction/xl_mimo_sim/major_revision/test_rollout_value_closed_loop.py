import unittest
from dataclasses import replace
import torch
from future_vr.config import make_config
from major_revision.action_value import action_episode
from major_revision.pilot import trajectory
from major_revision.rollout_value_fit import DifferenceNetwork
from major_revision.rollout_value_closed_loop import learned_episode


class RolloutClosedLoopTests(unittest.TestCase):
    def test_zero_difference_reproduces_causal_search_exactly(self):
        cfg=replace(make_config('validation'),episode_steps=20)
        tr=trajectory(cfg,1000900100,(2.,1.))
        model=DifferenceNetwork(cfg)
        for setting in (dict(margin=1.08,minimum=1),dict(margin=1.04,minimum=4)):
            baseline=action_episode(cfg,tr,mode='cost',**setting)
            learned=learned_episode(cfg,tr,model,setting,8.)
            for key in ('energy_j','outage','switching_j','pilot_j','operating_j','mean_count'):
                self.assertAlmostEqual(baseline[key],learned[key],places=11)
            self.assertEqual(learned['changed_from_mu_fraction'],0.)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
