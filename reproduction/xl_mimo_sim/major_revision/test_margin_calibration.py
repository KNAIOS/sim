import unittest
from dataclasses import replace
from future_vr.config import make_config
from major_revision.pilot import episode,trajectory
from major_revision.margin_calibration import CostAuditRollout


class MarginCalibrationTests(unittest.TestCase):
    def test_cost_replay_matches_original(self):
        for delay in (0,2,4,6):
            cfg=replace(make_config('validation'),episode_steps=20,wake_delay_slots=delay)
            tr=trajectory(cfg,979910000+delay,(3.,1.15))
            expected=episode(cfg,tr,'Switch-aware PRS',rate_margin=1.04)
            actual=CostAuditRollout(cfg,tr).finish_search(1.04)
            actual['method']='Switch-aware PRS'
            self.assertEqual(expected,actual)

    def test_subunit_margin_changes_only_predictor_threshold(self):
        cfg=replace(make_config('validation'),episode_steps=12)
        tr=trajectory(cfg,979910011,(3.,1.15))
        before=tr.requirements_mbps.copy()
        actual=CostAuditRollout(cfg,tr).finish_search(.95)
        self.assertTrue((before==tr.requirements_mbps).all())
        self.assertGreater(actual['energy_j'],0.)
        self.assertTrue(0<=actual['outage']<=1)


if __name__=='__main__':
    unittest.main()
