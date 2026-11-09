import unittest
from dataclasses import replace
from future_vr.config import make_config
from major_revision.pilot import episode,trajectory
from major_revision.search_headroom import SearchAuditRollout


class SearchHeadroomTests(unittest.TestCase):
    def test_search_engine_reproduces_baseline(self):
        for delay in (0,2,4,6):
            cfg = replace(make_config('validation'),episode_steps=20,wake_delay_slots=delay)
            tr = trajectory(cfg,979700000+delay,(3.,1.15))
            expected = episode(cfg,tr,'PRS',rate_margin=1.04)
            actual = SearchAuditRollout(cfg,tr).finish_search(1.04)
            actual['method']='PRS'
            self.assertEqual(expected,actual)


if __name__=='__main__':
    unittest.main()
