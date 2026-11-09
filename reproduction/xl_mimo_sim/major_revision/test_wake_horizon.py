import unittest
import numpy as np
from future_vr.config import make_config
from major_revision.action_value import hold_predictions
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.wake_horizon_audit import pair_checks


class WakeHorizonTests(unittest.TestCase):
    def test_sleeping_addition_has_cost_but_no_service_in_first_block(self):
        cfg=make_config('validation')
        run=FullDomainRollout(cfg,trajectory(cfg,995900160,(2.,1.)))
        run.prepare()
        run.hw.active[:8]=False
        run.hw.command[:8]=False
        _,score=run.base_observation()
        prediction=hold_predictions(cfg,run.tracker,run.hw,score)
        rows=pair_checks(cfg,run.hw,prediction,run.req)
        self.assertGreater(len(rows),0)
        for row in rows:
            self.assertAlmostEqual(row['first_block_energy_delta_j'],.33,places=10)
            self.assertEqual(row['first_block_outage_delta'],0.)
            self.assertGreater(row['first_block_cost_delta'],0.)


if __name__=='__main__':
    unittest.main()
