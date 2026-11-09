import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from major_revision.audit_rollout import AuditRollout
from major_revision.pilot import trajectory
from major_revision.reward_constraint_audit import discounted_metrics


class RewardAuditTests(unittest.TestCase):
    def test_gamma_one_matches_episode_cost(self):
        cfg=replace(make_config('validation'),episode_steps=8,rl_gamma=1.)
        run=AuditRollout(cfg,trajectory(cfg,979900000,(2.,1.)))
        run.slot_energy=[1.,2.,3.,4.,2.,3.,4.,5.]
        run.outages=[0.,0.,1/3,0.,0.,1/3,0.,0.]
        result=discounted_metrics(run)
        self.assertAlmostEqual(result['discounted_energy_cost'],result['undiscounted_energy_cost'])
        self.assertAlmostEqual(result['discounted_outage_cost'],np.mean(run.outages))
        self.assertAlmostEqual(result['discounted_price4_cost'],result['undiscounted_price4_cost'])

    def test_earlier_block_gets_more_weight(self):
        cfg=replace(make_config('validation'),episode_steps=8,rl_gamma=.5)
        run=AuditRollout(cfg,trajectory(cfg,979900001,(2.,1.)))
        run.slot_energy=[1.]*8
        run.outages=[1.]*4+[0.]*4
        self.assertAlmostEqual(discounted_metrics(run)['discounted_outage_cost'],2/3)


if __name__=='__main__':
    unittest.main()
