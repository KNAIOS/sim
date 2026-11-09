import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.action_value import observation,full_slot_energy
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.rollout_value_data import branch_return,collect_trace


class RolloutValueDataTests(unittest.TestCase):
    def test_branch_cost_units_isolation_and_horizon(self):
        cfg=replace(make_config('validation'),episode_steps=20)
        run=FullDomainRollout(cfg,trajectory(cfg,1000900001,(2.,1.)))
        before,score=observation(run)
        command=select_top(score,16)
        setting=dict(margin=1.08,minimum=1)
        values,blocks=branch_return(run,command,setting)
        self.assertAlmostEqual(values[0,0],1.,places=12)
        np.testing.assert_allclose(values[1],(blocks*cfg.rl_gamma**np.arange(4)[:,None]).sum(0))
        self.assertEqual(run.t,0)
        self.assertEqual(run.energy,0.)
        np.testing.assert_array_equal(observation(run)[0],before)
        a=branch_return(run,select_top(score,2),setting)[0]
        b=branch_return(run,select_top(score,2),setting)[0]
        np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):
            branch_return(run,command,setting,24)

    def test_baseline_return_replays_actual_suffix_and_target_zero(self):
        cfg=replace(make_config('validation'),episode_steps=20)
        tr=trajectory(cfg,1000900002,(3.,1.15))
        x,ref,costs,difference,metrics,error=collect_trace(cfg,tr,dict(margin=1.04,minimum=4),times=(0,4))
        self.assertEqual(x.shape,(2,1436))
        np.testing.assert_allclose(x[:,-2:],np.tile([1.04,.25],(2,1)))
        self.assertEqual(costs.shape,(2,16,2,2))
        self.assertLess(error,1e-10)
        np.testing.assert_array_equal(difference[np.arange(2),ref],0.)
        np.testing.assert_allclose(difference,costs-costs[np.arange(2),ref][:,None])
        self.assertGreater(metrics['energy_j'],0.)


if __name__=='__main__':
    unittest.main()
