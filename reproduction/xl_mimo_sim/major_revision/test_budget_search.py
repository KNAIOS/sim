import copy
import unittest
from dataclasses import replace

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.experiment import _method_command, _prediction
from future_vr.rate_search import rate_search_command
from major_revision.audit_rollout import AuditRollout
from major_revision.budget_search import STRIDES, forecasts, predicted_rates, search, learned, stride_counts, local_counts
from major_revision.pilot import trajectory
from major_revision.power_policy_study import controllers


class BudgetSearchTests(unittest.TestCase):
    def test_count_grids_and_neighborhoods(self):
        expected={1:13,2:7,3:5,4:4,6:3,12:2}
        for stride in STRIDES:
            counts=stride_counts(4,16,stride)
            self.assertEqual(len(counts),expected[stride])
            for current in range(4,17):
                local=local_counts(4,16,current,len(counts))
                self.assertEqual(len(local),len(counts))
                self.assertEqual((local[0],local[-1]),(4,16))
        np.testing.assert_array_equal(stride_counts(4,16,2),[4,6,8,10,12,14,16])
        with self.assertRaises(ValueError):
            stride_counts(4,16,0)

    def test_gram_matches_original_forecast_rates(self):
        cfg=replace(make_config('validation'),episode_steps=16)
        tr=trajectory(cfg,1700000001,(2.,1.))
        run=AuditRollout(cfg,tr)
        for t in range(16):
            run.prepare()
            score,_=_prediction(cfg,run.tracker,run.hw,run.req)
            h=forecasts(cfg,run.tracker)
            counts=np.arange(4,17)
            batch=predicted_rates(cfg,run.tracker,run.hw,score,counts)
            direct=np.array([[run.system.achievable_rates(h[k],select_top(score,m)&(run.hw.readiness()<=k+1))
                              for k in range(cfg.horizon)] for m in counts])
            np.testing.assert_allclose(batch,direct,rtol=1e-9,atol=1e-6)
            full=search(cfg,run.tracker,run.hw,run.req)
            original=rate_search_command(cfg,run.tracker,run.hw,run.req,score)
            np.testing.assert_array_equal(full,original)
            run.step(full if t%4==0 else None)

    def test_lean_rl_has_same_action_and_does_not_train(self):
        torch.set_num_threads(1)
        cfg=replace(make_config('validation'),episode_steps=8)
        tr=trajectory(cfg,1700000002,(2.,1.))
        models=controllers(cfg)
        run=AuditRollout(cfg,tr)
        run.prepare()
        for model in models.values():
            old=copy.deepcopy(model.online.state_dict())
            result=learned(cfg,run.tracker,run.hw,run.req,0.,model)
            original=_method_command('RA-FVV',cfg,run.tracker,run.hw,run.req,model,
                tr,0,np.zeros(cfg.subarrays,bool),1.,0.)[0]
            np.testing.assert_array_equal(result,original)
            for key,value in old.items():
                torch.testing.assert_close(value,model.online.state_dict()[key],rtol=0,atol=0)

    def test_zero_channel_and_ranking_ties(self):
        cfg=make_config('validation')
        tr=trajectory(cfg,1700000003,(.5,.85))
        run=AuditRollout(cfg,tr)
        run.prepare()
        run.tracker.channel_estimate[:]=0.
        rates=predicted_rates(cfg,run.tracker,run.hw,np.ones(16),np.arange(4,17))
        np.testing.assert_array_equal(rates,np.zeros((13,8,3)))


if __name__=='__main__':
    unittest.main()
