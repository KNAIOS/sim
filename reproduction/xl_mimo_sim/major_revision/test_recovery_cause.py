import copy
import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.action_value import observation, hold_predictions
from major_revision.full_domain import FullDomainRollout
from major_revision.horizon_prior import FixedHorizon
from major_revision.pilot import trajectory
from major_revision.recovery_cause import available_masks, forecast_grams_now, evaluate_grams, full_upper_bound, diagnose


class RecoveryCauseTests(unittest.TestCase):
    def test_same_prediction_and_no_mutation_with_delays(self):
        for delay in (0,4,9):
            cfg=replace(make_config('validation'),episode_steps=12,wake_delay_slots=delay)
            run=FullDomainRollout(cfg,trajectory(cfg,995900300+delay,(2.,1.)))
            run.prepare()
            run.hw.active[5:]=False
            run.hw.command[5:]=False
            if delay:
                run.hw.countdown[5:8]=[1,2,3]
            _,score=run.base_observation()
            before=copy.deepcopy(run.hw)
            p=hold_predictions(cfg,run.tracker,run.hw,score)
            available=available_masks(cfg,run.hw,p['masks'],cfg.horizon)
            grams=forecast_grams_now(cfg,run.tracker)
            rates=evaluate_grams(cfg,grams,available)
            np.testing.assert_allclose(rates,p['rates'],atol=1e-7,rtol=1e-9)
            self.assertTrue(np.all(rates<=full_upper_bound(cfg,grams)[None]+1e-6))
            for attr in ('active','command','countdown'):
                np.testing.assert_array_equal(getattr(run.hw,attr),getattr(before,attr))

    def test_oracle_evaluation_cannot_change_causal_policy_or_accounting(self):
        cfg=replace(make_config('validation'),episode_steps=12)
        tr=trajectory(cfg,995900310,(2.,1.))
        run=FullDomainRollout(cfg,tr)
        model=FixedHorizon(cfg)
        state,score=observation(run)
        action,count=model.choose(state)
        before=run.clone()
        row,outage=diagnose(run,action,score)
        state_after,_=observation(run)
        np.testing.assert_array_equal(state,state_after)
        self.assertEqual(model.choose(state_after),(action,count))
        self.assertEqual(row['user_slots'],12)
        for replay in (run,before):
            replay.step(select_top(score,count))
            for _ in range(3):
                replay.step()
        self.assertEqual(run.metrics(),before.metrics())
        self.assertAlmostEqual(outage,np.mean(run.outages),places=12)


if __name__=='__main__':
    unittest.main()
