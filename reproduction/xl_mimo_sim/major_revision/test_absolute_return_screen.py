import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from major_revision.action_value import hold_predictions,action_episode
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.absolute_return_screen import current_table,search_command,episode


class AbsoluteReturnScreenTests(unittest.TestCase):
    def test_current_table_changes_no_future_or_energy_and_matches_causal_csi(self):
        cfg=replace(make_config('validation'),episode_steps=20)
        run=FullDomainRollout(cfg,trajectory(cfg,1060900001,(2.,1.)))
        _,score=run.base_observation()
        run.step(select_top(score,4))
        for _ in range(3):
            run.step()
        before,score=run.base_observation()
        old=hold_predictions(cfg,run.tracker,run.hw,score)
        table,_=current_table(run)
        np.testing.assert_array_equal(old['rates'][:,1:],table['rates'][:,1:])
        np.testing.assert_array_equal(old['energy'],table['energy'])
        for i,mask in enumerate(table['masks']):
            rates=run.system.achievable_rates(run.tracker.channel_estimate,mask&run.hw.active)
            np.testing.assert_allclose(table['rates'][i,0],rates,atol=1e-6,rtol=1e-10)
            trial=run.clone()
            trial.step(mask)
            self.assertAlmostEqual(float(np.mean(rates<run.req)),trial.outages[-1],places=12)
        for mode in ('current-prs','current-cost'):
            command=search_command(run,mode,1.04,1)
            self.assertGreaterEqual(command.sum(),1)
            self.assertLessEqual(command.sum(),16)
        np.testing.assert_array_equal(before,run.base_observation()[0])

    def test_old_search_replay_is_unchanged(self):
        cfg=replace(make_config('validation'),episode_steps=20)
        tr=trajectory(cfg,1060900002,(3.,1.15))
        for mode in ('prs','cost','full'):
            old=action_episode(cfg,tr,mode=mode,minimum=4,margin=1.04)
            new=episode(cfg,tr,mode,dict(minimum=4,margin=1.04))
            for key in ('energy_j','outage','switching_j','pilot_j','operating_j','transitions','mean_count'):
                self.assertEqual(old[key],new[key])


if __name__=='__main__':
    unittest.main()
