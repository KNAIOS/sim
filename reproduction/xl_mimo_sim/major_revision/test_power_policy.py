"""Adapter/statistics tests. Artificial statistics fixtures are not experiment data."""
import unittest
from dataclasses import replace

import numpy as np
import torch

from future_vr.config import make_config
from future_vr.experiment import simulate_episode
from major_revision.audit_rollout import AuditRollout
from major_revision.pilot import trajectory
from major_revision.power_policy_study import SEEDS, command, controllers, summarize


class PowerPolicyTests(unittest.TestCase):
    def test_original_policy_adapter(self):
        torch.set_num_threads(1)
        cfg = replace(make_config('validation'), episode_steps=8)
        tr = trajectory(cfg, 12340001, (2.,1.))
        models = controllers(cfg)
        for seed in (11,33,0):
            run = AuditRollout(cfg,tr)
            while run.t < cfg.episode_steps:
                run.step(command(run,models.get(seed)) if run.t%4==0 else None)
            _, original = simulate_episode(cfg,tr,'RA-FVV' if seed else 'Predictive rate search',
                                            models.get(seed),1.,12340001,control_period=4)
            np.testing.assert_array_equal(run.slot_energy,original['energy_j'])
            np.testing.assert_array_equal(run.outages,original['outage'])
            np.testing.assert_array_equal(run.counts,original['action_count'])

    def test_paired_gap_sign_and_zero_variation(self):
        # Deliberately constant, invented unit-test fixture; never saved as research output.
        rows=[]
        for j in range(3):
            for method,energy,saving in [('RA-DQN',140.,15.),('Manuscript PRS',120.,12.),
                                        ('Calibrated search',125.,13.),('Full array',300.,20.)]:
                for seed in SEEDS if method=='RA-DQN' else (0,):
                    for i in range(8):
                        rows.append(dict(scenario=j,trajectory=i,method=method,controller_seed=seed,
                            original_energy_j=energy,counterfactual_energy_j=energy-saving,
                            saving_j=saving,original_outage=.03))
        summary,gaps,per_seed=summarize(rows)
        self.assertEqual((len(summary),len(gaps),len(per_seed)),(12,24,24))
        for row in gaps:
            self.assertEqual(row['mean'],row['ci95_low'])
            self.assertEqual(row['mean'],row['ci95_high'])
            if row['baseline']=='Manuscript PRS':
                expected={'original_energy_gap_j':20.,'adjusted_energy_gap_j':17.,
                          'gap_change_j':-3.,'outage_gap_pp':0.}
                self.assertEqual(row['mean'],expected[row['metric']])


if __name__ == '__main__':
    unittest.main()
