import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from future_vr.config import make_config
from future_vr.belief import VisibilityBelief
from future_vr.controllers import readiness_aligned_value
from future_vr.experiment import _prediction, simulate_episode
from future_vr.channel import generate_trajectory
from future_vr.system import HardwareState, XLSystem


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(make_config('smoke'), episode_steps=12)

    def test_current_csi_does_not_call_predict(self):
        belief = VisibilityBelief(self.cfg, np.full((10, 2), -80.0))
        with patch.object(belief, 'predict', side_effect=AssertionError('future access')):
            score, gain = _prediction(self.cfg, belief, HardwareState(self.cfg),
                                      np.array([400., 400.]), future=False, readiness=False)
        np.testing.assert_equal(gain[0], belief.gain_db)

    def test_ready_beyond_horizon_has_zero_value(self):
        shape = (3, 2, 2)
        score = readiness_aligned_value(self.cfg, np.zeros(shape), np.ones(shape),
                                       np.zeros(shape), np.array([4, 0]), np.ones(2))
        self.assertEqual(score[0], 0)
        self.assertGreater(score[1], 0)

    def test_only_active_chain_observations_after_initialization(self):
        trajectory = generate_trajectory(self.cfg, 901, 1.0)
        masks = []
        original = VisibilityBelief.update
        def spy(tracker, obs, mask):
            masks.append(mask.copy())
            original(tracker, obs, mask)
        with patch.object(VisibilityBelief, 'update', spy):
            simulate_episode(self.cfg, trajectory, 'RA-FVV matched', None, 1., 901,
                             forced_counts=np.full(12, 3), control_period=4)
        self.assertEqual(masks[0].sum(), 10)
        self.assertTrue(all(mask.sum() <= 3 for mask in masks[1:]))

    def test_pilot_energy_and_rate_accounting(self):
        tr = generate_trajectory(self.cfg, 902, 1.)
        full = np.ones(10, bool)
        system = XLSystem(self.cfg)
        a = system.evaluate(tr.channel[0], full, ~full, 0, False, tr.requirements_mbps[0], 10)
        b = system.evaluate(tr.channel[0], full, ~full, 0, False, tr.requirements_mbps[0], 0)
        self.assertAlmostEqual(a.energy_j-b.energy_j, 10*self.cfg.scan_energy_j_per_subarray)
        self.assertAlmostEqual(a.delivered_mbit, a.rates_mbps.sum()*self.cfg.slot_s)

    def test_rate_search_runs_with_causal_observations(self):
        tr = generate_trajectory(self.cfg, 903, 1.)
        metric, _ = simulate_episode(self.cfg, tr, 'Predictive rate search', None, 1., 903)
        self.assertTrue(np.isfinite(metric.energy_j))

if __name__ == '__main__':
    unittest.main()
