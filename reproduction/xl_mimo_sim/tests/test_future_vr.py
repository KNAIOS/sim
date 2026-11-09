from __future__ import annotations

import unittest
import numpy as np

from future_vr.belief import VisibilityBelief
from future_vr.channel import generate_trajectory
from future_vr.config import make_config
from future_vr.controllers import readiness_aligned_value, select_top
from future_vr.experiment import _symmetric_count_mask, _variant_flags
from future_vr.matched_policy_study import DISPLAY, METHODS
from future_vr.system import HardwareState, XLSystem


class FutureVRTests(unittest.TestCase):
    def test_matched_policy_methods_have_unique_display_names(self) -> None:
        self.assertEqual(len(METHODS), len({DISPLAY[method] for method in METHODS}))
        self.assertEqual(DISPLAY["RA-FVV matched"], "RA-CRL")

    def test_learned_variants_isolate_forecast_and_readiness(self) -> None:
        self.assertEqual(_variant_flags("CSI-CRL"), (False, False))
        self.assertEqual(_variant_flags("F-CRL"), (True, False))
        self.assertEqual(_variant_flags("RA-FVV"), (True, True))

    def setUp(self) -> None:
        self.cfg = make_config("smoke")

    def test_channel_is_reproducible_finite_and_bounded(self) -> None:
        first = generate_trajectory(self.cfg, 71, 1.5)
        second = generate_trajectory(self.cfg, 71, 1.5)
        self.assertTrue(np.allclose(first.channel, second.channel))
        self.assertTrue(np.isfinite(first.gain_db).all())
        self.assertTrue(((first.visibility >= 0.0) & (first.visibility <= 1.0)).all())
        self.assertEqual(first.channel.shape, (self.cfg.episode_steps, self.cfg.users, self.cfg.antennas))

    def test_full_support_prediction_has_expected_shapes(self) -> None:
        trajectory = generate_trajectory(self.cfg, 72, 2.0)
        belief = VisibilityBelief(self.cfg, trajectory.gain_db[0])
        belief.update(trajectory.gain_db[1], np.arange(self.cfg.subarrays) % 2 == 0)
        gain, probability, uncertainty = belief.predict(self.cfg.horizon)
        shape = (self.cfg.horizon, self.cfg.subarrays, self.cfg.users)
        self.assertEqual(gain.shape, shape)
        self.assertEqual(probability.shape, shape)
        self.assertEqual(uncertainty.shape, shape)
        self.assertTrue(((probability > 0.0) & (probability < 1.0)).all())

    def test_wake_delay_blocks_rf_until_countdown_completes(self) -> None:
        hardware = HardwareState(self.cfg)
        hardware.apply(np.zeros(self.cfg.subarrays, dtype=bool), 1.0)
        command = np.zeros(self.cfg.subarrays, dtype=bool)
        command[2] = True
        available, _, _ = hardware.apply(command, 1.0)
        self.assertFalse(available[2])
        for _ in range(self.cfg.wake_delay_slots):
            hardware.tick()
        self.assertTrue(hardware.active[2])

    def test_readiness_count_matches_future_horizon_index(self) -> None:
        hardware = HardwareState(self.cfg)
        hardware.apply(np.zeros(self.cfg.subarrays, dtype=bool), 1.0)
        command = np.zeros(self.cfg.subarrays, dtype=bool)
        command[1] = True
        hardware.apply(command, 1.0)
        self.assertEqual(hardware.readiness()[1], self.cfg.wake_delay_slots)
        for remaining in range(self.cfg.wake_delay_slots - 1, 0, -1):
            hardware.tick()
            self.assertEqual(hardware.readiness()[1], remaining)
            self.assertFalse(hardware.active[1])
        hardware.tick()
        self.assertTrue(hardware.active[1])

    def test_readiness_alignment_produces_finite_ranking(self) -> None:
        h, q, k = self.cfg.horizon, self.cfg.subarrays, self.cfg.users
        gain = np.full((h, q, k), -100.0)
        probability = np.full((h, q, k), 0.1)
        uncertainty = np.full((h, q, k), 0.1)
        gain[:, 1, :] = -70.0
        probability[:, 1, :] = 0.9
        gain[-2:, 7, :] = -68.0
        probability[-2:, 7, :] = 0.95
        readiness = np.zeros(q, dtype=int)
        readiness[7] = h - 2
        score = readiness_aligned_value(
            self.cfg, gain, probability, uncertainty, readiness, np.full(k, 400.0),
        )
        self.assertIn(1, np.flatnonzero(select_top(score, 2)))
        self.assertTrue(np.isfinite(score).all())

    def test_readiness_value_starts_at_the_dth_future_slot(self) -> None:
        h, q, k = 4, 2, self.cfg.users
        gain = np.full((h, q, k), -120.0)
        probability = np.zeros((h, q, k))
        uncertainty = np.zeros_like(probability)
        gain[2, 0, :] = -70.0
        probability[2, 0, :] = 1.0
        gain[3, 1, :] = -70.0
        probability[3, 1, :] = 1.0
        readiness = np.array([3, 4])
        score = readiness_aligned_value(
            self.cfg, gain, probability, uncertainty, readiness,
            np.full(k, 400.0),
        )
        self.assertGreater(score[0], 0.0)
        self.assertGreater(score[1], 0.0)
        self.assertGreater(score[0], score[1])

    def test_system_reports_finite_energy_and_rate(self) -> None:
        trajectory = generate_trajectory(self.cfg, 73, 1.0)
        full = np.ones(self.cfg.subarrays, dtype=bool)
        result = XLSystem(self.cfg).evaluate(
            trajectory.channel[0], full, np.zeros_like(full), 0.0, False, trajectory.requirements_mbps[0]
        )
        self.assertGreater(result.energy_j, 0.0)
        self.assertTrue(np.isfinite(result.rates_mbps).all())
        self.assertGreaterEqual(result.outage, 0.0)
        self.assertLessEqual(result.outage, 1.0)

    def test_fixed_control_period_is_independent_of_wake_delay(self) -> None:
        trajectory = generate_trajectory(self.cfg, 74, 1.0)
        forced = np.full(self.cfg.episode_steps, 3, dtype=int)
        metric, trace = __import__(
            "future_vr.experiment", fromlist=["simulate_episode"]
        ).simulate_episode(
            self.cfg.with_delay(5), trajectory, "RA-FVV matched", None, 1.0, 74,
            forced_counts=forced, control_period=2,
        )
        self.assertTrue(np.isfinite(metric.energy_j))
        self.assertTrue(np.all(trace["action_count"] == 3))

    def test_vanilla_mask_is_contiguous_and_has_requested_count(self) -> None:
        score = np.array([0.0, 0.1, 0.8, 1.0, 0.7, 0.1, 0.0])
        mask = _symmetric_count_mask(score, 3)
        selected = np.flatnonzero(mask)
        self.assertEqual(mask.sum(), 3)
        self.assertTrue(np.all(np.diff(selected) == 1))


if __name__ == "__main__":
    unittest.main()
