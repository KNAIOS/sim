from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xlsleep.channel import generate_trajectory
from xlsleep.config import make_config
from xlsleep.system import XLSystem
from xlsleep.controllers import ranked_mask, utility_score


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = make_config("smoke")

    def test_trajectory_is_reproducible_and_finite(self) -> None:
        a = generate_trajectory(self.cfg, 7, 1.5)
        b = generate_trajectory(self.cfg, 7, 1.5)
        self.assertTrue(np.allclose(a.channel, b.channel))
        self.assertTrue(np.isfinite(a.subarray_gain_db).all())
        self.assertEqual(a.channel.shape, (self.cfg.episode_steps, self.cfg.users, self.cfg.antennas))
        self.assertTrue(np.array_equal(a.rate_requirement_mbps, b.rate_requirement_mbps))
        self.assertTrue(set(np.unique(a.rate_requirement_mbps)).issubset({140.0, 180.0, 220.0}))

    def test_visibility_is_cluster_specific_and_bounded(self) -> None:
        trajectory = generate_trajectory(self.cfg, 8, 1.5)
        self.assertTrue((trajectory.visibility >= 0.0).all())
        self.assertTrue((trajectory.visibility <= 1.0).all())
        self.assertEqual(trajectory.visibility.shape[0], self.cfg.episode_steps)
        self.assertTrue(np.isfinite(trajectory.active_clusters).all())
        self.assertFalse(np.allclose(trajectory.visibility[:, 1], trajectory.visibility[:, 2]))

    def test_full_array_link_is_finite(self) -> None:
        trajectory = generate_trajectory(self.cfg, 9, 1.5)
        system = XLSystem(self.cfg)
        mask = np.ones(self.cfg.subarrays, dtype=bool)
        result = system.evaluate(trajectory.channel[0], mask, mask, 1.0)
        self.assertTrue(np.isfinite(result.rates_mbps).all())
        self.assertGreater(result.energy_j, 0.0)

    def test_transition_energy_is_accounted(self) -> None:
        trajectory = generate_trajectory(self.cfg, 10, 1.5)
        system = XLSystem(self.cfg)
        full = np.ones(self.cfg.subarrays, dtype=bool)
        half = np.arange(self.cfg.subarrays) % 2 == 0
        no_switch = system.evaluate(trajectory.channel[0], half, half, 2.0)
        switched = system.evaluate(trajectory.channel[0], half, full, 2.0)
        expected = np.count_nonzero(half != full) * self.cfg.switch_energy_j_per_subarray * 2.0
        self.assertAlmostEqual(switched.energy_j - no_switch.energy_j, expected)

    def test_ranking_can_select_noncontiguous_subarrays(self) -> None:
        mean = np.full((self.cfg.subarrays, self.cfg.users), -100.0)
        mean[[0, 3, 7]] = -70.0
        mask = ranked_mask(utility_score(mean), 3)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 3, 7])


if __name__ == "__main__":
    unittest.main()
