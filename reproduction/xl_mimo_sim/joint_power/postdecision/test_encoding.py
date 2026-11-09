import copy
import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from future_vr.system import HardwareState
from joint_power.postdecision.encoding import hardware_features


class HardwareEncodingTests(unittest.TestCase):
    def test_all_timers_commands_match_hardware(self):
        rng = np.random.default_rng(2026092211)
        for delay in (0, 1, 4, 7):
            cfg = replace(make_config('validation'), wake_delay_slots=delay)
            for _ in range(10):
                hw = HardwareState(cfg)
                hw.active = rng.random(16) < .5
                hw.countdown = np.where(hw.active, 0, rng.integers(0, delay+1, 16))
                masks = rng.random((17, 16)) < .5
                features = hardware_features(hw, masks)
                for action, command in enumerate(masks):
                    real = copy.deepcopy(hw)
                    active, _, _ = real.apply(command, 1.)
                    np.testing.assert_array_equal(features[action, :, 0], command)
                    np.testing.assert_array_equal(features[action, :, 1], active)
                    np.testing.assert_allclose(features[action, :, 2]*max(delay, 1), real.countdown, atol=1e-6)
                    for _ in range(4):
                        real.tick()
                    np.testing.assert_array_equal(features[action, :, 3], real.active)
                    np.testing.assert_allclose(features[action, :, 4]*max(delay, 1), real.countdown, atol=1e-6)

    def test_positions_not_only_cardinality(self):
        hw = HardwareState(make_config('validation'))
        masks = np.array([np.arange(16) < 4, np.arange(16) >= 12])
        features = hardware_features(hw, masks)
        self.assertEqual(masks[0].sum(), masks[1].sum())
        self.assertFalse(np.array_equal(features[0], features[1]))

    def test_no_hardware_mutation(self):
        hw = HardwareState(make_config('validation'))
        active, countdown = hw.active.copy(), hw.countdown.copy()
        hardware_features(hw, np.zeros((17, 16), bool))
        np.testing.assert_array_equal(active, hw.active)
        np.testing.assert_array_equal(countdown, hw.countdown)


if __name__ == '__main__':
    unittest.main()
