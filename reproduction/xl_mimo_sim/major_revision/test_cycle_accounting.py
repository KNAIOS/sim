import unittest
from dataclasses import replace
import numpy as np

from future_vr.config import make_config
from major_revision.cycle_accounting import (
    replay, completed_cycles, bridge, scalar_bridge, cycle_difference, break_even_slots,
)
from major_revision.pilot import trajectory


class CycleAccountingTests(unittest.TestCase):
    def test_formula_independent_hardware_and_zero_delay_pilot_boundary(self):
        for delay in (0, 1, 3, 4, 6):
            cfg = replace(make_config('validation'), episode_steps=32, wake_delay_slots=delay)
            tr = trajectory(cfg, 1230990001, (2., 1.))
            for duration in (1, 4, 8):
                command = np.ones((32, cfg.subarrays), bool)
                command[4:4+duration, 0] = False
                trace = replay(cfg, tr, command, block=1)
                cycles, censored = completed_cycles(cfg, trace)
                self.assertEqual(len(cycles), 1)
                self.assertEqual(len(censored), 0)
                a, b = bridge(cfg, tr, trace, cycles[0]), scalar_bridge(cfg, tr, trace, cycles[0])
                self.assertAlmostEqual(a['energy_difference_j'], b['energy_difference_j'], places=10)
                self.assertAlmostEqual(a['energy_difference_j'], cycle_difference(cfg, duration), places=10)
                self.assertEqual(a['failed_user_slot_difference'], b['failed_user_slot_difference'])

    def test_terminal_censoring_and_no_fabricated_recovery(self):
        cfg = replace(make_config('validation'), episode_steps=20)
        tr = trajectory(cfg, 1230990002, (2., 1.))
        commands = np.ones((20, cfg.subarrays), bool)
        commands[4:, 0] = False
        commands[8:18, 1] = False  # Return lies beyond this episode.
        commands[8:16, 2] = False  # Ready exactly at its end: legitimate complete cycle.
        trace = replay(cfg, tr, commands, block=1)
        cycles, censored = completed_cycles(cfg, trace)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0]['ready'], 20)
        self.assertEqual({r['reason'] for r in censored}, {'no_wake_before_end', 'recovery_beyond_episode'})
        self.assertAlmostEqual(scalar_bridge(cfg, tr, trace, cycles[0])['energy_difference_j'], cycle_difference(cfg, 8), places=10)

    def test_aborted_wake_is_not_misclassified_as_complete(self):
        cfg = replace(make_config('validation'), episode_steps=20, wake_delay_slots=6)
        tr = trajectory(cfg, 1230990003, (2., 1.))
        commands = np.ones((20, cfg.subarrays), bool)
        commands[0:4, 0] = False
        commands[8:12, 0] = False
        trace = replay(cfg, tr, commands, block=1)
        with self.assertRaises(ValueError):
            completed_cycles(cfg, trace)

    def test_affine_break_even_and_default_minimum_cycle(self):
        cfg = make_config('validation')
        slope = cycle_difference(cfg, 5)-cycle_difference(cfg, 4)
        intercept = cycle_difference(cfg, 4)-4*slope
        self.assertAlmostEqual(break_even_slots(cfg), -intercept/slope, places=12)
        self.assertAlmostEqual(cycle_difference(cfg, 4), .4546144, places=10)
        self.assertGreater(slope, 0)
        self.assertLess(break_even_slots(cfg), 0)


if __name__ == '__main__':
    unittest.main()
