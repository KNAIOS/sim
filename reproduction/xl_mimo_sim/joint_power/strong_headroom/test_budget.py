import unittest

import numpy as np

from joint_power.strong_headroom.budget_analysis import choose, parse_episode


class BudgetHeadroomTests(unittest.TestCase):
    def test_episode_reader_preserves_reference_name(self):
        row = parse_episode(dict(scenario='1', trajectory='2', reference='Expanded PRS',
                                 energy_j='90.2', outage='0.025'))
        self.assertEqual(row['reference'], 'Expanded PRS')
        self.assertEqual(row['scenario'], 1)
        self.assertEqual(row['energy_j'], 90.2)

    def test_remaining_allowance_and_empty_fallback(self):
        y = np.zeros((3, 3, 6))
        y[:, :, 0] = [[5, 5, 5], [2, 2, 2], [1, 1, 1]]
        y[:, -1, 1] = [0, 2, 3]
        self.assertEqual(choose(y, 0, 19, 21, 2), (1, True))
        self.assertEqual(choose(y, 0, 21, 21, 2), (0, True))
        self.assertEqual(choose(y, 0, 22, 21, 2), (0, False))

    def test_feasible_repair_may_cost_more_energy(self):
        y = np.zeros((2, 3, 6))
        y[:, :, 0] = [[1, 1, 1], [3, 3, 3]]
        y[:, -1, 1] = [4, 0]
        self.assertEqual(choose(y, 0, 19, 21, 2), (1, True))


if __name__ == '__main__':
    unittest.main()
