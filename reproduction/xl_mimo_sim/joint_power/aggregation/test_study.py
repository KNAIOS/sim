import unittest

import numpy as np

from joint_power.aggregation.control import select_options
from joint_power.aggregation.train import tensors


class StudyTests(unittest.TestCase):
    def test_selection_prioritizes_feasibility_over_low_energy(self):
        rows = []
        for method in ('PRS-added', 'Visitation-added'):
            for seed in (11, 22):
                for price, energy, outage in ((1., 80., .06), (2., 100., .04), (4., 101., .01)):
                    for scenario in range(3):
                        rows.append(dict(method=method, seed=seed, parameter=price,
                            scenario=scenario, trajectory=0, energy_j=energy, outage=outage))
        selected, chosen = select_options(rows)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(chosen), 12)
        self.assertTrue(all(r['parameter'] == 2. for r in selected))

    def test_remaining_time_and_full_tail_are_the_training_targets(self):
        y = np.arange(2*17*3*2).reshape(2, 17, 3, 2)
        data = dict(x=np.zeros((2, 17, 343), np.float32), reference=np.array([3, 4]),
                    identity=np.array([[0, 0, 16], [2, 3, 112]]), y=y)
        x, reference, remaining, outcome = tensors(data)
        np.testing.assert_array_equal(remaining.numpy(), [124., 28.])
        np.testing.assert_array_equal(outcome.numpy(), y[:, :, -1])
        np.testing.assert_array_equal(reference.numpy(), data['reference'])
        self.assertEqual(tuple(x.shape), (2, 17, 343))


if __name__ == '__main__':
    unittest.main()
