import unittest
import numpy as np
from major_revision.penalty_sweep import objective


class PriceTests(unittest.TestCase):
    def test_objective_rewards_energy_and_penalizes_outage(self):
        metrics=[dict(energy_j=100.,outage=.05),dict(energy_j=200.,outage=.1)]
        self.assertAlmostEqual(objective(metrics,4.,300.),.8)
        self.assertLess(objective(metrics,.25,300.),objective(metrics,4.,300.))

    def test_streams_are_disjoint(self):
        train={951000000+s*10000+i for s in (11,22) for i in range(100)}
        val={952000000+j*10000+i for j in range(3) for i in range(4)}
        dev={953000000+j*10000+i for j in range(3) for i in range(4)}
        self.assertFalse(train&val or train&dev or val&dev)

if __name__=='__main__':
    unittest.main()
