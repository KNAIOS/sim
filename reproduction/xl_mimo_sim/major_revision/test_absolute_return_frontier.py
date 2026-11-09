import unittest
from major_revision.absolute_return_frontier import mixture_frontier


class MixtureFrontierTests(unittest.TestCase):
    def test_known_two_point_solution_and_pure_or_infeasible_cases(self):
        points=[dict(energy_j=100.,outage=.1),dict(energy_j=200.,outage=0.)]
        answer=mixture_frontier(points,.04)
        self.assertAlmostEqual(answer['energy_j'],160.)
        self.assertAlmostEqual(answer['outage'],.04)
        self.assertAlmostEqual(answer['weight_a'],.6)
        self.assertEqual(mixture_frontier(points,.1)['energy_j'],100.)
        self.assertIsNone(mixture_frontier(points,-.01))

    def test_dominated_points_do_not_improve_the_frontier(self):
        points=[dict(energy_j=100.,outage=.1),dict(energy_j=200.,outage=0.),dict(energy_j=190.,outage=.05)]
        self.assertAlmostEqual(mixture_frontier(points,.05)['energy_j'],150.)


if __name__=='__main__':
    unittest.main()
