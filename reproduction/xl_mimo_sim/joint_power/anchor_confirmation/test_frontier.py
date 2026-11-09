import unittest

from joint_power.anchor_confirmation.frontier import envelope


class FrontierTests(unittest.TestCase):
    def test_known_two_policy_expectation(self):
        # Synthetic arithmetic fixture, never a manuscript result.
        e, c, i, j, p = envelope([100., 60.], [0., .1], .025)
        self.assertAlmostEqual(e, 90.)
        self.assertAlmostEqual(c, .025)
        self.assertEqual((i, j), (0, 1))
        self.assertAlmostEqual(p, .75)

    def test_pure_dominates_and_infeasible_is_explicit(self):
        self.assertEqual(envelope([80., 90., 100.], [.02, .03, 0.], .04)[0], 80.)
        with self.assertRaises(ValueError):
            envelope([10., 20.], [.1, .2], .01)


if __name__ == '__main__':
    unittest.main()
