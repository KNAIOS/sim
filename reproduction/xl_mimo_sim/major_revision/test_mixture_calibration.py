import unittest
from major_revision.mixture_calibration import mixture_weight


class MixtureCalibrationTests(unittest.TestCase):
    def test_boundary_when_energy_and_outage_trade_off(self):
        p,ok=mixture_weight(100.,.02,80.,.08)
        self.assertTrue(ok)
        self.assertAlmostEqual(p,.5)
    def test_both_feasible_select_lower_energy(self):
        self.assertEqual(mixture_weight(100.,.01,80.,.04),(1.,True))
    def test_costlier_reliable_component_uses_minimum_needed(self):
        p,ok=mixture_weight(80.,.08,100.,.02)
        self.assertTrue(ok)
        self.assertAlmostEqual(p,.5)
    def test_no_feasible_component_is_not_certified(self):
        self.assertEqual(mixture_weight(80.,.08,100.,.06),(1.,False))
    def test_equal_outage_chooses_lower_energy_and_validates_inputs(self):
        self.assertEqual(mixture_weight(80.,.02,100.,.02),(0.,True))
        with self.assertRaises(ValueError):
            mixture_weight(80.,-1.,100.,.02)


if __name__=='__main__':
    unittest.main()
