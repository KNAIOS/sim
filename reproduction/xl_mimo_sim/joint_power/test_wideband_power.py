import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from future_vr.system import XLSystem
from joint_power.model import required_powers
from joint_power.wideband_power import allocate, rates, required


class WidebandPowerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(make_config('validation'), users=2)
        self.noise = XLSystem(self.cfg).noise_w

    def test_one_frequency_matches_analytic_coupled_power(self):
        gain = self.noise*np.array([[10., 1.], [2., 20.]])
        target = np.full(2, 39.4)
        exact, ok = required_powers(self.cfg, gain, target)
        power, status, _ = required(self.cfg, gain[None], target)
        self.assertTrue(ok)
        self.assertEqual(status, 'feasible')
        np.testing.assert_allclose(power, exact, atol=1e-8)

    def test_two_frequency_orthogonal_closed_form(self):
        direct = np.array([[10., 20.], [40., 50.]])
        gain = self.noise*np.array([np.diag(g) for g in direct])
        target = np.array([45., 55.])
        a, b = direct
        z = np.exp2(2*target/39.4)-1
        exact = 2*z/(a+b+np.sqrt((a+b)**2+4*a*b*z))
        power, status, _ = required(self.cfg, gain, target)
        self.assertEqual(status, 'feasible')
        np.testing.assert_allclose(power, exact, atol=1e-8)
        np.testing.assert_allclose(rates(self.cfg, gain, power), target, atol=1e-6)

    def test_identical_bins_equivalent_to_one_bin(self):
        gain = self.noise*np.array([[30., 2.], [1., 40.]])
        one = allocate(self.cfg, gain[None], [50., 60.])
        many = allocate(self.cfg, np.repeat(gain[None], 99, axis=0), [50., 60.])
        np.testing.assert_allclose(one[0], many[0], atol=1e-8)
        np.testing.assert_allclose(one[1], many[1], atol=1e-6)

    def test_infeasible_case_preserves_existing_service(self):
        gain = self.noise*np.array([[[2., 1.], [.5, 3.]], [[5., .3], [.6, 2.]]])
        equal = np.full(2, .6)
        before = rates(self.cfg, gain, equal)
        target = before*np.array([.8, 1.5])
        power, after, metadata = allocate(self.cfg, gain, target)
        self.assertLessEqual(power.sum(), 1.2+1e-10)
        self.assertTrue(np.all(after >= np.minimum(before, target)-1e-7))
        self.assertNotEqual(metadata['mode'], 'full_targets')

    def test_iteration_limit_is_not_reported_as_infeasibility(self):
        gain = self.noise*np.array([[[10., 1.], [2., 20.]]])
        _, status, _ = required(self.cfg, gain, np.full(2, 39.4), iterations=1)
        self.assertEqual(status, 'iteration_limit')

    def test_zero_channel_and_zero_demand(self):
        power, achieved, _ = allocate(self.cfg, np.zeros((99, 2, 2)), [40., 50.])
        np.testing.assert_array_equal(power, 0.)
        np.testing.assert_array_equal(achieved, 0.)
        power, status, _ = required(self.cfg, np.zeros((1, 2, 2)), np.zeros(2))
        self.assertEqual(status, 'feasible')
        np.testing.assert_array_equal(power, 0.)


if __name__ == '__main__':
    unittest.main()
