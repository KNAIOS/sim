import unittest
from dataclasses import replace

import numpy as np

from future_vr.config import make_config
from future_vr.system import XLSystem
from major_revision.power_headroom import allocation, beam_gains, minimum_power, rates


class PowerHeadroomTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config('validation')

    def test_orthogonal_exact_solution(self):
        noise = XLSystem(self.cfg).noise_w
        g = np.diag(np.array([20., 30., 40.])*noise)
        target = np.array([20., 35., 45.])
        p = minimum_power(self.cfg, g, target)
        expected = np.expm1(np.log(2)*target/39.4)/np.array([20., 30., 40.])
        np.testing.assert_allclose(p, expected, rtol=1e-12)
        np.testing.assert_allclose(rates(self.cfg, g, p), target, atol=1e-12)

    def test_interference_closed_form(self):
        cfg = replace(self.cfg, users=2)
        noise = XLSystem(cfg).noise_w
        g = noise*np.array([[10., 1.], [2., 20.]])
        target = np.full(2, 39.4)  # gamma=1
        p = minimum_power(cfg, g, target)
        np.testing.assert_allclose(p, [(.1+.1*.05)/.99, (.05+.1*.1)/.99])
        np.testing.assert_allclose(rates(cfg, g, p), target, atol=1e-12)

    def test_original_phy_and_unused_csi(self):
        rng = np.random.default_rng(240922)
        channel = 1e-6*(rng.normal(size=(3, 128))+1j*rng.normal(size=(3, 128)))
        active = np.arange(16) < 5
        gain = beam_gains(self.cfg, channel, active)
        original = rates(self.cfg, gain, np.full(3, .4))
        np.testing.assert_allclose(original, XLSystem(self.cfg).achievable_rates(channel, active), atol=1e-9)
        altered = channel.copy()
        altered[:, np.repeat(~active, 8)] = np.nan
        np.testing.assert_array_equal(gain, beam_gains(self.cfg, altered, active))
        req = original*np.array([.7, 1.2, .9])
        result = allocation(self.cfg, gain, req)
        self.assertLess(result['power'].sum(), 1.2)
        np.testing.assert_array_equal(result['before'] < req, result['after'] < req)
        self.assertTrue(np.all(result['after'] >= np.minimum(original, req)-1e-8))

    def test_zero_channel_and_infeasibility(self):
        g = np.zeros((3, 3))
        result = allocation(self.cfg, g, np.ones(3)*300)
        np.testing.assert_array_equal(result['before'], result['after'])
        self.assertTrue(np.isfinite(result['power']).all())
        with self.assertRaises(ValueError):
            minimum_power(self.cfg, g, np.ones(3))
        noise = XLSystem(self.cfg).noise_w
        g = noise*np.ones((3, 3))
        with self.assertRaises(ValueError):
            minimum_power(self.cfg, g, np.ones(3)*100)


if __name__ == '__main__':
    unittest.main()
