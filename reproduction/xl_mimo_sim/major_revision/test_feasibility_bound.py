import unittest
import numpy as np
from future_vr.config import make_config
from future_vr.system import XLSystem
from major_revision.full_domain_feasibility import rate_upper_bound


class FeasibilityBoundTests(unittest.TestCase):
    def test_bounds_subset_rates(self):
        cfg=make_config('validation')
        system=XLSystem(cfg)
        rng=np.random.default_rng(19)
        for _ in range(20):
            channel=(rng.normal(size=(cfg.users,cfg.antennas))+
                     1j*rng.normal(size=(cfg.users,cfg.antennas)))*1e-4
            upper=rate_upper_bound(cfg,channel,cfg.tx_power_w/cfg.users)
            for count in (1,4,16):
                mask=np.arange(cfg.subarrays)<count
                self.assertTrue(np.all(system.achievable_rates(channel,mask)<=upper+1e-6))

    def test_orthogonal_full_channel_attains_equal_power_bound(self):
        cfg=make_config('validation')
        channel=np.eye(cfg.users,cfg.antennas,dtype=complex)*1e-4
        actual=XLSystem(cfg).achievable_rates(channel,np.ones(cfg.subarrays,bool))
        upper=rate_upper_bound(cfg,channel,cfg.tx_power_w/cfg.users)
        np.testing.assert_allclose(actual,upper,atol=1e-9)


if __name__=='__main__':
    unittest.main()
