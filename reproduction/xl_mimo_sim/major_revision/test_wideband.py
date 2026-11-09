import unittest
import numpy as np
from future_vr.config import make_config
from future_vr.system import XLSystem
from major_revision.wideband import WidebandSystem


class WidebandTests(unittest.TestCase):
    def test_batched_rates_equal_explicit_frequency_average(self):
        cfg=make_config('smoke')
        rng=np.random.default_rng(712)
        h=(rng.normal(size=(7,cfg.users,cfg.antennas))+1j*rng.normal(size=(7,cfg.users,cfg.antennas)))*1e-5
        mask=np.arange(cfg.subarrays)%2==0
        expected=np.mean([XLSystem(cfg).achievable_rates(x,mask) for x in h],axis=0)
        np.testing.assert_allclose(WidebandSystem(cfg).achievable_rates(h,mask),expected,rtol=1e-12)

    def test_flat_frequency_channel_matches_original(self):
        cfg=make_config('smoke')
        rng=np.random.default_rng(713)
        h=(rng.normal(size=(cfg.users,cfg.antennas))+1j*rng.normal(size=(cfg.users,cfg.antennas)))*1e-5
        mask=np.ones(cfg.subarrays,bool)
        np.testing.assert_allclose(WidebandSystem(cfg).achievable_rates(np.repeat(h[None],33,axis=0),mask),
                                   XLSystem(cfg).achievable_rates(h,mask),rtol=1e-12)

if __name__=='__main__':
    unittest.main()
