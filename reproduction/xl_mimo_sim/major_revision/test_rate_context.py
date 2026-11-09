import unittest
import numpy as np
from future_vr.config import make_config
from future_vr.belief import VisibilityBelief
from future_vr.system import HardwareState
from major_revision.pilot import state
from major_revision.rate_context import candidate_rate_context


class RateContextTests(unittest.TestCase):
    def test_equal_power_states_hide_multiuser_correlation(self):
        cfg=make_config('smoke')
        gain=np.full((cfg.subarrays,cfg.users),-100.)
        a,b=VisibilityBelief(cfg,gain),VisibilityBelief(cfg,gain)
        correlated=np.ones((cfg.users,cfg.antennas),dtype=complex)*1e-5
        orthogonal=correlated*np.exp(2j*np.pi*np.arange(cfg.users)[:,None]*np.arange(cfg.antennas)[None,:]/cfg.antennas)
        np.testing.assert_allclose(abs(correlated),abs(orthogonal))
        a.channel_estimate=correlated
        b.channel_estimate=orthogonal
        hw=HardwareState(cfg)
        req=np.full(cfg.users,360.)
        sa,score=state(cfg,a,hw,req,req,np.zeros(cfg.subarrays),0.,True)
        sb,_=state(cfg,b,hw,req,req,np.zeros(cfg.subarrays),0.,True)
        np.testing.assert_array_equal(sa,sb)
        ra=candidate_rate_context(cfg,a,score,req)
        rb=candidate_rate_context(cfg,b,score,req)
        self.assertEqual(ra.size,(cfg.subarrays-cfg.users)*cfg.users)
        self.assertGreater(np.max(abs(ra-rb)),.5)

    def test_uses_only_supplied_channel_estimate(self):
        cfg=make_config('smoke')
        tr=VisibilityBelief(cfg,np.full((cfg.subarrays,cfg.users),-100.))
        tr.channel_estimate=np.ones((cfg.users,cfg.antennas),complex)*1e-5
        before=tr.channel_estimate.copy()
        r=candidate_rate_context(cfg,tr,np.ones(cfg.subarrays),np.full(cfg.users,360.))
        self.assertTrue(np.isfinite(r).all())
        np.testing.assert_array_equal(before,tr.channel_estimate)

if __name__=='__main__':
    unittest.main()
