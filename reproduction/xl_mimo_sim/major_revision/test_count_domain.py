import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.system import XLSystem
from major_revision.pilot import trajectory
from major_revision.search_headroom import SearchAuditRollout
from major_revision.count_domain_audit import MinimumSearchRollout


class CountDomainTests(unittest.TestCase):
    def test_one_eight_antenna_module_can_have_three_streams(self):
        cfg=make_config('validation')
        h=np.zeros((cfg.users,cfg.antennas),dtype=complex)
        h[:,:cfg.antennas_per_subarray]=1e-4*np.exp(2j*np.pi*
            np.arange(cfg.users)[:,None]*np.arange(cfg.antennas_per_subarray)[None,:]/cfg.antennas_per_subarray)
        available=np.arange(cfg.subarrays)==0
        self.assertEqual(np.linalg.matrix_rank(h[:,:cfg.antennas_per_subarray]),cfg.users)
        rates=XLSystem(cfg).achievable_rates(h,available)
        self.assertTrue(np.all(rates>560.))

    def test_four_module_adapter_reproduces_original_search(self):
        for delay in (0,2,4,6):
            cfg=replace(make_config('validation'),episode_steps=20,wake_delay_slots=delay)
            tr=trajectory(cfg,979950000+delay,(3.,1.15))
            self.assertEqual(MinimumSearchRollout(cfg,tr,4).finish_search(1.04),
                             SearchAuditRollout(cfg,tr).finish_search(1.04))


if __name__=='__main__':
    unittest.main()
