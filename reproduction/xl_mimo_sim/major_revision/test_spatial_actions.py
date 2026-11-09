import itertools
import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.belief import VisibilityBelief
from future_vr.system import HardwareState
from major_revision.spatial_actions import retained_mask,exchange_mask,causal_forecasts,predicted_metrics
from major_revision.spatial_audit import prs_reference


class SpatialActionsTests(unittest.TestCase):
    def test_reference_lookup_filters_method_before_indexing(self):
        prs=dict(scenario=0,trajectory=0,method='PRS',energy_j=138.)
        full=dict(scenario=0,trajectory=0,method='Full array',energy_j=305.)
        self.assertEqual(prs_reference([prs,full]),{(0,0):prs})
        self.assertEqual(prs_reference([full,prs]),{(0,0):prs})
        with self.assertRaises(ValueError):
            prs_reference([prs,prs])
        with self.assertRaises(ValueError):
            prs_reference([full])

    def test_retention_has_minimum_hamming_distance(self):
        previous=np.array([True,False,True,False,False,True])
        score=np.array([1.,3.,2.,6.,4.,5.])
        for count in range(7):
            result=retained_mask(previous,score,count)
            alternatives=[np.array(x,dtype=bool) for x in itertools.product((0,1),repeat=6) if sum(x)==count]
            self.assertEqual(result.sum(),count)
            self.assertEqual(np.count_nonzero(result!=previous),
                             min(np.count_nonzero(x!=previous) for x in alternatives))
        self.assertTrue(np.array_equal(previous,[True,False,True,False,False,True]))

    def test_exchange_can_repair_a_rank_deficient_selected_channel(self):
        cfg=make_config('validation')
        channel=np.full((cfg.users,cfg.subarrays*cfg.antennas_per_subarray),1e-4,dtype=complex)
        block=np.exp(2j*np.pi*np.arange(cfg.users)[:,None]*np.arange(cfg.antennas_per_subarray)[None,:]/cfg.antennas_per_subarray)*1e-4
        channel[:,4*cfg.antennas_per_subarray:5*cfg.antennas_per_subarray]=block
        gain=10*np.log10(np.mean(np.abs(channel.reshape(cfg.users,cfg.subarrays,cfg.antennas_per_subarray))**2,axis=2).T)
        tracker=VisibilityBelief(cfg,gain)
        tracker.channel_estimate=channel.copy()
        hardware=HardwareState(cfg)
        anchor=np.arange(cfg.subarrays)<4
        original_active=hardware.active.copy()
        result,old,new=exchange_mask(cfg,tracker,hardware,np.full(cfg.users,360.),anchor)
        self.assertEqual(result.sum(),anchor.sum())
        self.assertEqual(np.count_nonzero(result!=anchor),2)
        self.assertTrue(result[4])
        self.assertLess(new['shortfall'],old['shortfall'])
        self.assertTrue(np.array_equal(hardware.active,original_active))
        self.assertTrue(np.array_equal(tracker.channel_estimate,channel))

    def test_full_mask_is_a_no_op_and_transition_cost_matches_hardware(self):
        cfg=make_config('validation')
        hardware=HardwareState(cfg)
        tracker=VisibilityBelief(cfg,np.full((cfg.subarrays,cfg.users),-80.))
        tracker.channel_estimate=np.full((cfg.users,cfg.subarrays*cfg.antennas_per_subarray),1e-4,dtype=complex)
        anchor=np.ones(cfg.subarrays,dtype=bool)
        result,old,new=exchange_mask(cfg,tracker,hardware,np.full(cfg.users,360.),anchor)
        self.assertTrue(np.array_equal(result,anchor))
        self.assertEqual(old,new)
        command=np.arange(cfg.subarrays)<4
        prediction=predicted_metrics(cfg,hardware,np.full(cfg.users,360.),causal_forecasts(cfg,tracker),command)
        _,energy,_=hardware.apply(command,1.)
        self.assertEqual(energy,prediction['transition_j'])

    def test_sleeping_chain_forecast_respects_recovery_mask(self):
        cfg=replace(make_config('validation'),wake_delay_slots=20)
        hardware=HardwareState(cfg)
        hardware.active[:]=False
        hardware.command[:]=False
        forecasts=np.ones((cfg.horizon,cfg.users,cfg.subarrays*cfg.antennas_per_subarray),dtype=complex)
        result=predicted_metrics(cfg,hardware,np.ones(cfg.users),forecasts,np.ones(cfg.subarrays,dtype=bool))
        self.assertEqual(result['outage'],1.)
        self.assertEqual(result['shortfall'],1.)


if __name__=='__main__':
    unittest.main()
