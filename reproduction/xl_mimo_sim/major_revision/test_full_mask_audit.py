import unittest
from dataclasses import replace
import numpy as np

from future_vr.config import make_config
from future_vr.system import HardwareState
from major_revision.action_value import hold_predictions
from major_revision.full_domain import FullDomainRollout
from major_revision.full_mask_audit import (mask_id,masks_from_ids,causal_forecast,
    predicted_batch,enumerate_costs,choose_masks,scalar_hold)
from major_revision.pilot import trajectory


class FullMaskAuditTests(unittest.TestCase):
    def test_complete_mask_roundtrip_and_empty_excluded(self):
        ids=np.arange(1,65536,dtype=np.uint32)
        masks=masks_from_ids(ids,16)
        self.assertEqual(len({mask_id(m) for m in masks}),65535)
        self.assertTrue(np.all(masks.sum(1)>0))
        np.testing.assert_array_equal(masks[-1],np.ones(16,bool))

    def test_arbitrary_masks_match_scalar_hardware_and_rzf(self):
        cfg=replace(make_config('validation'),episode_steps=24)
        run=FullDomainRollout(cfg,trajectory(cfg,1130900001,(2.,1.)))
        run.base_observation()
        run.hw.active[3:]=False
        run.hw.countdown[4:7]=[1,2,4]
        grams,channels=causal_forecast(cfg,run.tracker)
        masks=masks_from_ids([1,3,127,32768,65535],16)
        before=run.hw.active.copy()
        batch=predicted_batch(cfg,run.hw,grams,run.req*1.08,masks)
        for i,mask in enumerate(masks):
            scalar=scalar_hold(cfg,run.hw,channels,run.req*1.08,mask)
            np.testing.assert_allclose(batch[i],scalar,rtol=1e-10,atol=1e-9)
        np.testing.assert_array_equal(run.hw.active,before)

    def test_prefix_and_small_complete_enumeration(self):
        cfg=replace(make_config('validation'),subarrays=5,episode_steps=20)
        run=FullDomainRollout(cfg,trajectory(cfg,1130900002,(.5,.85)))
        _,score=run.base_observation()
        grams,_=causal_forecast(cfg,run.tracker)
        ids,costs=enumerate_costs(cfg,run.hw,grams,run.req,batch=7)
        old=hold_predictions(cfg,run.tracker,run.hw,score)
        indexes=np.array([mask_id(m) for m in old['masks']])-1
        np.testing.assert_allclose(costs[indexes,0],old['energy'].sum(1),atol=1e-10)
        np.testing.assert_array_equal(costs[indexes,1],(old['rates']<run.req).mean((1,2)))
        chosen=choose_masks(cfg,ids,costs,old['masks'][2])
        for identifier in chosen[1:3]:
            self.assertEqual(masks_from_ids([identifier],5).sum(),3)
        self.assertLessEqual(np.sum(masks_from_ids([chosen[1]],5)[0]!=old['masks'][2]),2)

    def test_ties_prefer_reference_and_all_domain_can_change_count(self):
        cfg=replace(make_config('validation'),subarrays=4)
        ids=np.arange(1,16,dtype=np.uint32)
        ref=masks_from_ids([3],4)[0]
        costs=np.tile([10.,0.,0.],(15,1))
        np.testing.assert_array_equal(choose_masks(cfg,ids,costs,ref),[3,3,3,3])
        costs[0,0]=9.
        self.assertEqual(choose_masks(cfg,ids,costs,ref)[-1],1)
        costs[0,1]=1.
        self.assertEqual(choose_masks(cfg,ids,costs,ref)[-1],3)


if __name__=='__main__':
    unittest.main()
