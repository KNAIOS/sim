import copy
import unittest
import numpy as np
from future_vr.config import make_config
from future_vr.controllers import select_top
from future_vr.system import XLSystem
from major_revision.full_domain import FullDomainRollout
from major_revision.pilot import trajectory
from major_revision.spatial_actions import causal_forecasts
from major_revision.gram_control import (subarray_grams,rates_from_gram,
    prefix_forecast_rates,gram_belief_features,representation_size)


def forecast_alias():
    """Constructed algebraic states, not samples claimed from the channel model."""
    cfg=make_config('validation')
    run=FullDomainRollout(cfg,trajectory(cfg,995900000,(2.,1.)))
    run.prepare()
    run.req[:]=360.
    run.previous_req=run.req.copy()
    run.age[:]=0.
    power=7e-11
    run.tracker.gain_db[:]=10*np.log10(power)
    # A filtered gain belief need not equal the latest complex-CSI power.
    run.tracker.gain_db[14]+=6.
    run.tracker.gain_db[13]+=3.
    run.tracker.velocity[:]=0.
    run.tracker.probability=run.tracker._gain_to_probability(run.tracker.gain_db)
    run.tracker.uncertainty[:]=.08
    run.tracker._forecast_cache.clear()
    _,score=run.base_observation()
    order=np.argsort(score,kind='stable')[::-1]
    runs=[]
    element=np.arange(cfg.antennas_per_subarray)
    for variant in (0,1):
        h=np.empty((cfg.users,cfg.subarrays,cfg.antennas_per_subarray),complex)
        h[0]=1.
        h[1]=np.exp(2j*np.pi*element/cfg.antennas_per_subarray)
        h[2]=np.exp(4j*np.pi*element/cfg.antennas_per_subarray)
        h[1,order[0]]=1.
        h[1,order[1]]=1j
        h[1,order[2]]=1. if variant==0 else 1j
        other=copy.deepcopy(run)
        other.tracker.channel_estimate=h.reshape(cfg.users,-1)*np.sqrt(power)
        runs.append(other)
    return runs


class GramControlTests(unittest.TestCase):
    def setUp(self):
        self.cfg=make_config('validation')

    def test_random_mask_rzf_exactness_including_empty(self):
        rng=np.random.default_rng(995900001)
        for correlated in (False,True):
            h=(rng.normal(size=(3,128))+1j*rng.normal(size=(3,128)))*1e-5
            if correlated:
                h[1]=h[0]+1e-4*h[1]
            grams=subarray_grams(self.cfg,h)
            for n in range(17):
                mask=np.zeros(16,bool)
                mask[rng.permutation(16)[:n]]=True
                expected=XLSystem(self.cfg).achievable_rates(h,mask)
                actual=rates_from_gram(self.cfg,grams[mask].sum(axis=0),n*8)
                np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-5)

    def test_forecast_and_readiness_masks_match_original(self):
        run=FullDomainRollout(self.cfg,trajectory(self.cfg,995900002,(2.,1.)))
        run.prepare()
        run.hw.active[8:]=False
        run.hw.command[8:]=False
        run.hw.countdown[8:12]=[1,2,3,4]
        run.hw.command[8:12]=True
        _,score=run.base_observation()
        before=copy.deepcopy(run.hw)
        forecast=causal_forecasts(self.cfg,run.tracker)
        actual=prefix_forecast_rates(self.cfg,run.tracker,run.hw,score)
        for count in range(1,17):
            mask=select_top(score,count)
            expected=[run.system.achievable_rates(forecast[h-1],mask&(run.hw.readiness()<=h))
                      for h in range(1,9)]
            np.testing.assert_allclose(actual[count-1],expected,rtol=1e-7,atol=1e-5)
        for attr in ('active','command','countdown'):
            np.testing.assert_array_equal(getattr(run.hw,attr),getattr(before,attr))

    def test_same_235_features_can_hide_different_forecast_rates(self):
        a,b=forecast_alias()
        old_a,score_a=a.observation()
        old_b,score_b=b.observation()
        np.testing.assert_array_equal(old_a,old_b)
        np.testing.assert_array_equal(score_a,score_b)
        rates=[prefix_forecast_rates(self.cfg,r.tracker,r.hw,score_a) for r in (a,b)]
        self.assertGreater(np.max(np.abs(rates[0]-rates[1])),1.)
        features=[gram_belief_features(self.cfg,r.tracker,r.hw,r.req,r.previous_req,r.age,r.last_outage)
                  for r in (a,b)]
        self.assertEqual(features[0].shape,(362,))
        self.assertEqual(representation_size(self.cfg),362)
        self.assertGreater(np.max(np.abs(features[0]-features[1])),.5)


if __name__=='__main__':
    unittest.main()
