import copy
import unittest
from dataclasses import replace

import numpy as np

from future_vr.channel import generate_trajectory
from future_vr.config import make_config
from future_vr.system import XLSystem
from joint_power.model import Environment, allocate, effective_gains, rates, required_powers, search_action


class JointModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg=make_config('validation')
        self.tr=generate_trajectory(self.cfg,2026092201,2.)

    def test_equal_power_matches_original_phy(self):
        for t in (0,13,71):
            for count in (1,3,8,16):
                mask=np.arange(16)<count
                h=self.tr.channel[t][:,np.repeat(mask,8)]
                gain=effective_gains(self.cfg,h@h.conj().T)
                np.testing.assert_allclose(rates(self.cfg,gain,np.full(3,.4)),
                    XLSystem(self.cfg).achievable_rates(self.tr.channel[t],mask),atol=1e-7)

    def test_exact_orthogonal_minimum(self):
        noise=XLSystem(self.cfg).noise_w
        gain=np.diag(np.array([20.,30.,40.])*noise)
        target=np.array([20.,35.,45.])
        p,ok=required_powers(self.cfg,gain,target)
        self.assertTrue(ok)
        np.testing.assert_allclose(p,np.expm1(np.log(2)*target/39.4)/[20,30,40],rtol=1e-12)
        np.testing.assert_allclose(rates(self.cfg,gain,p),target,atol=1e-10)

    def test_interference_and_total_not_per_user_cap(self):
        cfg=replace(self.cfg,users=2)
        noise=XLSystem(cfg).noise_w
        gain=noise*np.array([[10.,1.],[2.,20.]])
        p,ok=required_powers(cfg,gain,np.full(2,39.4))
        self.assertTrue(ok)
        np.testing.assert_allclose(p,[(.1+.1*.05)/.99,(.05+.1*.1)/.99])
        target=np.array([39.4*np.log2(1+8),39.4*np.log2(1+2)])
        p,achieved,fallback=allocate(cfg,noise*np.diag([10.,20.]),target)
        self.assertGreater(p[0],.6)  # Full target is feasible after redistribution.
        self.assertLess(p.sum(),1.2)
        self.assertTrue(np.all(achieved>=target))
        self.assertFalse(fallback)

    def test_batch_scalar_and_cap(self):
        h=self.tr.channel[:9]
        g=effective_gains(self.cfg,h@h.conj().swapaxes(-2,-1))
        demand=self.tr.requirements_mbps[:9]*1.2
        bp,br,bf=allocate(self.cfg,g,demand)
        for i in range(9):
            p,r,f=allocate(self.cfg,g[i],demand[i])
            np.testing.assert_allclose(bp[i],p,atol=1e-10)
            np.testing.assert_allclose(br[i],r,atol=1e-8)
            self.assertEqual(bf[i],f)
        self.assertTrue(np.all(bp>=0))
        self.assertTrue(np.all(bp.sum(-1)<=1.2+1e-10))

    def test_infeasible_fallback_preserves_service(self):
        h=self.tr.channel[0][:,:8]
        gain=effective_gains(self.cfg,h@h.conj().T)
        before=rates(self.cfg,gain,np.full(3,.4))
        demand=before*np.array([.8,1.2,1.4])
        p,after,_=allocate(self.cfg,gain,demand)
        self.assertTrue(np.all(after>=np.minimum(before,demand)-1e-7))
        self.assertLessEqual(p.sum(),1.2+1e-10)
        p,r,f=allocate(self.cfg,np.zeros((3,3)),np.full(3,460.))
        np.testing.assert_array_equal(p,np.zeros(3))
        np.testing.assert_array_equal(r,np.zeros(3))
        self.assertFalse(f)

    def test_constant_channel_forecast_matches_physical_replay(self):
        tr=replace(self.tr,channel=np.repeat(self.tr.channel[:1],140,axis=0),
            gain_db=np.repeat(self.tr.gain_db[:1],140,axis=0),
            requirements_mbps=np.repeat(self.tr.requirements_mbps[:1],140,axis=0))
        env=Environment(self.cfg,tr)
        env.hw.active[8:]=False
        env.hw.command[10:]=False
        env.hw.countdown[8:10]=2
        candidates=env.candidates()
        for i in range(17):
            branch=copy.deepcopy(env)
            for _ in range(8):
                branch.step(candidates.masks[i])
            self.assertAlmostEqual(candidates.energy[i],branch.metrics()['energy_j']/8,places=9)
            self.assertAlmostEqual(candidates.outage[i],branch.metrics()['outage'],places=12)

    def test_sleeping_and_future_csi_hidden_from_current_decision(self):
        env=Environment(self.cfg,self.tr)
        env.block(np.arange(16)<4)
        altered=copy.deepcopy(env)
        channel=self.tr.channel.copy()
        hidden=np.repeat(~env.hw.active,8)
        channel[env.t,:,hidden] *= 1000
        channel[env.t+1:] *= 1000
        gain=self.tr.gain_db.copy()
        gain[env.t,~env.hw.active] += 60
        gain[env.t+1:] += 60
        altered.tr=replace(self.tr,channel=channel,gain_db=gain)
        first,second=env.candidates(),altered.candidates()
        np.testing.assert_array_equal(first.masks,second.masks)
        np.testing.assert_allclose(first.features,second.features,atol=1e-7)
        np.testing.assert_allclose(first.energy,second.energy,atol=1e-10)

    def test_fixed_schedule_energy_matches_original(self):
        env=Environment(self.cfg,self.tr,variable_power=False)
        for i in range(16):
            mask=np.arange(16)<(4 if i<8 else 7)
            env.prepare()
            hardware=copy.deepcopy(env.hw)
            active,switch,_=hardware.apply(mask,1.)
            expected=XLSystem(self.cfg).evaluate(self.tr.channel[i],active,hardware.countdown>0,
                switch,False,env.req,observed_count=int(env.observed.sum()))
            env.step(mask)
            self.assertAlmostEqual(env.records[-1]['energy_j'],expected.energy_j,places=12)
            self.assertAlmostEqual(env.records[-1]['outage'],expected.outage,places=12)

    def test_search_full_set_and_power_changes_objective(self):
        env=Environment(self.cfg,self.tr)
        c=env.candidates()
        self.assertEqual(c.masks.shape,(17,16))
        self.assertEqual(c.masks[0].sum(),1)
        self.assertEqual(c.masks[15].sum(),16)
        action=search_action(c)
        feasible=np.flatnonzero(c.outage<=.05)
        if len(feasible):
            self.assertEqual(c.energy[action],c.energy[feasible].min())
        self.assertGreater(c.features.shape[1],72)
        self.assertTrue(np.isfinite(c.features).all())


if __name__=='__main__':
    unittest.main()
