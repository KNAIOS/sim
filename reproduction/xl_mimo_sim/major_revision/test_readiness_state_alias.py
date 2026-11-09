"""Constructive audit: ready-all prefix rates do not identify ready-mask rates.

This is a state-representation counterexample, not a claim about its frequency in
the simulation data or the performance of an untested replacement controller.
"""
import unittest
import numpy as np
from future_vr.config import make_config
from future_vr.belief import VisibilityBelief
from future_vr.controllers import select_top
from future_vr.system import HardwareState, XLSystem
from major_revision.pilot import state
from major_revision.rate_context import candidate_rate_context
from major_revision.ready_rate_context import ready_rate_context


def counterexample():
    cfg = make_config('validation')
    element = np.arange(cfg.antennas_per_subarray)
    branch_power = 7e-11
    gain = np.full((cfg.subarrays, cfg.users), 10*np.log10(branch_power))
    # The sleeping module's advected gain belief differs from its last measured
    # complex CSI, which the active-only tracker retains until it is observed.
    gain[14] += 6.
    trackers = [VisibilityBelief(cfg, gain) for _ in range(2)]
    # All modules have the same per-user power. The three highest-ranked modules
    # have inter-user correlations 1, i, 1 versus 1, i, i. Every permitted full
    # prefix includes all three: |2+i| = |1+2i|. Other modules have zero correlation.
    for variant, tracker in enumerate(trackers):
        h = np.empty((cfg.users, cfg.subarrays, cfg.antennas_per_subarray), complex)
        h[0] = 1.
        h[1] = np.exp(2j*np.pi*element/cfg.antennas_per_subarray)
        h[2] = np.exp(4j*np.pi*element/cfg.antennas_per_subarray)
        h[1,15] = 1.
        h[1,14] = 1j
        h[1,13] = 1. if variant == 0 else 1j
        tracker.channel_estimate = h.reshape(cfg.users,-1)*np.sqrt(branch_power)
    hw = HardwareState(cfg)
    hw.active[:] = False
    hw.active[[11,12,13,15]] = True
    hw.command = hw.active.copy()
    # Module 14 is asleep, with the nominal four-slot recovery if commanded.
    # Its high predicted gain places it in the top four despite delayed readiness.
    provisional_req = np.full(cfg.users, 360.)
    _, score = state(cfg, trackers[0], hw, provisional_req, provisional_req,
                     np.zeros(cfg.subarrays), 0., True)
    command = select_top(score,4)
    np.testing.assert_array_equal(np.flatnonzero(command),[12,13,14,15])
    active_after_command = command & hw.active
    system = XLSystem(cfg)
    rates = [system.achievable_rates(t.channel_estimate,active_after_command) for t in trackers]
    req = provisional_req
    base_states = []
    contexts = []
    for tracker in trackers:
        base,actual_score = state(cfg,tracker,hw,req,req,np.zeros(cfg.subarrays),0.,True)
        np.testing.assert_allclose(actual_score,score)
        base_states.append(base)
        contexts.append(candidate_rate_context(cfg,tracker,actual_score,req))
    ready_contexts = [ready_rate_context(cfg,t,hw,score,req) for t in trackers]
    return base_states,contexts,rates,req,ready_contexts,hw


class ReadinessStateAliasTests(unittest.TestCase):
    def test_identical_full_prefix_rates_can_hide_current_outage(self):
        bases,contexts,rates,req,ready,_ = counterexample()
        np.testing.assert_allclose(bases[0],bases[1],rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(contexts[0],contexts[1],rtol=1e-6,atol=1e-6)
        self.assertGreater(abs(rates[0][0]-rates[1][0]),1.)
        self.assertNotEqual(bool(rates[0][0]<req[0]),bool(rates[1][0]<req[0]))
        self.assertGreater(abs(ready[0][0]-ready[1][0]),.05)
        self.assertLess(ready[0][0],1.)
        self.assertGreater(ready[1][0],1.)
        self.assertGreater(ready[0][3],1.)

    def test_ready_context_does_not_mutate_real_hardware(self):
        _,_,_,_,_,hw = counterexample()
        np.testing.assert_array_equal(np.flatnonzero(hw.active),[11,12,13,15])
        self.assertEqual(hw.countdown[14],0)
        np.testing.assert_array_equal(np.flatnonzero(hw.command),[11,12,13,15])


if __name__ == '__main__':
    bases,contexts,rates,req,ready,_ = counterexample()
    print('full-prefix feature max difference', np.max(abs(contexts[0]-contexts[1])))
    print('available-mask rates', [r.tolist() for r in rates], 'request',req.tolist())
    print('readiness-conditional feature max difference',np.max(abs(ready[0]-ready[1])))
    unittest.main()
