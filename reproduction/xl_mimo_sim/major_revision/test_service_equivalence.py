import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from future_vr.system import HardwareState
from major_revision.action_value import action_size,observation_size
from major_revision.gram_control import representation_size
from major_revision.service_equivalence import service_signatures,service_projection,project_risk


class ServiceEquivalenceTests(unittest.TestCase):
    def test_signatures_match_original_hardware_with_existing_countdowns(self):
        rng=np.random.default_rng(1070900001)
        for delay in (0,1,4,7):
            cfg=replace(make_config('validation'),wake_delay_slots=delay)
            for _ in range(5):
                q=cfg.subarrays
                active=rng.random(q)<.5
                countdown=rng.integers(0,delay+1,q)
                countdown[active]=0
                x=np.zeros((1,observation_size(cfg)),np.float32)
                start=q*cfg.users**2+3*q*cfg.users
                x[0,start:start+q]=active
                x[0,start+2*q:start+3*q]=countdown/cfg.horizon
                order=rng.permutation(q)
                commands=np.zeros((q,q),bool)
                for a in range(q):
                    commands[a,order[:a+1]]=True
                x[0,representation_size(cfg):].reshape(q,action_size(cfg))[:,:q]=commands
                signatures=service_signatures(cfg,x)[0]
                for a,command in enumerate(commands):
                    hardware=HardwareState(cfg)
                    hardware.active=active.copy()
                    hardware.countdown=countdown.copy()
                    for h in range(4):
                        actual,_,_=hardware.apply(command,1.)
                        np.testing.assert_array_equal(actual,signatures[a,h])
                        hardware.tick()

    def test_absolute_projection_identity_and_c16_counterexample(self):
        signatures=np.zeros((1,2,4,1),bool)
        projection=service_projection(signatures)
        np.testing.assert_allclose(projection[0]@projection[0],projection[0])
        np.testing.assert_array_equal(projection[0],projection[0].T)
        # Identical true C4 risk .5, but tail errors happen to cancel root errors.
        truth=np.zeros((1,2,2,2))
        truth[:,:,0,1]=.5
        truth[:,:,1,1]=1.5
        prediction=truth.copy()
        prediction[0,:,0,1]=[.4,.6]
        projected=project_risk(prediction,projection)
        old=((prediction[:,:,0,1]-truth[:,:,0,1])**2).sum()
        new=((projected[:,:,0,1]-truth[:,:,0,1])**2).sum()
        removed=((prediction[:,:,0,1]-projected[:,:,0,1])**2).sum()
        self.assertAlmostEqual(old,new+removed,places=12)
        self.assertLess(new,old)
        self.assertGreater(((projected[:,:,1,1]-truth[:,:,1,1])**2).sum(),
                           ((prediction[:,:,1,1]-truth[:,:,1,1])**2).sum())
        np.testing.assert_array_equal(prediction[:,:,:,0],projected[:,:,:,0])
        np.testing.assert_allclose(projected[:,:,1,1]-projected[:,:,0,1],
                                   prediction[:,:,1,1]-prediction[:,:,0,1])


if __name__=='__main__':
    unittest.main()
