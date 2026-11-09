import unittest
from dataclasses import fields,replace
import numpy as np
from future_vr.config import make_config
from future_vr.channel import generate_trajectory,Trajectory
from major_revision.branching_channel import BranchingChannel,replace_future


class BranchingChannelTests(unittest.TestCase):
    def test_every_field_matches_original_generator_bitwise(self):
        for mode,speed in (('smoke',.5),('validation',2.),('validation',3.)):
            cfg=replace(make_config(mode),episode_steps=140)
            expected=generate_trajectory(cfg,1040900001,speed)
            actual,_=BranchingChannel(cfg,1040900001,speed).generate(140)
            for field in fields(Trajectory):
                np.testing.assert_array_equal(getattr(actual,field.name),getattr(expected,field.name))

    def test_snapshot_exact_suffix_and_independent_future_preserve_prefix(self):
        cfg=replace(make_config('validation'),episode_steps=40)
        expected=generate_trajectory(cfg,1040900002,2.)
        engine=BranchingChannel(cfg,1040900002,2.)
        _,snapshots=engine.generate(40,capture=(0,16))
        for t in (0,16):
            suffix,_=snapshots[t].fork().generate(15)
            for field in fields(Trajectory):
                np.testing.assert_array_equal(getattr(suffix,field.name),getattr(expected,field.name)[t+1:t+16])
            fresh,_=snapshots[t].fork(seed=1040901000+t).generate(15)
            changed=replace_future(expected,t,fresh,1.)
            self.assertFalse(np.array_equal(changed.channel[t+1:t+16],expected.channel[t+1:t+16]))
            for field in fields(Trajectory):
                np.testing.assert_array_equal(getattr(changed,field.name)[:t+1],getattr(expected,field.name)[:t+1])
            np.testing.assert_array_equal(snapshots[t].fork().generate(15)[0].channel,suffix.channel)


if __name__=='__main__':
    unittest.main()
