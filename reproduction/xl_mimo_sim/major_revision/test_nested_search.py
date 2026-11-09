import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from major_revision.full_domain import FullDomainRollout
from major_revision.nested_search import teacher_command,choose_setting
from major_revision.pilot import trajectory


class NestedSearchTests(unittest.TestCase):
    def test_teacher_does_not_change_observation_dimension(self):
        cfg=replace(make_config('validation'),episode_steps=8)
        tr=trajectory(cfg,991111111,(2.,1.))
        for minimum in (1,4):
            expert=FullDomainRollout(cfg,tr,minimum)
            student=FullDomainRollout(cfg,tr,1)
            expected=expert.command(1.08,'prs')
            actual=teacher_command(student,dict(minimum=minimum,margin=1.08))
            np.testing.assert_array_equal(expected,actual)
            self.assertEqual(student.observation()[0].shape,(235,))
            self.assertEqual(student.minimum,1)

    def test_old_floor_not_dropped_when_better(self):
        rows=[dict(minimum=1,margin=1.16,energy_j=250,outage=.03),
              dict(minimum=4,margin=1.04,energy_j=190,outage=.04),
              dict(minimum=1,margin=1,energy_j=130,outage=.15)]
        self.assertEqual(choose_setting(rows)['minimum'],4)


if __name__=='__main__':
    unittest.main()
