import unittest
from dataclasses import replace
import numpy as np
from future_vr.config import make_config
from major_revision.demonstration_data import collect_episode
from major_revision.demonstration_training import imitation_key
from major_revision.full_domain_baselines import baseline_episode
from major_revision.pilot import trajectory


class DemonstrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=replace(make_config('validation'),episode_steps=12)
        self.tr=trajectory(replace(self.cfg,episode_steps=13),991234567,(2.,1.))

    def test_teacher_transition_collection_replays_exactly(self):
        setting=dict(minimum=4,margin=1.08)
        data,result=collect_episode(self.cfg,self.tr,setting)
        expected=baseline_episode(self.cfg,self.tr,'prs',1.08,minimum=4)
        for key in ('energy_j','outage','operating_j','switching_j','pilot_j','mean_count'):
            self.assertAlmostEqual(result[key],expected[key],places=9)
        self.assertEqual(data['state'].shape,(3,235))
        self.assertEqual(data['next_state'].shape,(3,235))
        np.testing.assert_array_equal(data['action'],data['expert_action'])
        np.testing.assert_array_equal(data['next_state'][:-1],data['state'][1:])
        self.assertEqual(data['slot'][-1],8)

    def test_student_actions_are_not_teacher_labels(self):
        class Student:
            def choose(self,state,epsilon):
                return 0,1
        data,_=collect_episode(self.cfg,self.tr,dict(minimum=4,margin=1.08),Student())
        self.assertTrue(np.all(data['action']==0))
        self.assertTrue(np.all(data['expert_action']>=3))

    def test_selection_respects_each_scenario(self):
        reference={j:dict(outage=.03) for j in range(3)}
        bad=[dict(scenario=j,energy_j=100.,outage=.03 if j<2 else .08) for j in range(3)]
        good=[dict(scenario=j,energy_j=120.,outage=.03) for j in range(3)]
        self.assertLess(imitation_key(good,reference),imitation_key(bad,reference))


if __name__=='__main__':
    unittest.main()
