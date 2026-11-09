import unittest
import numpy as np
from major_revision.rollout_value_decision_audit import action_result


class DecisionValueAuditTests(unittest.TestCase):
    def test_predicted_gain_does_not_substitute_realized_gain(self):
        predicted=np.array([[-.1,0.],[0.,0.],[.1,0.]])
        truth=np.array([[.2,0.],[0.,0.],[-.3,0.]])
        row=action_result(predicted,truth,1,2.)
        self.assertEqual(row['action'],1)
        self.assertEqual(row['predicted_improvement_sample_worse'],1)
        self.assertAlmostEqual(row['sampled_cost_difference'],.2)
        row=action_result(predicted*0.,truth,1,2.)
        self.assertEqual(row['changed'],0)
        self.assertEqual(row['sampled_cost_difference'],0.)


if __name__=='__main__':
    unittest.main()
