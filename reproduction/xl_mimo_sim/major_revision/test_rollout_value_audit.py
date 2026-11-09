import unittest
import numpy as np
from major_revision.rollout_value_audit import validate_arrays


class RolloutValueAuditTests(unittest.TestCase):
    def test_checks_reference_subtraction_and_unique_states(self):
        costs=np.ones((2,16,2,2))
        costs[:,:,1,:]=2.
        data=dict(features=np.zeros((2,1436)),reference=np.array([1,2]),costs=costs,
            difference=np.zeros_like(costs),keys=np.array([[0,0,0],[0,0,16]]))
        validate_arrays(data,2)
        data['difference'][0,0,0,0]=1.
        with self.assertRaises(AssertionError):
            validate_arrays(data,2)
        data['difference'][0,0,0,0]=0.
        data['keys'][1]=data['keys'][0]
        with self.assertRaises(AssertionError):
            validate_arrays(data,2)


if __name__=='__main__':
    unittest.main()
