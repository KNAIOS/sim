import unittest
from major_revision.action_value_report import screening_gate


class ActionValueGateTests(unittest.TestCase):
    def test_learning_energy_and_reliability_all_required(self):
        ref=dict(energy_j=100.,outage=.03)
        prior=dict(energy_j=110.,outage=.10)
        good=dict(energy_j=97.,outage=.034)
        self.assertTrue(screening_gate(50,good,ref,prior))
        self.assertFalse(screening_gate(0,good,ref,prior))
        self.assertFalse(screening_gate(50,dict(good,energy_j=99.),ref,prior))
        self.assertFalse(screening_gate(50,dict(good,outage=.036),ref,prior))
        self.assertFalse(screening_gate(50,good,None,prior))
        self.assertFalse(screening_gate(50,good,ref,dict(prior,energy_j=90.)))


if __name__=='__main__':
    unittest.main()
