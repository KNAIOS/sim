import unittest
from major_revision.demonstration_report import initialization_gate


class InitializationGateTests(unittest.TestCase):
    def test_all_three_conditions_are_required(self):
        self.assertTrue(initialization_gate(.90,[1.,1.1,1.],[0.,.01,0.]))
        self.assertFalse(initialization_gate(.89,[1.,1.,1.],[0.,0.,0.]))
        self.assertFalse(initialization_gate(.99,[1.,1.11,1.],[0.,0.,0.]))
        self.assertFalse(initialization_gate(.99,[1.,1.,1.],[0.,0.,.011]))


if __name__=='__main__':
    unittest.main()
