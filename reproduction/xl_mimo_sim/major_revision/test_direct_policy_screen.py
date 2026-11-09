import unittest
from major_revision.direct_policy_screen import select_point, promotion


class DirectPolicyScreenTests(unittest.TestCase):
    def test_selection_does_not_select_by_favourable_comparison(self):
        points = [dict(margin=.9,energy_j=80.,outage=.06),
                  dict(margin=1.,energy_j=90.,outage=.05),
                  dict(margin=1.1,energy_j=100.,outage=.01)]
        self.assertEqual(select_point(points)['margin'],1.)
        self.assertEqual(select_point(points[:1])['margin'],.9)

    def test_gate_keeps_reliability_and_state_dependence_requirements(self):
        state = dict(energy_j=97.,outage=.04)
        pure = dict(energy_j=100.,outage=.04)
        matched = dict(energy_j=100.)
        bias = dict(energy_j=98.)
        self.assertTrue(promotion(state,pure,matched,bias,True)['passed'])
        self.assertFalse(promotion(state,pure,matched,bias,False)['passed'])
        self.assertFalse(promotion(state,pure,dict(energy_j=98.),bias,True)['passed'])
        self.assertFalse(promotion(state,pure,matched,dict(energy_j=97.),True)['passed'])
        self.assertFalse(promotion(state,None,matched,bias,True)['passed'])


if __name__ == '__main__':
    unittest.main()
