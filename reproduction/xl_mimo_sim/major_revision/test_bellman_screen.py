import unittest

from major_revision.bellman_screen import promotion, select_price


class BellmanScreenTests(unittest.TestCase):
    def test_selection_is_feasibility_then_energy_not_frontier_gain(self):
        points = [dict(price=.5, energy_j=80., outage=.1),
                  dict(price=1., energy_j=95., outage=.05),
                  dict(price=2., energy_j=100., outage=.03)]
        self.assertEqual(select_price(points)['price'], 1.)
        self.assertEqual(select_price(points[:1])['price'], .5)

    def test_gate_requires_both_fair_controls_and_all_scenario_feasibility(self):
        proposed = dict(energy_j=97., outage=.04)
        pure = dict(energy_j=100., outage=.039)
        matched = dict(energy_j=100.)
        myopic = dict(energy_j=98.)
        self.assertTrue(promotion(proposed, pure, matched, myopic, True)['passed'])
        self.assertFalse(promotion(proposed, pure, matched, myopic, False)['passed'])
        self.assertFalse(promotion(proposed, pure, dict(energy_j=98.), myopic, True)['passed'])
        self.assertFalse(promotion(proposed, pure, matched, dict(energy_j=96.), True)['passed'])
        self.assertFalse(promotion(proposed, None, matched, myopic, True)['passed'])


if __name__ == '__main__':
    unittest.main()
