import unittest
from major_revision.action_value_study import choose_baseline


class ActionValueSelectionTests(unittest.TestCase):
    def test_feasibility_then_energy_with_stable_ties(self):
        base=dict(minimum=1,margin=1.)
        rows=[dict(base,energy_j=90.,outage=.1),dict(base,energy_j=100.,outage=.04),
              dict(base,minimum=4,energy_j=100.,outage=.03)]
        self.assertEqual(choose_baseline(rows),rows[1])
        self.assertEqual(choose_baseline([rows[0],dict(base,energy_j=150.,outage=.06)])['outage'],.06)


if __name__=='__main__':
    unittest.main()
