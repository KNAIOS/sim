"""Transfer-plan cardinalities and paired aggregation, synthetic fixtures only."""
import unittest

from joint_power.standard_transfer.control import REFERENCES
from joint_power.standard_transfer.study import LOADS, NUMBER, STATES, analyze, specifications


class TransferStudyTests(unittest.TestCase):
    def test_locked_plan_keeps_both_original_and_information_matched_search(self):
        self.assertEqual(NUMBER, 20)
        self.assertEqual(LOADS, (.5, 1.))
        self.assertEqual(len(specifications()), 9)
        self.assertEqual(2*NUMBER*len(LOADS)*len(specifications()), 720)
        self.assertTrue({'PRS4', 'budgetPRS4', 'PRS1', 'budgetPRS1', 'WBbudgetPRS4',
                         'WBbudgetPRS1', 'Full array'} == set(REFERENCES))

    def test_constant_paired_fixture_is_preserved(self):
        rows = []
        for state in STATES:
            for load in LOADS:
                for method, seed in specifications():
                    for index in range(NUMBER):
                        learned = method == 'Value4'
                        rows.append(dict(state=state, load=load, method=method, seed=seed,
                            trajectory=index, energy_j=100.+index+int(learned), outage=.01*int(learned),
                            circuit_j=0., pa_j=0., switching_j=0., pilot_j=0., mean_active=0.,
                            numerical_fallbacks=0))
        summaries, paired = analyze(rows)
        self.assertEqual(len(summaries), 32)
        self.assertEqual(len(paired), 56)
        for row in paired:
            expected = 1. if row['metric'] == 'energy_j' else .01
            for key in ('difference', 'ci_low', 'ci_high'):
                self.assertAlmostEqual(row[key], expected, places=12)


if __name__ == '__main__':
    unittest.main()
