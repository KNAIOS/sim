"""Synthetic fixtures test the fixed analysis; they are not experimental data."""
import unittest

from joint_power.focus_confirmation.study import analyze, NUMBER, SPECS


def fixture(energy_delta=-2., outage=.01):
    return [dict(state='NLOS', load=.5, trajectory=i, method=method, seed=seed,
        energy_j=100.+i+(energy_delta if seed else 0.), outage=outage if seed else .005,
        circuit_j=0., pa_j=0., switching_j=0., pilot_j=0., mean_active=4., numerical_fallbacks=0)
        for i in range(4) for method, seed in SPECS]


class ConfirmationTests(unittest.TestCase):
    def test_locked_scope_and_cardinality(self):
        self.assertEqual(NUMBER, 40)
        self.assertEqual(SPECS, (('Value4', 11), ('Value4', 22), ('WBbudgetPRS4', 0)))

    def test_constant_paired_difference(self):
        _, paired, report = analyze(fixture(), number=4)
        for name in ('difference', 'ci_low', 'ci_high'):
            self.assertAlmostEqual(paired[0][name], -2.)
        self.assertTrue(report['independent_confirmation_passed'])
        self.assertFalse(report['globally_optimal_proven'])

    def test_more_energy_fails(self):
        _, _, report = analyze(fixture(energy_delta=1.), number=4)
        self.assertFalse(report['energy_supported'])
        self.assertFalse(report['independent_confirmation_passed'])

    def test_outage_violation_fails_even_when_energy_saves(self):
        _, _, report = analyze(fixture(outage=.07), number=4)
        self.assertTrue(report['energy_supported'])
        self.assertFalse(report['reliability_supported'])
        self.assertFalse(report['independent_confirmation_passed'])

    def test_missing_or_duplicate_rows_are_rejected(self):
        rows = fixture()
        with self.assertRaises(AssertionError):
            analyze(rows[:-1], number=4)
        with self.assertRaises(AssertionError):
            analyze(rows[:-1]+[rows[0]], number=4)

    def test_pairing_removes_shared_trace_variation(self):
        rows = fixture()
        for row in rows:
            row['energy_j'] += row['trajectory']**3*1000
        _, paired, _ = analyze(rows, number=4)
        self.assertAlmostEqual(paired[0]['ci_low'], -2.)
        self.assertAlmostEqual(paired[0]['ci_high'], -2.)


if __name__ == '__main__':
    unittest.main()
