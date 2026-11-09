"""Synthetic unit fixtures test the gate only; never experimental evidence."""
import unittest

from joint_power.matched_four.high_confirmation import NUMBER, PRIMARY, analyze, specifications


def fixture(primary_energy=90., second_seed_outage=.03):
    rows = []
    for method, seed in specifications():
        for i in range(NUMBER):
            energy = primary_energy if method == PRIMARY else 100.
            if method == 'Expanded budgetPRS':
                energy = 80.
            outage = second_seed_outage if method == PRIMARY and seed == 22 else .03
            rows.append(dict(method=method, seed=seed, trajectory=i, energy_j=energy,
                outage=outage, circuit_j=energy, pa_j=0., switching_j=0., pilot_j=0.,
                mean_active=4., mean_command=4.))
    return rows


class ConfirmationGateTests(unittest.TestCase):
    def test_stronger_expanded_search_is_not_ignored(self):
        _, _, gate = analyze(fixture())
        self.assertEqual(gate['reference']['method'], 'Expanded budgetPRS')
        self.assertFalse(gate['passed'])

    def test_conditional_pass_does_not_close_revision(self):
        _, _, gate = analyze(fixture(primary_energy=70.))
        self.assertTrue(gate['passed'])
        self.assertFalse(gate['R1_closed'])
        self.assertTrue(gate['nominal_development_failure_unchanged'])

    def test_each_seed_must_meet_outage_constraint(self):
        _, _, gate = analyze(fixture(primary_energy=70., second_seed_outage=.06))
        self.assertFalse(gate['passed'])


if __name__ == '__main__':
    unittest.main()
