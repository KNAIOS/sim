import unittest
from major_revision.rollout_value_fit_audit import fit_gate


class RolloutValueFitAuditTests(unittest.TestCase):
    def test_both_targets_and_seeds_required(self):
        rows=[dict(seed=seed,horizon='C16',split='validation',scenario=-1,
            energy_mse=.5,energy_zero_mse=1.,outage_mse=.2,outage_zero_mse=1.) for seed in (11,22)]
        self.assertTrue(fit_gate(rows)['permit_closed_loop'])
        rows[1]['outage_mse']=.95
        self.assertFalse(fit_gate(rows)['permit_closed_loop'])
        with self.assertRaises(AssertionError):
            fit_gate(rows[:1])


if __name__=='__main__':
    unittest.main()
