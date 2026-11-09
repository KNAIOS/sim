import unittest
from major_revision.absolute_return_audit import screen_gate,rectangular


class AbsoluteReturnAuditTests(unittest.TestCase):
    def test_gate_does_not_promote_savings_with_reliability_loss(self):
        baseline=dict(energy_j=100.,outage=.04)
        local=dict(energy_j=105.,outage=.04)
        self.assertTrue(screen_gate(dict(energy_j=97.,outage=.04),baseline,local)['passed'])
        self.assertFalse(screen_gate(dict(energy_j=90.,outage=.06),baseline,local)['passed'])
        self.assertFalse(screen_gate(dict(energy_j=99.,outage=.04),baseline,local)['passed'])
        self.assertFalse(screen_gate(dict(energy_j=97.,outage=.046),baseline,local)['passed'])
        self.assertFalse(screen_gate(dict(energy_j=97.,outage=.04),None,local)['passed'])
        self.assertTrue(screen_gate(dict(energy_j=97.,outage=.04),baseline,dict(energy_j=90.,outage=.08))['passed'])

    def test_union_schema_keeps_missing_values_blank(self):
        result=rectangular([dict(method='search',margin=1.),dict(method='C16',seed=11)])
        self.assertEqual(result,[dict(method='search',margin=1.,seed=''),dict(method='C16',margin='',seed=11)])


if __name__=='__main__':
    unittest.main()
