import unittest
from major_revision.rollout_value_closed_audit import promotion,rectangular


class RolloutClosedAuditTests(unittest.TestCase):
    def test_union_schema_preserves_all_values_without_first_row_bias(self):
        rows=[dict(method='cost',minimum=1),dict(method='C16',seed=11,price=2.)]
        result=rectangular(rows)
        self.assertEqual(set(result[0]),set(result[1]))
        self.assertEqual(result[0]['minimum'],1)
        self.assertEqual(result[1]['seed'],11)
        self.assertEqual(result[0]['price'],'')
        self.assertNotIn('seed',rows[0])

    def test_reliability_search_and_ablation_requirements(self):
        candidate=dict(energy_j=95.,outage=.045)
        reference=dict(energy_j=100.,outage=.04)
        local=dict(energy_j=100.,outage=.04)
        self.assertTrue(promotion(candidate,reference,local)['passed'])
        candidate['outage']=.051
        self.assertFalse(promotion(candidate,reference,local)['passed'])
        candidate['outage']=.04
        local['energy_j']=94.
        self.assertFalse(promotion(candidate,reference,local)['passed'])
        local['outage']=.08
        self.assertTrue(promotion(candidate,reference,local)['passed'])
        self.assertFalse(promotion(candidate,None,local)['passed'])


if __name__=='__main__':
    unittest.main()
