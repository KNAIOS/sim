import unittest
from major_revision.full_domain_report import selection_from_validation,summarize


class FullDomainReportTests(unittest.TestCase):
    def test_selection_does_not_use_development(self):
        rows=[]
        for price in (1,2,4):
            for episode in (100,200,300):
                for j in range(3):
                    for i in range(12):
                        rows.append(dict(price=price,episode=episode,scenario=j,trajectory=i,
                            outage=.08 if price==1 else .03,
                            energy_j=100+price+episode/1000))
        result=selection_from_validation(rows)
        self.assertEqual((result['price'],result['episode']),(2,100))
        self.assertEqual(result['violation'],0.)

    def test_paired_gate_keeps_reliability(self):
        rows=[]
        for j in range(3):
            for i in range(16):
                for method,seeds,energy in (('Full-domain DQN',(11,22),100.),
                        ('prs',(0,),110.),('cost',(0,),120.),('full',(0,),305.)):
                    for seed in seeds:
                        rows.append(dict(scenario=j,trajectory=i,seed=seed,method=method,
                            energy_j=energy+i,outage=.02,
                            operating_j=energy+i-2,switching_j=1.,pilot_j=1.,
                            mean_count=4.,below_four_fraction=.1))
        _,paired,_,_,gates=summarize(rows)
        self.assertTrue(all(r['promote'] for r in gates))
        comparison=next(r for r in paired if r['scenario']==0 and r['baseline']=='prs')
        self.assertEqual(comparison['energy_j_trace_low'],-10.)
        self.assertEqual(comparison['energy_j_trace_high'],-10.)
        for row in rows:
            if row['method']=='Full-domain DQN':
                row['outage']=.06
        self.assertTrue(all(not r['promote'] for r in summarize(rows)[-1]))


if __name__=='__main__':
    unittest.main()
