import unittest
from itertools import product
import numpy as np
from foc_subsequence_compare import chain_search, replay

class SubsequenceTests(unittest.TestCase):
    def test_three_through_five_against_small_exhaustive_search(self):
        rng=np.random.default_rng(21)
        for d in [3,4,5]:
            seq=list(map(str,range(d)))
            data={m:dict(scores=rng.choice([.1,.3,.6,.9],60),correct=rng.integers(0,2,60).astype(float),costs=rng.uniform(1+int(m),1.5+int(m),60)) for m in seq}
            budgets=np.linspace(data['0']['costs'].mean(),sum(data[m]['costs'].mean() for m in seq),15)
            tt,cc,qq,stats,_=chain_search(data,seq,budgets,iters=6,stride=3)
            rc,rq=replay(np.zeros(len(budgets),int),tt,[seq],data)
            np.testing.assert_allclose(cc,rc,atol=1e-12,rtol=0)
            np.testing.assert_allclose(qq,rq,atol=1e-12,rtol=0)
            self.assertTrue(np.all(cc<=budgets+1e-12))
            brute=np.array(list(product([-np.inf,.3,.6,.9,np.inf],repeat=d-1)))
            bc,bq=replay(np.zeros(len(brute),int),brute,[seq],data)
            optimum=np.array([bq[bc<=b+1e-12].max() for b in budgets])
            self.assertTrue(np.all(qq<=optimum+1e-12))
            self.assertTrue(np.all(np.diff(qq)>=-1e-12))
            self.assertGreater(stats['starts'],len(budgets))

    def test_unreachable_suffix_and_infeasible_chain(self):
        data={str(i):dict(scores=np.ones(60),correct=np.full(60,float(i==0)),costs=np.full(60,float(i+1))) for i in range(4)}
        ts,c,q,_,_=chain_search(data,list(data),np.array([.5,1.,2.,10.]))
        self.assertLess(q[0],0)
        np.testing.assert_allclose(q[1:],1)
        np.testing.assert_allclose(c[1:],1)

    def test_positive_benefit_terminal_endpoint(self):
        for d in [4,5]:
            data={str(i):dict(scores=np.linspace(0,1,60),correct=np.full(60,float(i==d-1)),costs=np.full(60,float(i+1))) for i in range(d)}
            budget=d*(d+1)/2
            ts,c,q,_,_=chain_search(data,list(data),np.array([budget]))
            self.assertEqual(q[0],1)
            self.assertAlmostEqual(c[0],budget)

if __name__=='__main__':unittest.main()
