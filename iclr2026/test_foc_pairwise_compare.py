import unittest
from itertools import combinations
import numpy as np
from foc_pairwise_compare import pair_surface, stationary_candidates, select, evaluate

class PairwiseTests(unittest.TestCase):
    def test_ties_and_variable_cost_replay(self):
        s=np.array([.8,.2,.2,.5])
        ca=np.array([1.,2.,1.,1.]); cb=np.array([3.,2.,4.,1.])
        qa=np.array([1.,0.,1.,0.]); qb=np.array([0.,1.,1.,1.])
        costs,q,cuts,_=pair_surface(s,ca,cb,qa,qb,3)
        for c,n,t in zip(costs,q,cuts):
            e=s<t
            self.assertAlmostEqual(c,np.mean(ca+e*cb))
            self.assertEqual(n,np.where(e,qb,qa).sum())
        self.assertEqual(len(costs),4)

    def test_stationary_crossing_and_plateau(self):
        np.testing.assert_array_equal(stationary_candidates([1,1,0,0,-1,-1]),[0,1,2,3,4,5])
        np.testing.assert_array_equal(stationary_candidates([1,1,1]),[0,2])
        np.testing.assert_array_equal(stationary_candidates([-1,0,1]),[0,2])

    def test_against_independent_bruteforce(self):
        rng=np.random.default_rng(13)
        data={str(i):dict(scores=rng.choice([.1,.3,.7,.9],20),costs=rng.uniform(i+1,i+2,20),correct=rng.integers(0,2,20).astype(float)) for i in range(4)}
        budgets=np.linspace(1.8,10,35)
        exact,seqs,_=select(data,budgets,'exact')
        policies=[]
        for m in data:
            policies.append((data[m]['costs'].mean(),data[m]['correct'].mean()))
        for a,b in combinations(data,2):
            for t in [-np.inf,.3,.7,.9,np.inf]:
                e=data[a]['scores']<t
                policies.append(((data[a]['costs']+e*data[b]['costs']).mean(),np.where(e,data[b]['correct'],data[a]['correct']).mean()))
        for j,budget in enumerate(budgets):
            c,q=min((p for p in policies if p[0]<=budget+1e-12),key=lambda p:(-p[1],p[0]))
            self.assertAlmostEqual(c,exact['cal_cost'][j])
            self.assertAlmostEqual(q,exact['cal_accuracy'][j])
        for method in ['foc','boundary']:
            p,seqs,_=select(data,budgets,method,6)
            c,q=evaluate(p,seqs,data)
            self.assertTrue(np.all(c<=budgets+1e-12))
            self.assertTrue(np.all(q<=exact['cal_accuracy']+1e-12))

    def test_positive_benefit_hits_budget(self):
        data={'a':dict(scores=np.linspace(0,1,20),costs=np.ones(20),correct=np.zeros(20)),
              'b':dict(scores=np.zeros(20),costs=np.full(20,2.),correct=np.ones(20))}
        budgets=np.array([1.,1.3,1.9])
        p,_,_=select(data,budgets,'foc',6)
        np.testing.assert_allclose(p['cal_cost'],budgets)
        np.testing.assert_allclose(p['cal_accuracy'],[0,.15,.45])

if __name__=='__main__': unittest.main()
