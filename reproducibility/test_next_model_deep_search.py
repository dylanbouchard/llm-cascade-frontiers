import unittest
from itertools import product
import numpy as np
from next_model_diff01 import fit_edges,cutoffs,predict,evaluate
from next_model_deep_search import search,merge,kernel
from exact_five_bounded import search_chain

class DeepTests(unittest.TestCase):
    def data(self,n=8):
        rng=np.random.default_rng(524)
        return {m:dict(x=rng.integers(0,3,(n,5)).astype(float),y=rng.integers(0,2,n).astype(float),c=rng.uniform(i+1,i+1.1,n)) for i,m in enumerate('abcde')}
    def test_brute_force(self):
        cal=self.data();budgets=np.linspace(1.2,16,31)
        for arm in ('diff01_ridge','correctness_ridge','mean_token_negentropy'):
            models,edges=fit_edges(cal,arm)
            for k in (4,5):
                seq=tuple(models[:k]); options=[cutoffs(predict(cal[a]['x'],edges[a,b])) for a,b in zip(seq[:-1],seq[1:])]
                candidates=[]
                for ts in product(*options):
                    p=dict(models=list(seq),stages=[dict(predictor=edges[a,b],threshold=t) for (a,b),t in zip(zip(seq[:-1],seq[1:]),ts)])
                    c,q,_=evaluate(p,cal);candidates.append((c,q))
                p=dict(models=[models[0]],stages=[],cal_cost=cal[models[0]]['c'].mean(),cal_quality=cal[models[0]]['y'].mean())
                old=[p]*len(budgets)
                best,witness,cuts,stats=search(cal,seq,edges,budgets,old)
                self.assertTrue(stats['complete'])
                selected=merge(old.copy(),budgets,seq,edges,best,witness,cuts)
                for b,p in zip(budgets,selected):
                    eligible=[(c,q) for c,q in candidates if c<=b+1e-12]
                    q=max(q for c,q in eligible);c=min(c for c,q1 in eligible if q1==q)
                    np.testing.assert_allclose(evaluate(p,cal)[:2],[c,q],atol=1e-12)
    def test_existing_five_solver(self):
        cal=self.data(35);models,edges=fit_edges(cal,'diff01_ridge');seq=tuple(models)
        budgets=np.linspace(1.2,16,50)
        p=dict(models=[models[0]],stages=[],cal_cost=cal[models[0]]['c'].mean(),cal_quality=cal[models[0]]['y'].mean())
        old=[p]*len(budgets)
        best,witness,cuts,stats=search(cal,seq,edges,budgets,old)
        data={m:dict(correct=cal[m]['y'],costs=cal[m]['c'],scores=predict(cal[m]['x'],edges[m,models[i+1]]) if i<4 else np.zeros(35)) for i,m in enumerate(models)}
        ref,_,stat=search_chain(data,seq,max(budgets),0.,old)
        self.assertTrue(stat['complete'] and stats['complete'])
        for b in budgets:
            q1=np.flatnonzero(best<=b+1e-12)[-1];q2=np.flatnonzero(ref<=b+1e-12)[-1]
            self.assertEqual(q1,q2)
            self.assertAlmostEqual(best[q1],ref[q2],places=12)

if __name__=='__main__':unittest.main()
