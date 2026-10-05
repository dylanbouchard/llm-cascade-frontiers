"""Independent checks of pruning against already-optimal shallower policies."""
import unittest
from itertools import product
import numpy as np
from next_model_diff01 import fit_edges,select,cutoffs,predict,evaluate
from next_model_deep_search import search,merge

class IncumbentTests(unittest.TestCase):
    def test_nested_incumbent_pruning(self):
        for seed in range(5):
            rng=np.random.default_rng(seed)
            cal={m:dict(x=rng.integers(0,3,(7,5)).astype(float),y=rng.integers(0,2,7).astype(float),c=rng.uniform(i+1,i+1.1,7)) for i,m in enumerate('abcde')}
            budgets=np.linspace(1.2,16,31)
            models,edges=fit_edges(cal,'diff01_ridge')
            frozen,_=select(cal,budgets,'diff01_ridge')
            for k in (4,5):
                seq=tuple(models[:k]);old=frozen['3']
                options=[cutoffs(predict(cal[a]['x'],edges[a,b])) for a,b in zip(seq[:-1],seq[1:])]
                candidates=[]
                for ts in product(*options):
                    p=dict(models=list(seq),stages=[dict(predictor=edges[a,b],threshold=t) for (a,b),t in zip(zip(seq[:-1],seq[1:]),ts)])
                    c,q,_=evaluate(p,cal);candidates.append((c,q))
                best,witness,cuts,stats=search(cal,seq,edges,budgets,old)
                result=merge(list(old),budgets,seq,edges,best,witness,cuts)
                self.assertTrue(stats['complete'])
                for b,original,chosen in zip(budgets,old,result):
                    eligible=[(c,q) for c,q in candidates if c<=b+1e-12]+[(original['cal_cost'],original['cal_quality'])]
                    quality=max(q for c,q in eligible)
                    cost=min(c for c,q in eligible if q==quality)
                    np.testing.assert_allclose(evaluate(chosen,cal)[:2],[cost,quality],atol=1e-12)

if __name__=='__main__':unittest.main()
