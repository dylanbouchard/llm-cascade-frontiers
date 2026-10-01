import tempfile
import unittest
from itertools import combinations, product
from pathlib import Path
from unittest.mock import patch
import numpy as np
from exact_heldout_depth import exact_select, evaluate_policy, encode
from exact_heldout_four import select_four
from exact_heldout_five import select_five
from optuna_frontier import simulate_cascade


class HeldoutFiveTest(unittest.TestCase):
    def test_nested_class_against_brute_force(self):
        for seed in range(12):
            rng=np.random.default_rng(seed)
            data={str(m):dict(scores=rng.integers(0,3,7).astype(float),
                             correct=rng.integers(0,2,7).astype(float),
                             costs=rng.integers(0,8,7).astype(float)/8) for m in range(6)}
            budgets=np.linspace(0,3,41)
            old,_=exact_select(data,budgets)
            four,_=select_four(data,budgets,encode(old['3']))
            selected,_=select_five(data,budgets,encode(four))
            models=sorted(data,key=lambda m:(data[m]['costs'].mean(),m))
            candidates=[]
            for depth in range(1,6):
                for seq in combinations(models,depth):
                    cuts=[np.r_[-np.inf,np.unique(data[m]['scores'])[1:],np.inf] for m in seq[:-1]]
                    candidates.extend(simulate_cascade(seq,t,data) for t in product(*cuts))
            for budget,p in zip(budgets,selected):
                feasible=[(c,q) for c,q in candidates if c<=budget+1e-12]
                if not feasible:
                    self.assertIsNone(p)
                    continue
                c,q=min(feasible,key=lambda p:(-p[1],p[0]))
                self.assertAlmostEqual(c,p['cal_cost'])
                self.assertAlmostEqual(q,p['cal_quality'])
                pc,pq,_=evaluate_policy(encode(p),data)
                self.assertAlmostEqual(c,pc)
                self.assertAlmostEqual(q,pq)

    def test_fifth_stage_needed_and_checkpoint_resume(self):
        data={str(m):dict(scores=(np.arange(5)==m).astype(float),
                         correct=(np.arange(5)==m).astype(float),costs=np.ones(5)) for m in range(4)}
        data['4']=dict(scores=np.ones(5),correct=np.ones(5),costs=np.full(5,5.))
        budgets=[3.8]
        old,_=exact_select(data,budgets)
        four,_=select_four(data,budgets,old['3'])
        with tempfile.TemporaryDirectory() as temp:
            selected,stats=select_five(data,budgets,four,Path(temp),'fixture')
            self.assertEqual(selected[0]['cal_quality'],1.)
            self.assertEqual(len(selected[0]['models']),5)
            self.assertLess(four[0]['cal_quality'],selected[0]['cal_quality'])
            with patch('exact_heldout_five.search_chain',side_effect=AssertionError('Recomputed checkpoint')):
                resumed,resumed_stats=select_five(data,budgets,four,Path(temp),'fixture')
            self.assertEqual(selected,resumed)
            self.assertEqual(stats,resumed_stats)
            with self.assertRaises(ValueError):
                select_five(data,budgets,four,Path(temp),'changed')


if __name__=='__main__':
    unittest.main()
