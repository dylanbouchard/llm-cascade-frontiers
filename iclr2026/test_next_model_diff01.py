import unittest
from itertools import combinations, product
import numpy as np
from next_model_diff01 import fit_edges, select, ARMS, predict, cutoffs, evaluate


class Diff01Tests(unittest.TestCase):
    def data(self, n=12):
        rng = np.random.default_rng(819)
        return {m: dict(x=rng.integers(0, 3, (n, 5)).astype(float),
                       y=rng.integers(0, 2, n).astype(float),
                       c=rng.uniform(i+1, i+1.2, n)) for i, m in enumerate('abcd')}

    def test_exact_search_matches_brute_force_and_replays(self):
        data = self.data()
        budgets = np.linspace(1.2, 11, 23)
        for arm in ARMS:
            models, edges = fit_edges(data, arm)
            candidates = []
            for depth in (1, 2, 3):
                for seq in combinations(models, depth):
                    transitions = list(zip(seq[:-1], seq[1:]))
                    options = [cutoffs(predict(data[a]['x'], edges[a,b])) for a,b in transitions]
                    for thresholds in product(*options):
                        p = dict(models=list(seq), stages=[dict(predictor=edges[e], threshold=t)
                                                          for e,t in zip(transitions, thresholds)])
                        c,q,_ = evaluate(p, data)
                        candidates.append((depth,c,q))
            frozen,_ = select(data,budgets,arm)
            for depth in (1,2,3):
                for budget,p in zip(budgets,frozen[str(depth)]):
                    eligible = [(c,q) for k,c,q in candidates if k<=depth and c<=budget+1e-12]
                    best_q = max(q for c,q in eligible)
                    best_c = min(c for c,q in eligible if q==best_q)
                    np.testing.assert_allclose(evaluate(p,data)[:2], [best_c,best_q],atol=1e-12)
                    np.testing.assert_allclose(evaluate(p,data)[:2], [p['cal_cost'],p['cal_quality']],atol=1e-12)

    def test_next_model_only_target(self):
        data = self.data()
        _, before = fit_edges(data, 'diff01_ridge')
        data['c']['y'] = 1-data['c']['y']
        _, after = fit_edges(data, 'diff01_ridge')
        self.assertEqual(before['a','b'], after['a','b'])
        self.assertNotEqual(before['b','c'], after['b','c'])

    def test_harmful_and_helpful_have_opposite_scores(self):
        x = np.zeros((30,5))
        x[:,0] = np.tile([-1.,0.,1.],10)
        low = (x[:,0] == -1).astype(float)
        high = (x[:,0] == 1).astype(float)
        data = {'a':dict(x=x,y=low,c=np.ones(30)), 'b':dict(x=x,y=high,c=np.ones(30)*2)}
        _, edges = fit_edges(data,'diff01_ridge')
        s = predict(x,edges['a','b'])
        self.assertGreater(s[0],s[1])
        self.assertGreater(s[1],s[2])

    def test_constant_scores_and_endpoints(self):
        data = self.data()
        for d in data.values():
            d['x'][:] = 1.
        for arm in ARMS:
            frozen,_ = select(data,np.linspace(1.2,11,12),arm)
            for depth in ('2','3'):
                for p in frozen[depth]:
                    np.testing.assert_allclose(evaluate(p,data)[:2],[p['cal_cost'],p['cal_quality']],atol=1e-12)


if __name__ == '__main__':
    unittest.main()
