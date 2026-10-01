import unittest
from itertools import combinations, product
import numpy as np
from exact_heldout_depth import exact_select, evaluate_policy, encode
from optuna_frontier import simulate_cascade


class ExactDepthTest(unittest.TestCase):
    def test_matches_brute_force_with_ties_and_variable_costs(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            cal = {str(m): dict(scores=rng.integers(0, 3, 9).astype(float),
                   correct=rng.integers(0, 2, 9).astype(float),
                   costs=rng.uniform(.1, 1., 9)+m) for m in range(4)}
            budgets = np.linspace(0, 12, 71)
            selected, counts = exact_select(cal, budgets)
            candidates = []
            models = sorted(cal, key=lambda m: (cal[m]['costs'].mean(), m))
            for depth in (1, 2, 3):
                for seq in combinations(models, depth):
                    cuts = [np.r_[-np.inf, np.unique(cal[m]['scores'])[1:], np.inf] for m in seq[:-1]]
                    for tau in product(*cuts):
                        c, q = simulate_cascade(seq, tau, cal)
                        candidates.append((c, q))
                for b, p in zip(budgets, selected[str(depth)]):
                    feasible = [(c, q) for c, q in candidates if c <= b+1e-12]
                    if not feasible:
                        self.assertIsNone(p)
                        continue
                    c, q = min(feasible, key=lambda v: (-v[1], v[0]))
                    self.assertAlmostEqual(p['cal_quality'], q)
                    self.assertAlmostEqual(p['cal_cost'], c)
                    ec, eq, calls = evaluate_policy(p, cal)
                    self.assertAlmostEqual(ec, c)
                    self.assertAlmostEqual(eq, q)
                    self.assertTrue(1 <= calls <= depth)

    def test_endpoints_and_frozen_test_execution(self):
        cal = {str(m): dict(scores=np.ones(4), correct=np.full(4, m, dtype=float),
                           costs=np.ones(4)) for m in (0, 1)}
        selected, _ = exact_select(cal, [1., 2.])
        self.assertEqual(selected['3'][1]['models'], ['1'])
        p = dict(models=['0', '1'], thresholds=[np.inf])
        test = {m: dict(d, scores=np.full(4, 100.)) for m, d in cal.items()}
        self.assertEqual(evaluate_policy(p, test), (2., 1., 2.))
        self.assertEqual(evaluate_policy(encode(p), test), (2., 1., 2.))
        p['thresholds'] = [-np.inf]
        self.assertEqual(evaluate_policy(p, test), (1., 0., 1.))


if __name__ == '__main__':
    unittest.main()
