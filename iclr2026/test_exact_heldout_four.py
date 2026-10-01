import unittest
from itertools import combinations, product
import numpy as np
from exact_heldout_depth import exact_select, evaluate_policy, encode
from exact_heldout_four import select_four
from optuna_frontier import simulate_cascade


class HeldoutFourTest(unittest.TestCase):
    def test_nested_selection_against_brute_force(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            data = {str(m): dict(scores=rng.integers(0, 3, 7).astype(float),
                                 correct=rng.integers(0, 2, 7).astype(float),
                                 costs=rng.integers(0, 8, 7).astype(float)/8)
                    for m in range(5)}
            budgets = np.linspace(0, 3, 41)
            old, _ = exact_select(data, budgets)
            frozen, _ = select_four(data, budgets, encode(old['3']))
            models = sorted(data, key=lambda m: (data[m]['costs'].mean(), m))
            candidates = []
            for depth in range(1, 5):
                for seq in combinations(models, depth):
                    cuts = [np.r_[-np.inf, np.unique(data[m]['scores'])[1:], np.inf]
                            for m in seq[:-1]]
                    candidates.extend(simulate_cascade(seq, ts, data) for ts in product(*cuts))
            for budget, policy in zip(budgets, frozen):
                feasible = [(c, q) for c, q in candidates if c <= budget+1e-12]
                if not feasible:
                    self.assertIsNone(policy)
                    continue
                c, q = min(feasible, key=lambda p: (-p[1], p[0]))
                self.assertAlmostEqual(policy['cal_cost'], c)
                self.assertAlmostEqual(policy['cal_quality'], q)
                pc, pq, _ = evaluate_policy(encode(policy), data)
                self.assertAlmostEqual(pc, c)
                self.assertAlmostEqual(pq, q)

    def test_shallower_tie_and_unseen_score_endpoints(self):
        data = {str(m): dict(scores=np.ones(4), correct=np.ones(4), costs=np.zeros(4))
                for m in range(4)}
        old, _ = exact_select(data, [0])
        selected, _ = select_four(data, [0], encode(old['3']))
        self.assertEqual(len(selected[0]['models']), 1)
        policy = dict(models=list(data), thresholds=['always', 'never', 'always'])
        test = {m: dict(d, scores=np.full(4, 100), costs=np.ones(4)) for m, d in data.items()}
        self.assertEqual(evaluate_policy(policy, test), (2., 1., 2.))


if __name__ == '__main__':
    unittest.main()
