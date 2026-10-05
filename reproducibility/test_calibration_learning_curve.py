import unittest
import numpy as np
import pandas as pd
from calibration_learning_curve import stratified_order, aggregate


class LearningCurveTest(unittest.TestCase):
    def test_nested_stratification(self):
        labels = np.r_[np.zeros(73), np.ones(27)]
        indices = np.arange(100)
        order = stratified_order(indices, labels, 42)
        np.testing.assert_array_equal(np.sort(order), indices)
        np.testing.assert_array_equal(order, stratified_order(indices, labels, 42))
        for n in range(1, 101):
            self.assertLessEqual(abs(labels[order[:n]].sum() - .27*n), 1)
        for a, b in zip((10, 25, 50, 75), (25, 50, 75, 100)):
            self.assertTrue(set(order[:a]) <= set(order[:b]))

    def test_common_mask_and_paired_mean(self):
        rows = []
        for size in (.1, 1.):
            for j in range(3):
                row = dict(dataset='toy', split=0, fraction=size, n_cal=int(size*100),
                    n_test=100, budget_index=j, budget=2.)
                for d in (2, 3):
                    row.update({f'd{d}_feasible': not(size == .1 and j == 0 and d == 2),
                        f'd{d}_accuracy': 1. if j == 0 else d*.1,
                        f'd{d}_cal_accuracy': d*.1, f'd{d}_cost': float(d),
                        f'd{d}_overshoot': float(d-2),
                        f'd{d}_relative_overshoot': (d-2)/2., f'd{d}_calls': float(d)})
                rows.append(row)
        result = aggregate(pd.DataFrame(rows))
        np.testing.assert_allclose(result.test_gain_pp, 10.)
        np.testing.assert_allclose(result.coverage, 2/3)
        np.testing.assert_allclose(result.cost_gain_pct, 50.)
        self.assertEqual(result.common_budget_count.tolist(), [2, 2])
        self.assertAlmostEqual(result.d2_infeasible_pct.iloc[0], 100/3)
        self.assertEqual(result.d2_infeasible_pct.iloc[1], 0.)


if __name__ == '__main__':
    unittest.main()
