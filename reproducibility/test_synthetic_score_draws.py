"""Regression checks for draw independence and the two variability summaries."""
import unittest
import numpy as np
import pandas as pd
from synthetic_score_draws import draw_seed, summarize, validate_match
from simulate_correlated_confidence import orthogonal_noise, calibrate


class DrawTests(unittest.TestCase):
    def test_separate_draw_and_split_variability(self):
        rows = [dict(dataset='toy', target_auroc=.8, draw=d, split=s,
                     s3_minus_s2_pp=value)
                for d, values in enumerate(([0., 2.], [10., 12.]))
                for s, value in enumerate(values)]
        per_draw, across = summarize(pd.DataFrame(rows), 2, 2)
        np.testing.assert_allclose(per_draw.mean_gain_pp, [1, 11])
        np.testing.assert_allclose(per_draw.split_sd_pp, np.sqrt(2))
        self.assertAlmostEqual(across.iloc[0].mean_gain_pp, 6)
        self.assertAlmostEqual(across.iloc[0].across_draw_sd_pp, np.sqrt(50))
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame(rows[:-1]), 2, 2)
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame(rows + rows[:1]), 2, 2)
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame(rows[:2]), 2, 2)

    def test_independent_reproducible_constrained_draws(self):
        outcomes = np.random.default_rng(42).integers(0, 2, size=(600, 3)).astype(float)
        scores = []
        for draw in (0, 1, 0):
            rng = np.random.default_rng(np.random.SeedSequence(draw_seed('mmlu', draw, 123)))
            y, noise, corr = orthogonal_noise(outcomes, rng)
            result, aucs, achieved = calibrate(.8, y, noise, corr, np.eye(3), outcomes)
            np.testing.assert_allclose(aucs, .8, atol=.001)
            np.testing.assert_allclose(achieved, np.eye(3), atol=1e-10)
            scores.append(result)
        np.testing.assert_array_equal(scores[0], scores[2])
        self.assertFalse(np.array_equal(scores[0], scores[1]))
        self.assertNotEqual(draw_seed('mmlu', 0, 123), draw_seed('triviaqa', 0, 123))

    def test_matching_rejects_changed_rows_or_budgets(self):
        ref = dict(dataset='toy', target_auroc=.8, split=0, seed=42, pool=['a', 'b'],
                   cal_idx=[0, 2], test_idx=[1, 3], budgets=[1, 2], low='a', high='b')
        validate_match(ref, ref.copy())
        for key, changed in [('cal_idx', [0, 3]), ('budgets', [1, 3]), ('pool', ['a'])]:
            with self.assertRaises(ValueError):
                validate_match(dict(ref, **{key: changed}), ref)


if __name__ == '__main__':
    unittest.main()
