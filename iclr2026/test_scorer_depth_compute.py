import unittest
import numpy as np
import pandas as pd
from scorer_depth_compute import score_models, BASE_SCORERS, RESPONSE_SCORER


class ScorerTrainingTest(unittest.TestCase):
    def test_test_labels_cannot_change_scores(self):
        rng = np.random.default_rng(9)
        raw = {'m': pd.DataFrame(rng.uniform(size=(20, 5)), columns=BASE_SCORERS)}
        raw['m']['correct'] = np.arange(20) % 2
        embeddings = {'m': rng.normal(size=(20, 4))}
        cal = np.arange(10)
        changed = {'m': raw['m'].copy()}
        changed['m'].loc[10:, 'correct'] = 1-changed['m'].loc[10:, 'correct']
        for scorer in BASE_SCORERS + ['logreg_ensemble', RESPONSE_SCORER]:
            a, fitted_a = score_models(raw, embeddings, cal, scorer)
            b, fitted_b = score_models(changed, embeddings, cal, scorer)
            np.testing.assert_allclose(a['m'], b['m'], rtol=0, atol=1e-14)
            self.assertEqual(fitted_a, fitted_b)

    def test_single_class_fallback(self):
        raw = {'m': pd.DataFrame({s: [.2, .4, .6] for s in BASE_SCORERS})}
        raw['m']['correct'] = [1, 1, 0]
        for scorer in ['logreg_ensemble', RESPONSE_SCORER]:
            scores, fitted = score_models(raw, {}, np.array([0, 1]), scorer)
            np.testing.assert_array_equal(scores['m'], [.2, .4, .6])
            self.assertEqual(fitted['m']['fallback'], 'mean_token_negentropy')


if __name__ == '__main__':
    unittest.main()
