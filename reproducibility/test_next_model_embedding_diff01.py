import unittest
from itertools import combinations, product

import numpy as np

from continuation_benefit_compute import evaluate, fit_linear, predict
from next_model_embedding_diff01 import (
    cutoffs,
    fit_embedding_edges,
    select_three,
)


class EmbeddingDiffTests(unittest.TestCase):
    def data(self, n=12, p=7):
        rng = np.random.default_rng(144)
        return {
            model: {
                "x": rng.normal(size=(n, p)),
                "y": rng.integers(0, 2, n).astype(float),
                "c": rng.uniform(index + 1, index + 1.2, n),
            }
            for index, model in enumerate("abcd")
        }

    def test_joint_fit_matches_separate_ridge_fits(self):
        data = self.data(30, 9)
        models, edges = fit_embedding_edges(data, "embedding_diff01_ridge")
        current = models[0]
        for downstream in models[1:]:
            target = data[downstream]["y"] - data[current]["y"]
            coefficient, intercept = fit_linear(data[current]["x"], target)
            expected = -(data[current]["x"] @ coefficient + intercept)
            np.testing.assert_allclose(
                predict(data[current]["x"], edges[current, downstream]),
                expected,
                rtol=1e-11,
                atol=1e-11,
            )

    def test_next_model_target_ignores_later_model(self):
        data = self.data()
        _, before = fit_embedding_edges(data, "embedding_diff01_ridge")
        data["d"]["y"] = 1 - data["d"]["y"]
        _, after = fit_embedding_edges(data, "embedding_diff01_ridge")
        self.assertEqual(before["a", "b"], after["a", "b"])
        self.assertNotEqual(before["c", "d"], after["c", "d"])

    def test_exact_s1_s2_s3_matches_brute_force(self):
        data = self.data(9, 5)
        budgets = np.linspace(1.2, 11, 31)
        for arm in ("embedding_diff01_ridge",):
            models, edges = fit_embedding_edges(data, arm)
            candidates = []
            for depth in (1, 2, 3):
                for sequence in combinations(models, depth):
                    transitions = list(zip(sequence[:-1], sequence[1:]))
                    options = [
                        cutoffs(predict(data[a]["x"], edges[a, b]))
                        for a, b in transitions
                    ]
                    for thresholds in product(*options):
                        policy = {
                            "models": list(sequence),
                            "stages": [
                                {"predictor": edges[edge], "threshold": threshold}
                                for edge, threshold in zip(transitions, thresholds)
                            ],
                        }
                        cost, quality, _ = evaluate(policy, data)
                        candidates.append((depth, cost, quality))
            frozen = select_three(data, budgets, models, edges)
            for depth in (1, 2, 3):
                for budget, policy in zip(budgets, frozen[str(depth)]):
                    eligible = [
                        (cost, quality)
                        for candidate_depth, cost, quality in candidates
                        if candidate_depth <= depth and cost <= budget + 1e-12
                    ]
                    best_quality = max(quality for cost, quality in eligible)
                    best_cost = min(
                        cost for cost, quality in eligible if quality == best_quality
                    )
                    np.testing.assert_allclose(
                        evaluate(policy, data)[:2],
                        [best_cost, best_quality],
                        atol=1e-12,
                    )


if __name__ == "__main__":
    unittest.main()
