import unittest
import numpy as np

from shared_difficulty_depth import (signal_columns, calibrate_column,
                                     stage_diagnostics, synthetic)


class SharedDifficultyTests(unittest.TestCase):
    def test_difficulty_excludes_own_label(self):
        y = np.random.default_rng(3).integers(0, 2, (300, 8)).astype(float)
        _, d = signal_columns(y)
        changed = y.copy()
        changed[:, 2] = 1-changed[:, 2]
        _, other = signal_columns(changed)
        np.testing.assert_array_equal(d[:, 2], other[:, 2])
        self.assertGreater(np.max(np.abs(d[:, 1]-other[:, 1])), .1)

    def test_calibration_and_infeasibility(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 1000).astype(float)
        noise = rng.normal(size=1000)
        score, _, _ = calibrate_column(y, noise, y, .9)
        self.assertLessEqual(abs(synthetic.empirical_auc(score, y)-.9), .001)
        with self.assertRaises(ValueError):
            calibrate_column(-y, noise, y, .9)

    def test_continuation_and_reach(self):
        rng = np.random.default_rng(8)
        scores = rng.random((200, 3))
        labels = rng.integers(0, 2, (200, 3)).astype(float)
        policy = dict(models=['a', 'b', 'c'], thresholds=[.4, .6])
        rows = stage_diagnostics(policy, scores, labels, ['a', 'b', 'c'])
        downstream = np.where(scores[:, 1] >= .6, labels[:, 1], labels[:, 2])
        self.assertAlmostEqual(rows[0]['A_downstream'], synthetic.empirical_auc(scores[:, 0], downstream))
        reach = scores[:, 0] < .4
        self.assertEqual(rows[1]['reached'], reach.sum())
        self.assertAlmostEqual(rows[1]['A_downstream'], synthetic.empirical_auc(scores[reach, 1], labels[reach, 2]))
        # Independently integrate every threshold operating point.
        order = np.argsort(scores[:, 0])
        delta = downstream[order]-labels[order, 0]
        t = np.arange(201)/200
        gain = np.r_[0, np.cumsum(delta)]/200-t*delta.mean()
        self.assertAlmostEqual(np.trapz(gain, t), rows[0]['eq10_rhs'])

    def test_ties_degenerate_labels_and_empty_reach(self):
        scores = np.array([[0., .2, 0.], [0., .7, 0.], [1., .5, 0.], [1., .1, 0.]])
        labels = np.array([[0., 1., 0.], [1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
        rows = stage_diagnostics(dict(models=['a', 'b', 'c'], thresholds=['never', .5]),
                                 scores, labels, ['a', 'b', 'c'])
        self.assertEqual(rows[1]['reached'], 0)
        self.assertEqual(rows[1]['eq10_rhs'], 0)
        self.assertTrue(np.isnan(rows[1]['A_i']))


if __name__ == '__main__':
    unittest.main()
