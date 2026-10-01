"""Regression checks for mixing seven- and eight-model manuscript results."""
import copy
import json
import unittest
import numpy as np
from manuscript_sources import ROOT, result_path, validate_learning, validate_exact

class ManuscriptSourceTests(unittest.TestCase):
    def setUp(self):
        self.exact_path = result_path('livecodebench', 'exact_heldout_five')
        self.exact = json.loads((self.exact_path/'livecodebench/split_00.json').read_text())
        self.learning = json.loads((result_path('livecodebench', 'calibration_learning_curve')/'livecodebench/split_00.json').read_text())

    def test_current_sources_agree(self):
        validate_learning(self.learning, self.exact, 'livecodebench', 0)
        with np.load(self.exact_path/'inputs/livecodebench.npz') as z:
            validate_exact(self.exact, z, 'livecodebench', 0)

    def test_historical_seven_model_learning_cache_is_rejected(self):
        old = json.loads((ROOT/'results/calibration_learning_curve/livecodebench/split_00.json').read_text())
        with self.assertRaisesRegex(ValueError, 'eight-model'):
            validate_learning(old, self.exact, 'livecodebench', 0)

    def test_historical_seven_model_exact_cache_is_rejected(self):
        old = json.loads((ROOT/'results/exact_heldout_five/livecodebench/split_00.json').read_text())
        with np.load(ROOT/'results/exact_heldout_five/inputs/livecodebench.npz') as z:
            with self.assertRaisesRegex(ValueError, 'eight-model'):
                validate_exact(old, z, 'livecodebench', 0)

    def test_mismatched_fingerprint_is_rejected(self):
        bad = copy.deepcopy(self.learning)
        bad['data_sha256'] = 'different-run'
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_learning(bad, self.exact, 'livecodebench', 0)

    def test_mismatched_queries_are_rejected(self):
        bad = copy.deepcopy(self.learning)
        bad['test_idx'] = bad['test_idx'][::-1]
        with self.assertRaisesRegex(ValueError, 'query identities'):
            validate_learning(bad, self.exact, 'livecodebench', 0)

    def test_mismatched_budgets_are_rejected(self):
        bad = copy.deepcopy(self.learning)
        bad['budgets'][100] *= 2
        with self.assertRaises(AssertionError):
            validate_learning(bad, self.exact, 'livecodebench', 0)

if __name__ == '__main__':
    unittest.main()
