"""Price construction, protocol conflicts, and exact frozen replay regression."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import synthetic_cost_depth as sc


class SyntheticCostTests(unittest.TestCase):
    def test_all_structures_and_price_reconstruction(self):
        tin = np.array([[10, 30, 50], [20, 20, 70], [30, 10, 90]])
        tout = np.array([[2, 4, 5], [5, 7, 8], [1, 1, 10]])
        prices = np.array([[1., 2.], [3., 4.], [6., 9.]]) * 1e-6
        obs = sc.recompute_costs(tin, tout, prices)
        means = obs.mean(axis=0)
        for setting in sc.SETTINGS:
            target, u, mult = sc.cost_structure(means, setting)
            new = sc.recompute_costs(tin, tout, prices * mult[:, None])
            np.testing.assert_allclose(new.mean(axis=0), target, rtol=1e-14, atol=0)
            np.testing.assert_allclose(new / obs, np.broadcast_to(mult, new.shape))
            self.assertEqual(new[:, [0, -1]].tobytes(), obs[:, [0, -1]].tobytes())
            self.assertTrue(np.all(np.diff(target) > 0))
            if setting == 'OBS':
                self.assertEqual(new.tobytes(), obs.tobytes())
            if setting == 'U':
                np.testing.assert_allclose(u, np.linspace(0, 1, 3))

    def test_fit_recovers_known_beta(self):
        t = np.linspace(0, 1, 8)
        obs = .001 * 100 ** sc.beta.ppf(t, 2., 4.)
        fit = sc.fit_observed(obs)
        np.testing.assert_allclose([fit['a'], fit['b']], [2., 4.], rtol=1e-5)
        self.assertLess(fit['sse'], 1e-15)

    def test_interior_accuracy_anchor_moves(self):
        costs = np.tile([1., 2., 10.], (4, 1))
        correct = np.tile([0., 1., 0.], (4, 1))
        target, _, mult = sc.cost_structure(costs.mean(axis=0), 'U')
        obs, budgets = sc.schedule(['a', 'b', 'c'], correct, costs, np.arange(4))
        new, other = sc.schedule(['a', 'b', 'c'], correct, costs * mult, np.arange(4))
        self.assertEqual(obs['high'], new['high'])
        self.assertEqual(obs['order'], new['order'])
        self.assertNotEqual(budgets.tobytes(), other.tobytes())

    def test_full_sample_order_does_not_imply_split_order(self):
        costs = np.array([[1., 8.8, 10.], [1., 1.2, 10.]])
        _, _, mult = sc.cost_structure(costs.mean(axis=0), 'S--')
        labels = np.zeros_like(costs)
        obs, _ = sc.schedule(['a', 'b', 'c'], labels, costs, np.array([1]))
        new, _ = sc.schedule(['a', 'b', 'c'], labels, costs * mult, np.array([1]))
        self.assertNotEqual(obs['order'], new['order'])

    def test_manifest_precedes_evaluation_and_obs_matches_existing(self):
        models = ['a', 'b', 'c']
        raw = {}
        for j, m in enumerate(models):
            raw[m] = pd.DataFrame(dict(prompt=['x']*8, logprob=[[0.]*(j+1)]*8,
                correct=np.array([0, 1, 0, 1, 1, 0, 1, 0]) if j < 2 else np.ones(8),
                mean_token_negentropy=np.arange(8, dtype=float)))
        prices = {m: (float(j+1)*1e-6, float(j+1)*1e-6) for j, m in enumerate(models)}
        costs = {m: np.full(8, (len(sc._enc.encode('x'))+j+1)*prices[m][0]) for j, m in enumerate(models)}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(sc, 'prepare', return_value=(raw, costs, np.arange(8), {}, 'fixture')), \
                 patch.object(sc, 'EXPECTED_MODELS', set(models)), patch.object(sc, 'PRICE_PER_TOKEN', prices), \
                 patch.object(sc, 'sha', return_value='fixture-hash'):
                meta = sc.prepare_dataset('fixture', out, 1)
                original = sc.evaluate_policy
                calls = []
                def checked(policy, data):
                    if len(data['a']['scores']) == 4 and (out/'fixture/OBS/split_00.json').exists():
                        frozen = json.loads((out/'fixture/OBS/split_00.json').read_text())
                        self.assertIn('prices', frozen)
                        self.assertIn('frozen', frozen)
                        calls.append(True)
                    return original(policy, data)
                with patch.object(sc, 'evaluate_policy', side_effect=checked):
                    sc.run_cell('fixture', 'OBS', 'split_00', out, 'setting-specific')
                self.assertTrue(calls)
                result = pd.read_csv(out/'fixture/OBS/split_00.csv')
                cal_idx, test_idx = sc.make_split(8, raw['a'].correct.to_numpy(float), sc.SEED)
                cal = {m: dict(scores=raw[m].mean_token_negentropy.to_numpy()[cal_idx],
                              correct=raw[m].correct.to_numpy(float)[cal_idx], costs=costs[m][cal_idx]) for m in models}
                frozen, _ = sc.exact_select(cal, result.budget.to_numpy())
                np.testing.assert_allclose(result.d3_cal_accuracy, [p['cal_quality'] for p in frozen['3']])
                sc.run_cell('fixture', 'OBS', 'split_00', out, 'setting-specific')


if __name__ == '__main__':
    unittest.main()
