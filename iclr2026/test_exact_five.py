import unittest
from itertools import product
import numpy as np
from exact_five_pilot import search_chain
from optuna_frontier import simulate_cascade


class ExactFiveTest(unittest.TestCase):
    def test_against_brute_force(self):
        for seed in range(30):
            rng=np.random.default_rng(seed)
            data={str(m):dict(scores=rng.integers(0,4,8).astype(float),
                             correct=rng.integers(0,2,8).astype(float),
                             costs=rng.integers(0,8,8).astype(float)/8) for m in range(5)}
            seq=list(data)
            cuts=[np.r_[-np.inf,np.unique(data[m]['scores'])[1:],np.inf] for m in seq[:4]]
            candidates=[simulate_cascade(seq,t,data) for t in product(*cuts)]
            for budget in (0.,.5,1.,np.inf):
                expected=np.full(9,np.inf)
                for c,q in candidates:
                    if c<=budget+1e-12:
                        expected[round(q*8)]=min(expected[round(q*8)],c)
                actual,witness,stats=search_chain(data,seq,budget)
                self.assertTrue(stats['complete'])
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
                for q in np.flatnonzero(np.isfinite(actual)):
                    c,quality=simulate_cascade(seq,[cuts[j][r] for j,r in enumerate(witness[q])],data)
                    self.assertAlmostEqual(c,actual[q])
                    self.assertEqual(round(quality*8),q)

    def test_timeout_marks_partial(self):
        data={str(m):dict(scores=np.arange(100,dtype=float),correct=np.ones(100),costs=np.ones(100)) for m in range(5)}
        _,_,stats=search_chain(data,list(data),seconds_limit=1e-12)
        self.assertFalse(stats['complete'])


if __name__=='__main__':
    unittest.main()
