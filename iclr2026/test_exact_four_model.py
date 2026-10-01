import unittest
from itertools import product
import numpy as np
from exact_four_model import search_chain
from optuna_frontier import simulate_cascade


class ExactFourTest(unittest.TestCase):
    def test_brute_force(self):
        for seed in range(30):
            rng=np.random.default_rng(seed)
            n=8
            data={str(m):dict(scores=rng.integers(0,4,n).astype(float),
                             correct=rng.integers(0,2,n).astype(float),
                             costs=rng.integers(0,8,n).astype(float)/10)
                  for m in range(4)}
            seq=list(data)
            cuts=[np.r_[-np.inf,np.unique(data[m]['scores'])[1:],np.inf] for m in seq[:3]]
            candidates=[simulate_cascade(seq,ts,data) for ts in product(*cuts)]
            for budget in (0., .4, .8, np.inf):
                expected=np.full(n+1,np.inf)
                for c,q in candidates:
                    if c<=budget+1e-12:
                        expected[round(q*n)]=min(expected[round(q*n)],c)
                actual,witness,_,_=search_chain(data,seq,budget)
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
                for q in np.flatnonzero(np.isfinite(actual)):
                    ts=[cuts[j][r] for j,r in enumerate(witness[q])]
                    c,quality=simulate_cascade(seq,ts,data)
                    self.assertAlmostEqual(c,actual[q])
                    self.assertEqual(round(quality*n),q)


if __name__=='__main__':
    unittest.main()
