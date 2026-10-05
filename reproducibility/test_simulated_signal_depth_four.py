"""Regression checks for matched AUROC cache extension and report integration."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from exact_heldout_depth import exact_select, evaluate_policy, encode
from manuscript_sources import EXPECTED_MODELS
import simulated_signal_depth_four as experiment
from simulated_signal_depth_four_report import integrate


class SyntheticFourTest(unittest.TestCase):
    def test_extension_replays_preserves_and_resumes(self):
        rng=np.random.default_rng(45)
        models=sorted(EXPECTED_MODELS)
        n=20
        scores=rng.random((n,8))
        correct=rng.integers(0,2,(n,8)).astype(float)
        costs=rng.uniform(.001,.002,(n,8))*(np.arange(8)+1)
        rows=np.arange(n)*2
        with TemporaryDirectory() as directory:
            root=Path(directory); base=root/'base'; out=root/'out'
            dest=base/'mmlu'; dest.mkdir(parents=True)
            inp=dest/'inputs.npz'
            np.savez(inp,rows=rows,models=models,correct=correct,costs=costs,**{'scores_0.9':scores})
            def sliced(idx):
                return {m:dict(scores=scores[idx,j],correct=correct[idx,j],costs=costs[idx,j]) for j,m in enumerate(models)}
            cal=sliced(np.arange(10)); test=sliced(np.arange(10,20))
            budgets=np.linspace(costs[:10].mean(0).min(),costs[:10].mean(0).max(),11)
            frozen,counts=exact_select(cal,budgets)
            manifest=dict(version='synthetic-depth-v1',pool=models,
                input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                cal_idx=rows[:10].tolist(),test_idx=rows[10:].tolist(),budgets=budgets.tolist(),
                frozen=frozen,candidate_counts=counts)
            path=dest/'auroc_0.9'/'split_00.json'; path.parent.mkdir()
            path.write_text(json.dumps(encode(manifest)))
            records=[]
            for i,b in enumerate(budgets):
                row=dict(dataset='mmlu',target_auroc=.9,split=0,budget=b,fraction=i/10)
                for depth in (1,2,3):
                    p=frozen[str(depth)][i]
                    c,q,calls=evaluate_policy(p,test)
                    row.update({f'd{depth}_{k}':v for k,v in dict(cost=c,accuracy=q,mean_calls=calls,
                        selected_depth=len(p['models']),cal_cost=p['cal_cost'],cal_accuracy=p['cal_quality'],
                        overshoot=c-b).items()})
                records.append(row)
            pd.DataFrame(records).to_csv(path.with_suffix('.csv'),index=False)
            before={p:p.read_bytes() for p in (inp,path,path.with_suffix('.csv'))}
            with patch.object(experiment,'BASE',base):
                result=experiment.run('mmlu',.9,0,out)
                self.assertIsInstance(result[-1],float)
                for p,contents in before.items(): self.assertEqual(p.read_bytes(),contents)
                output=out/'mmlu'/'auroc_0.9'/'split_00.csv'
                frame=pd.read_csv(output)
                self.assertTrue((frame.d4_cal_accuracy>=frame.d3_cal_accuracy-1e-12).all())
                m=json.loads(output.with_suffix('.json').read_text())
                for i,p in enumerate(m['frozen']['4']):
                    np.testing.assert_allclose(evaluate_policy(p,test),frame.iloc[i][['d4_cost','d4_accuracy','d4_mean_calls']].to_numpy(float),rtol=0,atol=1e-12)
                self.assertEqual(experiment.run('mmlu',.9,0,out)[-1],'cached')
                path.with_suffix('.csv').write_text(path.with_suffix('.csv').read_text()+'\n')
                with self.assertRaisesRegex(ValueError,'Incompatible output'):
                    experiment.run('mmlu',.9,0,out)

    def test_integrals_use_budget_width_and_percentage_points(self):
        frame=pd.DataFrame(dict(fraction=[0,.2,1],d4_selected_depth=[1,4,4]))
        for d in (1,2,3,4):
            frame[f'd{d}_accuracy']=[.1*d,.1*d+.1,.1*d+.2]
            frame[f'd{d}_cost']=.001*d
            frame[f'd{d}_cal_accuracy']=.1*d
            frame[f'd{d}_mean_calls']=d
            frame[f'd{d}_overshoot']=[0,.001,0]
        r=integrate(frame)
        self.assertAlmostEqual(r['s1_accuracy_pp'],23)
        self.assertAlmostEqual(r['s4_minus_s3_pp'],10)
        self.assertAlmostEqual(r['s4_minus_s2_cost_per_1000'],2)
        self.assertAlmostEqual(r['s4_depth4_fraction'],.9)
        self.assertAlmostEqual(r['s1_overshoot_fraction'],1/3)


if __name__=='__main__': unittest.main()
