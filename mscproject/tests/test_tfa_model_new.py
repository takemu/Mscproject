import unittest

import pandas as pd

from mscproject.simulation.tfa_model_new import TFAModel


class TestTFAModelBaseline(unittest.TestCase):
    def test_tfa_small_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(conditions=pd.DataFrame([['arab__L']]))
        solution.to_csv('baseline/tfa_small.csv')

    def test_tfa_batch_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/tfa_batch.csv')

    def test_ptfa_small_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(alg='pfba', conditions=pd.DataFrame([['arab__L']]))
        solution.to_csv('baseline/ptfa_small.csv')

    def test_ptfa_batch_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/ptfa_batch.csv')

    def test_ptfa_batch_baseline2(self):
        tfa_model = TFAModel(objective_lb=0)
        solution = tfa_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/ptfa_batch2.csv')


if __name__ == '__main__':
    unittest.main()
