import unittest

import pandas as pd

from mscproject.simulation.etfl_model_new import ETFLModel


class TestETFLModelBaseline(unittest.TestCase):
    def test_etfl_small_baseline(self):
        etfl_model = ETFLModel()
        solution = etfl_model.solve(conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/etfl_small.csv')

    def test_etfl_batch_baseline(self):
        etfl_model = ETFLModel()
        solution = etfl_model.solve(conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/etfl_batch.csv')

    def test_etfl_small_baseline(self):
        etfl_model = ETFLModel()
        solution = etfl_model.solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/petfl_small.csv')

    def test_etfl_batch_baseline(self):
        etfl_model = ETFLModel()
        solution = etfl_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/petfl_batch.csv')

    def test_etfl_batch_baseline2(self):
        etfl_model = ETFLModel(min_biomass=0)
        solution = etfl_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/petfl_batch2.csv')


if __name__ == '__main__':
    unittest.main()
