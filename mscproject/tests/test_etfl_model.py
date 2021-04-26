import unittest

import pandas as pd

from mscproject.simulation.etfl_model import ETFLModel


class MyTestCase(unittest.TestCase):
    def test_one_condition(self):
        fba_model = ETFLModel(conditions=pd.DataFrame([['arab__L']]))
        fba_model.solve()

    def test_multiple_conditions(self):
        fba_model = ETFLModel(conditions=pd.read_csv('perturbations.csv'))
        fba_model.solve()

    def test_one_condition_w_sampling(self):
        fba_model = ETFLModel(conditions=pd.DataFrame([['arab__L']]), sampling_n=10000)
        fba_model.solve()

    def test_multiple_conditions_w_sampling(self):
        fba_model = ETFLModel(conditions=pd.read_csv('perturbations.csv'), sampling_n=10000)
        fba_model.solve()


if __name__ == '__main__':
    unittest.main()
