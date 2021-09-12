import unittest

import numpy as np
import pandas as pd

from mscproject.simulation.tfa_model_new import TFAModel


class TestTFAModelBaseline(unittest.TestCase):
    def test_tfa_small_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/tfa_small.csv')

    def test_tfa_batch_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/tfa_batch.csv')

    def test_ptfa_small_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/ptfa_small.csv')

    def test_ptfa_batch_baseline(self):
        tfa_model = TFAModel()
        solution = tfa_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/ptfa_batch.csv')

    def test_ptfa_batch_baseline2(self):
        tfa_model = TFAModel(min_biomass=0)
        solution = tfa_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/ptfa_batch2.csv')


class TestTFAModel(unittest.TestCase):
    def setUp(self):
        self.biomass_reaction = 'BIOMASS_Ec_iJO1366_core_53p95M'

    def test_ptfa_duplicates(self):
        ptfa_batch = pd.read_csv('baseline/ptfa_batch.csv', index_col=0)
        print('condition', 'biomass', 'net_flux')
        print('control', ptfa_batch.loc[self.biomass_reaction, 'control'], ptfa_batch.loc['net_flux', 'control'])
        duplicates = ptfa_batch['control']
        count = 0
        for col_name, column in ptfa_batch.iloc[:, 1:].iteritems():
            if np.isclose(column, ptfa_batch['control'], atol=1.e-1).all():
                print(col_name, column[self.biomass_reaction], column['net_flux'])
                duplicates = pd.concat([duplicates, ptfa_batch[col_name]], axis=1)
                count += 1
        print('Total duplicate columns:', count)
        duplicates = duplicates[(duplicates.T != 0).any()]
        duplicates.to_csv('output/ptfa_duplicates.csv')

    def test_ptfa_duplicates2(self):
        ptfa_batch2 = pd.read_csv('baseline/ptfa_batch2.csv', index_col=0)
        print('condition', 'biomass', 'net_flux')
        print('control', ptfa_batch2.loc[self.biomass_reaction, 'control'], ptfa_batch2.loc['net_flux', 'control'])
        duplicates = ptfa_batch2['control']
        count = 0
        for col_name, column in ptfa_batch2.iloc[:, 1:].iteritems():
            if np.isclose(column, ptfa_batch2['control'], atol=1.e-1).all():
                print(col_name, column[self.biomass_reaction], column['net_flux'])
                duplicates = pd.concat([duplicates, ptfa_batch2[col_name]], axis=1)
                count += 1
        print('Total duplicate columns:', count)
        duplicates = duplicates[(duplicates.T != 0).any()]
        duplicates.to_csv('output/ptfa_duplicates2.csv')


if __name__ == '__main__':
    unittest.main()
