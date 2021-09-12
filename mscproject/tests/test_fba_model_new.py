import unittest

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from mscproject.simulation.fba_model_new import FBAModel


class TestFBAModelBaseline(unittest.TestCase):
    def test_fba_small_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/fba_small.csv')

    def test_fba_batch_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/fba_batch.csv')

    def test_pfba_small_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/pfba_small.csv')

    def test_pfba_batch_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution.to_csv('baseline/pfba_batch.csv')

    def test_small_sampling(self):
        sampling_model = FBAModel()
        solution = sampling_model.solve(conditions=pd.read_csv('perturbations_small.csv'), sampling_n=10000)
        solution.to_csv('baseline/sampling_small.csv')


class TestFBAModel(unittest.TestCase):
    def setUp(self):
        self.biomass_reaction = 'BIOMASS_Ec_iJO1366_core_53p95M'

    def test_fba_small(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('perturbations_small.csv'))
        baseline = pd.read_csv('baseline/fba_small.csv', index_col=0)
        assert_frame_equal(solution, baseline)

    def test_fba_batch(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        baseline = pd.read_csv('baseline/fba_batch.csv', index_col=0)
        solution.columns = baseline.columns
        assert_frame_equal(solution, baseline)

    def test_pfba_small(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv'))
        baseline = pd.read_csv('baseline/pfba_small.csv', index_col=0)
        assert_frame_equal(solution, baseline)

    def test_pfba_batch(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        baseline = pd.read_csv('baseline/pfba_batch.csv', index_col=0)
        solution.columns = baseline.columns
        assert_frame_equal(solution, baseline)

    def test_yangs_duplicates(self):
        yangs_data = pd.read_csv('baseline/yangs_data.csv', index_col=0)
        print('condition', 'biomass', 'net_flux')
        print('control', yangs_data.loc[self.biomass_reaction, 'control'], yangs_data.loc['net_flux', 'control'])
        duplicates = yangs_data['control']
        count = 0
        for col_name, column in yangs_data.iloc[:, 1:].iteritems():
            if np.isclose(column, yangs_data['control'], atol=1.e-1).all():
                print(col_name, column[self.biomass_reaction], column['net_flux'])
                duplicates = pd.concat([duplicates, yangs_data[col_name]], axis=1)
                count += 1
        print('Total duplicate columns:', count)
        duplicates = duplicates[(duplicates.T != 0).any()]
        duplicates.to_csv('output/yangs_duplicates.csv')

    def test_pfba_duplicates(self):
        pfba_batch = pd.read_csv('baseline/pfba_batch.csv', index_col=0)
        print('condition', 'biomass', 'net_flux')
        print('control', pfba_batch.loc[self.biomass_reaction, 'control'], pfba_batch.loc['net_flux', 'control'])
        duplicates = pfba_batch['control']
        count = 0
        for col_name, column in pfba_batch.iloc[:, 1:].iteritems():
            if np.isclose(column, pfba_batch['control'], atol=1.e-1).all():
                print(col_name, column[self.biomass_reaction], column['net_flux'])
                duplicates = pd.concat([duplicates, pfba_batch[col_name]], axis=1)
                count += 1
        print('Total duplicate columns:', count)
        duplicates = duplicates[(duplicates.T != 0).any()]
        duplicates.to_csv('output/pfba_duplicates.csv')

    def test_comparison(self, condition='ala__D'):
        yangs_data = pd.read_csv('baseline/yangs_data.csv', index_col=0)
        yangs_data = yangs_data[['control', condition]]
        pfba_data = FBAModel().solve(alg='pfba', conditions=pd.DataFrame([[condition]]))

        x = yangs_data[condition].to_frame()
        x = x[(x.T != 0).any()]
        y = pfba_data[condition].to_frame()
        y = y[(y.T != 0).any()]
        compare_df = pd.merge(x, y, left_on=x.index, right_on=y.index, how='outer').fillna(0)
        compare_df = compare_df.set_index('key_0').sort_index()
        compare_df.to_csv('output/compare_df.csv', float_format='%.2f')

    def test_similarity(self):
        yangs_data = pd.read_csv('baseline/yangs_data.csv', index_col=0)
        pfba_data = pd.read_csv('baseline/pfba_batch.csv', index_col=0)
        diff_count = 0
        equal_count = 0
        total_count = 0
        total_similarity = 0
        diff_threshold = 1e-2
        for column in pfba_data.iloc[:, 0:]:
            if column == 'galt' or column == 'pser__L' or column[-2] == '.':
                continue
            x = yangs_data[column].to_frame()
            x = x[(x.T != 0).any()]
            y = pfba_data[column].to_frame()
            y = y[(y.T != 0).any()]
            compare_df = pd.merge(x, y, left_on=x.index, right_on=y.index, how='outer').fillna(0)
            compare_df = compare_df.set_index('key_0').sort_index()
            diff = (compare_df[column + '_x'] - compare_df[column + '_y']).abs()
            similarity = diff[diff < diff_threshold].count() / compare_df.shape[0]
            total_similarity += similarity
            if similarity < 0.9:
                print(column, similarity, diff.mean())
                diff_count += 1
            if similarity >= 0.9:
                equal_count += 1
            total_count += 1
        print('Total:', total_count, diff_count, equal_count)
        print(f'Avg. similarity:{total_similarity / total_count * 100:2.2f}%')

    def test_two_rounds_solve(self):
        fba_model = FBAModel()
        fba_model.model.reactions.get_by_id(self.biomass_reaction).bounds = 1, 1
        fba_model.solve(conditions=pd.read_csv('perturbations_small.csv')).to_csv('output/two_round.csv')


if __name__ == '__main__':
    unittest.main()
