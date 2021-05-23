import unittest

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from mscproject.simulation.fba_model import FBAModel, growth_reaction_id


class TestFBAModelBaseline(unittest.TestCase):
    def setUp(self):
        self.condition = 'arab__L'
        self.obj = growth_reaction_id

    def test_fba_baseline(self):
        fba_model = FBAModel(conditions=pd.DataFrame([[self.condition]]))
        solution = fba_model.solve()
        solution.to_csv('fba_small_baseline.csv')

    def test_fba_batch_baseline(self):
        fba_model = FBAModel(conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution = fba_model.solve()
        solution.to_csv('fba_batch_baseline.csv')

    def test_pfba_baseline(self):
        fba_model = FBAModel(conditions=pd.DataFrame([[self.condition]]), pfba=True)
        solution = fba_model.solve()
        solution.to_csv('pfba_small_baseline.csv')

    def test_pfba_batch_baseline(self):
        fba_model = FBAModel(conditions=pd.read_csv('../simulation/data/perturbations.csv'), pfba=True)
        solution = fba_model.solve()
        solution.to_csv('pfba_batch_baseline.csv')

    def test_sampling(self):
        sampling_model = FBAModel(conditions=pd.DataFrame([[self.condition]]), sampling_n=10000)
        solution = sampling_model.solve()
        solution.to_csv('sampling_baseline.csv')


class TestFBAModel(unittest.TestCase):
    def setUp(self):
        self.condition = 'arab__L'
        self.obj = growth_reaction_id

    def test_fba(self):
        fba_model = FBAModel(conditions=pd.DataFrame([[self.condition]]))
        solution = fba_model.solve()
        baseline = pd.read_csv('fba_small_baseline.csv', index_col=0)
        assert_frame_equal(solution, baseline)

    def test_fba_batch(self):
        fba_model = FBAModel(conditions=pd.read_csv('../simulation/data/perturbations.csv'))
        solution = fba_model.solve()
        baseline = pd.read_csv('fba_batch_baseline.csv', index_col=0)
        solution.columns = baseline.columns
        assert_frame_equal(solution, baseline)

    def test_pfba(self):
        fba_model = FBAModel(conditions=pd.DataFrame([[self.condition]]), pfba=True)
        solution = fba_model.solve()
        baseline = pd.read_csv('pfba_small_baseline.csv', index_col=0)
        assert_frame_equal(solution, baseline)

    def test_pfba_batch(self):
        fba_model = FBAModel(conditions=pd.read_csv('../simulation/data/perturbations.csv'), pfba=True)
        solution = fba_model.solve()
        baseline = pd.read_csv('pfba_batch_baseline.csv', index_col=0)
        solution.columns = baseline.columns
        assert_frame_equal(solution, baseline)

    def test_duplicate_yangs_data(self):
        yangs = pd.read_csv('yangs_data.csv', index_col=0)
        count = 0
        print('control', yangs.loc[self.obj, 'control'], yangs.loc['net_flux', 'control'])
        results = yangs['control']
        for column in yangs.iloc[:, 1:]:
            if np.isclose(yangs.loc[self.obj, column], yangs.loc[self.obj, 'control'], atol=1.e-3):
                if np.isclose(yangs.loc['net_flux', column], yangs.loc['net_flux', 'control'], atol=1.e-3):
                    if column[-2] != '.':
                        print(column, yangs.loc[self.obj, column], yangs.loc['net_flux', column])
                        results = pd.concat([results, yangs[column]], axis=1)
                        count += 1
        print('Total:', count)
        results = results[(results.T != 0).any()]
        results.to_csv('../../output/dup_yangs_data.csv', float_format='%.6f')

    def test_duplicate_pfba_data(self):
        pfba_data = pd.read_csv('pfba_batch_baseline.csv', index_col=0)
        count = 0
        print('control', pfba_data.loc[self.obj, 'control'], pfba_data.loc['net_flux', 'control'])
        results = pfba_data['control']
        for column in pfba_data.iloc[:, 1:]:
            if np.isclose(pfba_data.loc[self.obj, column], pfba_data.loc[self.obj, 'control'], atol=1.e-3):
                if np.isclose(pfba_data.loc['net_flux', column], pfba_data.loc['net_flux', 'control'], atol=1.e-3):
                    if column[-2] != '.':
                        print(column, pfba_data.loc[self.obj, column], pfba_data.loc['net_flux', column])
                        results = pd.concat([results, pfba_data[column]], axis=1)
                        count += 1
        print('Total:', count)
        results = results[(results.T != 0).any()]
        results.to_csv('../../output/dup_pfba_data.csv', float_format='%.6f')

    def test_comparison(self):
        condition = 'ala__D'
        yangs_data = pd.read_csv('yangs_data.csv', index_col=0)
        yangs_data = yangs_data[['control', condition]]
        pfba_data = FBAModel(conditions=pd.DataFrame([[condition]]), pfba=True).solve()

        x = yangs_data[condition].to_frame()
        x = x[(x.T != 0).any()]
        y = pfba_data[condition].to_frame()
        y = y[(y.T != 0).any()]
        compare_df = pd.merge(x, y, left_on=x.index, right_on=y.index, how='outer').fillna(0)
        compare_df = compare_df.set_index('key_0').sort_index()
        compare_df.to_csv('../../output/compare_df.csv', float_format='%.6f')

    def test_similarity(self):
        yangs_data = pd.read_csv('yangs_data.csv', index_col=0)
        pfba_data = pd.read_csv('pfba_batch_baseline.csv', index_col=0)
        diff_count = 0
        equal_count = 0
        total_count = 0
        total_similarity = 0
        diff_threshold = 1e-3
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
        print('Avg. similarity:', total_similarity / total_count)


if __name__ == '__main__':
    unittest.main()
