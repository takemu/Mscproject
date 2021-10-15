import unittest

import numpy as np
import pandas as pd

from mscproject.simulation.fba_model import FBAModel


class TestFBAModelBaseline(unittest.TestCase):
    def test_fba_small_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/fba_small.csv')

    def test_fba_batch_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('../mscproject/simulation/data/perturbations.csv'))
        solution.to_csv('baseline/fba_batch.csv')

    def test_pfba_small_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv'))
        solution.to_csv('baseline/pfba_small.csv')

    def test_pfba_batch_baseline(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv(
            '../mscproject/simulation/data/perturbations.csv'))
        solution.to_csv('baseline/pfba_batch.csv')

    def test_pfba_batch_iml1515_baseline(self):
        fba_model = FBAModel(model_code='ecoli:iML1515')
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv(
            '../mscproject/simulation/data/perturbations.csv'))
        solution.to_csv('baseline/pfba_batch_iml1515.csv')

    def test_small_sampling(self):
        sampling_model = FBAModel()
        solution = sampling_model.solve(conditions=pd.read_csv('perturbations_small.csv'), sampling_n=10000)
        solution.to_csv('baseline/sampling_small.csv')


class TestFBAModel(unittest.TestCase):
    def test_fba_small(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('perturbations_small.csv'))
        baseline = pd.read_csv('baseline/fba_small.csv', index_col=0)
        self.assertTrue(solution.equals(baseline))

    def test_fba_batch(self):
        fba_model = FBAModel()
        solution = fba_model.solve(conditions=pd.read_csv('../mscproject/simulation/data/perturbations.csv'))
        baseline = pd.read_csv('baseline/fba_batch.csv', index_col=0)
        solution.columns = baseline.columns
        self.assertTrue(solution.equals(baseline))

    def test_pfba_small(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv'))
        baseline = pd.read_csv('baseline/pfba_small.csv', index_col=0)
        self.assertTrue(solution.equals(baseline))

    def test_pfba_batch(self):
        fba_model = FBAModel()
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv(
            '../mscproject/simulation/data/perturbations.csv'))
        baseline = pd.read_csv('baseline/pfba_batch.csv', index_col=0)
        solution.columns = baseline.columns
        self.assertTrue(solution.equals(baseline))

    def test_pfba_batch_iml1515(self):
        fba_model = FBAModel(model_code='ecoli:iML1515')
        solution = fba_model.solve(alg='pfba', conditions=pd.read_csv(
            '../mscproject/simulation/data/perturbations.csv'))
        baseline = pd.read_csv('baseline/pfba_batch_iml1515.csv', index_col=0)
        solution.columns = baseline.columns
        self.assertTrue(solution.equals(baseline))

    # def test_pfba_duplicates(self):
    #     tasks = ['yangs_data', 'pfba_batch', 'pfba_batch_iml1515']
    #     for task in tasks:
    #         batch_data = pd.read_csv(f'baseline/{task}.csv', index_col=0)
    #         print(f'<< {task} >>')
    #         duplicates = batch_data['control']
    #         count = 0
    #         for col_name, column in batch_data.iloc[:, 1:].iteritems():
    #             if np.isclose(column, batch_data['control'], atol=1.e-1).all():
    #                 duplicates = pd.concat([duplicates, batch_data[col_name]], axis=1)
    #                 count += 1
    #         print('Duplicate columns to \"control\":')
    #         print(', '.join(list(duplicates.iloc[:, 1:].columns)))
    #         print('Total:', count)
    #         duplicates = duplicates[(duplicates.T != 0).any()]
    #         duplicates.to_csv(f'output/duplicates_{task}.csv')

    def test_pfba_duplicates(self):
        tasks = ['yangs_data', 'pfba_batch', 'pfba_batch_iml1515']
        for task in tasks:
            batch_data = pd.read_csv(f'baseline/{task}.csv', index_col=0)
            batch_data = batch_data.round(decimals=1)
            print(f'<< {task} >>')
            duplicate_set = set()
            duplicates = []
            pivot_cols_w_dup = []
            for i in range(batch_data.shape[1]):
                pivot_col = batch_data.iloc[:, i]
                col_duplicates = []
                for j in range(i + 1, batch_data.shape[1]):
                    col = batch_data.iloc[:, j]
                    if np.isclose(pivot_col, col, atol=1.e-1).all():
                        col_name = batch_data.columns.values[j]
                        if col_name not in duplicate_set:
                            duplicate_set.add(batch_data.columns.values[j])
                            col_duplicates.append(col_name)
                if len(col_duplicates) > 0:
                    pivot_cols_w_dup.append(batch_data.columns.values[i])
                    duplicates.append(col_duplicates)
            duplicate_df = pd.DataFrame(duplicates).T.set_axis(pivot_cols_w_dup, axis=1)
            duplicate_df.index += 1
            duplicate_df.to_csv(f'output/duplicates_{task}.csv')
            print('Total duplicates:', len(duplicate_set))
            self.assertTrue(len(duplicate_set) <= 70)

    def test_align_to_yangs(self):
        yangs_data = pd.read_csv('baseline/yangs_data.csv', index_col=0).round(decimals=2)
        pfba_data = FBAModel().solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv'))
        for _, condition in pd.read_csv('perturbations_small.csv').iterrows():
            condition = condition.dropna()
            c_name = '-'.join(condition)
            x = yangs_data[c_name].to_frame()
            x = x[(x.T != 0).any()]
            y = pfba_data[c_name].to_frame()
            y = y[(y.T != 0).any()]
            align_df = pd.merge(x, y, left_on=x.index, right_on=y.index, how='outer',
                                suffixes=('_yangs', '_ours')).fillna(0)
            align_df = align_df.set_index('key_0').sort_index()
            align_df.to_csv(f'output/align_{c_name}.csv')

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
            align_df = pd.merge(x, y, left_on=x.index, right_on=y.index, how='outer').fillna(0)
            align_df = align_df.set_index('key_0').sort_index()
            diff = (align_df[column + '_x'] - align_df[column + '_y']).abs()
            similarity = diff[diff < diff_threshold].count() / align_df.shape[0]
            total_similarity += similarity
            if similarity < 0.9:
                print(column, similarity, diff.mean())
                diff_count += 1
            if similarity >= 0.9:
                equal_count += 1
            total_count += 1
        print('Total:', total_count, diff_count, equal_count)
        print(f'Avg. similarity:{total_similarity / total_count * 100:2.2f}%')
        self.assertTrue(total_similarity / total_count > 0.9)

    def test_two_rounds_solve(self):
        fba_model = FBAModel()
        fba_model.model.reactions.get_by_id('BIOMASS_Ec_iJO1366_core_53p95M').bounds = 1, 1
        fba_model.solve(alg='pfba', conditions=pd.read_csv('perturbations_small.csv')).to_csv('output/two_round.csv')


if __name__ == '__main__':
    unittest.main()
