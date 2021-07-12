import unittest

import pandas as pd

from mscproject.simulation.old.fba_model import growth_reaction_id
from mscproject.simulation.old.tfa_model import TFAModel


class TestTFAModelBaseline(unittest.TestCase):
    def setUp(self):
        self.condition = 'arab__L'
        self.obj = growth_reaction_id

    def test_tfa_baseline(self):
        tfa_model = TFAModel(pfba=True, conditions=pd.DataFrame([[self.condition]]))
        solution = tfa_model.solve()
        solution.to_csv('tfa_small_baseline.csv')

    def test_tfa_batch_baseline(self):
        tfa_model = TFAModel(conditions=pd.read_csv('../../simulation/data/perturbations.csv'))
        solution = tfa_model.solve()
        # baseline = pd.read_csv('tfa_small_baseline.csv', index_col=0)
        # assert_frame_equal(solution, baseline)
        solution.to_csv('tfa_batch_baseline.csv')


if __name__ == '__main__':
    unittest.main()
