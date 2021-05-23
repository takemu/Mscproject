import unittest

import pandas as pd
from pandas._testing import assert_frame_equal

from mscproject.simulation.fba_model import growth_reaction_id
from mscproject.simulation.tfa_model import TFAModel


class TestTFAModel(unittest.TestCase):
    def setUp(self):
        self.condition = 'arab__L'
        self.obj = growth_reaction_id

    def test_tfa_baseline(self):
        tfa_model = TFAModel(conditions=pd.DataFrame([[self.condition]]))
        solution = tfa_model.solve()
        solution.to_csv('tfa_small_baseline.csv')

    def test_tfa(self):
        tfa_model = TFAModel(conditions=pd.DataFrame([[self.condition]]))
        solution = tfa_model.solve()
        baseline = pd.read_csv('tfa_small_baseline.csv', index_col=0)
        assert_frame_equal(solution, baseline)


if __name__ == '__main__':
    unittest.main()
