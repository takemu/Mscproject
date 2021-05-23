import unittest

import pandas as pd
from pandas._testing import assert_frame_equal

from mscproject.simulation.etfl_model import ETFLModel
from mscproject.simulation.fba_model import growth_reaction_id


class TestETFLModel(unittest.TestCase):
    def setUp(self):
        self.condition = 'arab__L'
        self.obj = growth_reaction_id

    def test_etfl_baseline(self):
        etfl_model = ETFLModel(conditions=pd.DataFrame([[self.condition]]))
        solution = etfl_model.solve()
        solution.to_csv('etfl_small_baseline.csv')

    def test_etfl(self):
        etfl_model = ETFLModel(conditions=pd.DataFrame([[self.condition]]))
        solution = etfl_model.solve()
        baseline = pd.read_csv('etfl_small_baseline.csv', index_col=0)
        assert_frame_equal(solution, baseline)


if __name__ == '__main__':
    unittest.main()
