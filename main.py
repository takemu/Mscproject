import sys

import pandas as pd

from mscproject.simulation.etfl_model import ETFLModel

if __name__ == '__main__':
    etfl_model = ETFLModel(model_code='ecoli:iML1515', min_biomass=0.1)

    if sys.argv[1] == 'glc':
        etfl_model.solve(conditions=pd.read_csv('data/glc_uptakes.csv')).to_csv('output/glc_etfl_fluxes.csv')
    elif 'slice' in sys.argv[1]:
        etfl_model.solve(conditions=pd.read_csv(f'data/perturbations/_{sys.argv[1]}.csv')).to_csv(
            f'output/etfl_fluxes/_{sys.argv[1]}.csv')
