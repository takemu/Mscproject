import os
import sys
from os.path import exists, splitext, join

import pandas as pd

from mscproject.simulation.etfl_model import ETFLModel


def split_csv(csv_file='data/perturbations.csv', slice_no=10):
    output_dir = splitext(csv_file)[0]
    if not exists(output_dir):
        os.mkdir(output_dir)
    df = pd.read_csv(csv_file)
    step = df.shape[0] // slice_no
    start = 0
    end = start + step
    for i in range(slice_no):
        slice_df = df.iloc[start: end, :]
        slice_df.to_csv(join(output_dir, f'_slice_{i}.csv'), index=False)
        start += step
        end += step
        if end > df.shape[0]:
            end = df.shape[0]


if __name__ == '__main__':
    if sys.argv[1] == 'split':
        split_csv()
    else:
        etfl_model = ETFLModel(model_code='ecoli:iML1515', min_biomass=0.1)
        if sys.argv[1] == 'glc':
            etfl_model.solve(conditions=pd.read_csv('data/glc_uptakes.csv')).to_csv('output/glc_etfl_fluxes.csv')
        elif 'slice' in sys.argv[1]:
            etfl_model.solve(conditions=pd.read_csv(f'data/perturbations/_{sys.argv[1]}.csv')).to_csv(
                f'output/etfl_fluxes_{sys.argv[1]}.csv')
