import os
import sys
from os.path import exists, splitext, join

import numpy as np
import pandas as pd

from mscproject.simulation.etfl_model import ETFLModel
from mscproject.simulation.tfa_model import TFAModel


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


def rename1():
    df = pd.read_csv('data/perturbations.csv')
    for i, row in df.iterrows():
        if isinstance(row['metabolite_1'], str) and row['metabolite_1'][0].isdigit():
            df.loc[i, 'metabolite_1'] = '_' + row['metabolite_1']
        if isinstance(row['metabolite_2'], str) and row['metabolite_2'][0].isdigit():
            df.loc[i, 'metabolite_2'] = '_' + row['metabolite_2']
    df = df.set_index('metabolite_1')
    df.to_csv('data/perturbations2.csv')


if __name__ == '__main__':
    if sys.argv[1] == 'split':
        split_csv(slice_no=int(sys.argv[2]))
    elif sys.argv[1] == 'rename1':
        rename1()
    else:
        if sys.argv[1] == 'glc':
            etfl_model = ETFLModel(model_code='ecoli:iML1515', min_biomass=0.1)
            etfl_model.solve(conditions=pd.read_csv('data/glc_uptakes.csv')).to_csv('output/glc_etfl_fluxes.csv')
        elif 'slice' in sys.argv[1]:
            etfl_model = ETFLModel(model_code='ecoli:iML1515', min_biomass=0.1)
            etfl_model.solve(conditions=pd.read_csv(f'data/perturbations/_{sys.argv[1]}.csv')).to_csv(
                f'output/etfl_fluxes_{sys.argv[1]}.csv')
        elif sys.argv[1] == 'tfa':
            tfa_model = TFAModel(model_code='ecoli:iML1515', min_biomass=0.1)
            tfa_model.solve(alg='pfba', conditions=pd.read_csv('data/perturbations.csv')).to_csv(
                'output/ptfa_fluxes.csv')
