import os
from os.path import join

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

data_dir = os.path.dirname(os.path.abspath(__file__))

iml1515_excluded_conditions = ['2obut', 'glx', 'glycogen', '4hbz', 'oxa', 'acglu', 'glu__D', 'citr__L', 'ppi', '2pg',
                               '3pg', 'pep', '2pglyc', '6pgc', 'man1p', '3sala', 'cyst__L']

ijo1366_excluded_conditions = iml1515_excluded_conditions + ['urate']

control_identical_conditions = ijo1366_excluded_conditions + ['acgal', 'arbt', 'crn-crn__D', 'glc__D', 'hom__L',
                                                              'inost', 'met__D', 'no2', 'no3', 'pi', 'so4', 'thym',
                                                              'tsul', 'tym', 'urea']


def load_train_data(name='ptfa', rm_dup=True, excludes=[]):
    X = pd.read_csv(join(data_dir, f'{name}_fluxes.csv'), index_col=0)
    X = X[:-1].T.fillna(0)
    y = pd.read_csv(join(data_dir, 'log_IC50.csv'), index_col=0)
    if rm_dup:
        X, y = remove_duplicates(X, y)
        X, y = remove_conditions(X, y, excludes)
    return X, y


def remove_duplicates(X, y):
    X = X.loc[~X.index.str.replace(r"(\.\d+)$", "", regex=True).duplicated(), :]
    y = y.loc[~y.index.str.replace(r"(\.\d+)$", "", regex=True).duplicated(), :]
    return X, y


def remove_conditions(X, y, conditions):
    X = X.drop(conditions, axis=0)
    y = y.drop(conditions, axis=0)
    return X, y


def split_train_data(X, y, train_ratio=0.8):
    n_train = int(X.shape[0] * train_ratio)
    # train_X = torch.from_numpy(X.iloc[:n_train, :].values).float()
    # train_y = torch.from_numpy(y.iloc[:n_train, :].values).float()
    # valid_X = torch.from_numpy(X.iloc[n_train:X.shape[0], :].values).float()
    # valid_y = torch.from_numpy(y.iloc[n_train:X.shape[0], :].values).float()
    train_X = X.iloc[:n_train, :]
    train_y = y.iloc[:n_train, :]
    valid_X = X.iloc[n_train:X.shape[0], :]
    valid_y = y.iloc[n_train:X.shape[0], :]
    return train_X, train_y, valid_X, valid_y


def to_tensor(x):
    if torch.is_tensor(x) or len(x) == 0:
        return x
    else:
        return torch.from_numpy(x.values).float()


def show_result(y, predicted_y):
    rmse = mean_squared_error(y, predicted_y)
    r2 = r2_score(y, predicted_y)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted_y, edgecolors=(0, 0, 0))
    ax.plot([-2.8, 2.8], [-2.8, 2.8], '--', lw=2)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.text(-2.5, 2, f'RMSE = {rmse:.6f}\nR$^2$ score = {r2:.6f}')
    plt.xlim([-2.9, 2.9])
    plt.ylim([-2.9, 2.9])
    plt.show()
