from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_predict

from mscproject.ml.data import data_dir

duplicated_names = ['2obut', '2pg', '2pglyc', '3pg', '3sala', '4hbz', '6pgc', 'acgal', 'acglu', 'arbt', 'citr__L',
                    'crn-crn__D', 'cyst__L', 'glc__D', 'glu__D', 'glx', 'glycogen', 'hom__L', 'inost', 'man1p',
                    'met__D', 'no2', 'no3', 'oxa', 'pep', 'pi', 'ppi', 'so4', 'thym', 'tsul', 'tym', 'urate', 'urea']


def remove_duplicated1(X, Y):
    X = X.loc[~X.index.str.replace("(\.\d+)$", "").duplicated(), :]
    Y = Y.loc[~Y.index.str.replace("(\.\d+)$", "").duplicated(), :]
    return X, Y


def remove_duplicated2(X, Y):
    X = X.drop(duplicated_names, axis=0)
    Y = Y.drop(duplicated_names, axis=0)
    return X, Y


def remove_duplicated(X, Y):
    X, Y = remove_duplicated1(X, Y)
    return remove_duplicated2(X, Y)


class MultiTaskElasticNetRegression:
    def __init__(self, alpha=0.05, cv=10):
        # self.alpha = alpha
        self.cv = cv
        self.clf = linear_model.MultiTaskElasticNetCV(cv=cv, max_iter=1e4, tol=1e6, alphas=[alpha, alpha, alpha])

    def fit(self, X, Y):
        self.clf.fit(X, Y)
        coefs = pd.DataFrame(self.clf.coef_.T, columns=Y.columns, index=X.columns)
        coefs = coefs.abs()
        # coefs = (coefs - coefs.mean()) / coefs.std()
        coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
        coefs = coefs.round(decimals=6)
        return coefs

    def validate(self, X, Y, title=''):
        predicted_Y = cross_val_predict(self.clf, X, Y, cv=self.cv)
        rmse = mean_squared_error(Y, predicted_Y)
        fig, ax = plt.subplots()
        ax.scatter(Y, predicted_Y, edgecolors=(0, 0, 0))
        ax.plot([-2.8, 2.8], [-2.8, 2.8], '--', lw=2)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.set_title(f'{title}\nCV = {self.cv}\nRMSE = {rmse:.6f}')
        plt.xlim([-2.9, 2.9])
        plt.ylim([-2.9, 2.9])
        plt.show()


def our_data():
    X = pd.read_csv(join(data_dir, 'ptfa_fluxes.csv'), index_col=0)
    X = X[:-1].T.fillna(0)
    Y = pd.read_csv(join(data_dir, 'log_IC50.csv'), index_col=0)
    mtenr = MultiTaskElasticNetRegression(cv=100)

    # mtenr.fit(X, Y)
    # mtenr.validate(X, Y, 'Orignal')
    #
    # X, Y = remove_duplicated1(X, Y)
    # mtenr.fit(X, Y)
    # mtenr.validate(X, Y, 'Remove duplicated conditions')
    #
    # X, Y = remove_duplicated2(X, Y)
    # results = mtenr.fit(X, Y)
    # mtenr.validate(X, Y, 'Remove conditions where metabolic fluxes are identical with Control')

    X, Y = remove_duplicated(X, Y)
    results = mtenr.fit(X, Y)
    results.to_csv('output/our_coefs.csv')


def yangs_data():
    X = pd.read_csv(join(data_dir, 'yangs_data.csv'), index_col=0)
    X = X[:-1].T.fillna(0)
    Y = pd.read_csv(join(data_dir, 'log_IC50.csv'), index_col=0)
    mtenr = MultiTaskElasticNetRegression(cv=50)
    results = mtenr.fit(X, Y)
    results.to_csv('output/yangs_coefs.csv')


if __name__ == '__main__':
    our_data()
    yangs_data()
