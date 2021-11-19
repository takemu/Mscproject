from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

from mscproject.ml.data import data_dir

duplicated_names = ['2obut', '2pg', '2pglyc', '3pg', '3sala', '4hbz', '6pgc', 'acgal', 'acglu', 'arbt', 'citr__L',
                    'crn-crn__D', 'cyst__L', 'glc__D', 'glu__D', 'glx', 'glycogen', 'hom__L', 'inost', 'man1p',
                    'met__D', 'no2', 'no3', 'oxa', 'pep', 'pi', 'ppi', 'so4', 'thym', 'tsul', 'tym', 'urate', 'urea']


def remove_duplicated1(X, Y):
    X = X.loc[~X.index.str.replace(r"(\.\d+)$", "", regex=True).duplicated(), :]
    Y = Y.loc[~Y.index.str.replace(r"(\.\d+)$", "", regex=True).duplicated(), :]
    return X, Y


def remove_duplicated2(X, Y):
    X = X.drop(duplicated_names, axis=0)
    Y = Y.drop(duplicated_names, axis=0)
    return X, Y


# def remove_duplicated(X, Y):
#     X, Y = remove_duplicated1(X, Y)
#     return remove_duplicated2(X, Y)


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
        r2 = r2_score(Y, predicted_Y)
        # adj_r2 = 1 - (1 - r2) * (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1)
        fig, ax = plt.subplots()
        ax.scatter(Y, predicted_Y, edgecolors=(0, 0, 0))
        ax.plot([-2.8, 2.8], [-2.8, 2.8], '--', lw=2)
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        # ax.set_title(f'{title}\nCV = {self.cv}\nRMSE = {rmse:.6f}\nAdjusted R2 score = {adj_r2:.6f}')
        # ax.set_title(f'RMSE = {rmse:.6f} R2 score = {r2:.6f}')
        ax.text(-2.5, 2, f'RMSE = {rmse:.6f}\nR$^2$ score = {r2:.6f}')
        plt.xlim([-2.9, 2.9])
        plt.ylim([-2.9, 2.9])
        plt.show()
        print(title, f'RMSE = {rmse:.6f}, R2 score = {r2:.6f}')


def validate(name='ptfa', cv=100):
    X = pd.read_csv(join(data_dir, f'{name}_fluxes.csv'), index_col=0)
    X = X[:-1].T.fillna(0)
    Y = pd.read_csv(join(data_dir, 'log_IC50.csv'), index_col=0)
    mtenr = MultiTaskElasticNetRegression(cv=cv)

    mtenr.fit(X, Y)
    mtenr.validate(X, Y, 'Orignal')

    X, Y = remove_duplicated1(X, Y)
    mtenr.fit(X, Y)
    mtenr.validate(X, Y, 'Remove duplicated conditions')

    # X, Y = remove_duplicated2(X, Y)
    # mtenr.fit(X, Y)
    # mtenr.validate(X, Y, 'Remove conditions where metabolic fluxes are identical with Control')


def test(name='ptfa', cv=100, rm_dup=True):
    X = pd.read_csv(join(data_dir, f'{name}_fluxes.csv'), index_col=0)
    X = X[:-1].T.fillna(0)
    Y = pd.read_csv(join(data_dir, 'log_IC50.csv'), index_col=0)
    if rm_dup:
        X, Y = remove_duplicated1(X, Y)
    mtenr = MultiTaskElasticNetRegression(cv=cv)
    results = mtenr.fit(X, Y)
    results.to_csv(f'output/{name}_lr_coefs.csv')


if __name__ == '__main__':
    test('yangs', 50, rm_dup=False)
    test('ptfa', 100)
    # validate('ptfa', 100)
