import time
from os.path import join

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from mscproject.ml.data import data_dir
from mscproject.ml.en_mlp import ElasticNetMLP
from mscproject.ml.mten_regr import remove_duplicated1


class ElasticNetMLPCV(ElasticNetMLP):
    def __init__(self, d_input, d_hidden, d_output, act_func='relu', opt_alg='sgd-adam', max_iter=1000, step_size=0.05,
                 minibatch_size=100, l1_ratio=0.5, alpha=0.05, cv=10):
        super().__init__(d_input, d_hidden, d_output, act_func, opt_alg, max_iter, step_size, minibatch_size, l1_ratio,
                         alpha)
        self.cv = cv

    def train(self, X, Y):
        kf = KFold(n_splits=self.cv)
        for t, v in kf.split(X):
            # train_X, train_Y, valid_X, valid_Y = X[t], train_Y[t], valid_X[v], valid_Y[v]
            train_X = torch.from_numpy(X.iloc[t, :].values).float()
            train_Y = torch.from_numpy(Y.iloc[t, :].values).float()
            valid_X = torch.from_numpy(X.iloc[v, :].values).float()
            valid_Y = torch.from_numpy(Y.iloc[v, :].values).float()
            super().train(train_X, train_Y, valid_X, valid_Y)


# def split_data(X, Y):
#     n_test = X.shape[0] // 10 * 8
#     train_X = torch.from_numpy(X.iloc[:n_test, :].values).float()
#     train_Y = torch.from_numpy(Y.iloc[:n_test, :].values).float()
#     valid_X = torch.from_numpy(X.iloc[n_test:X.shape[0], :].values).float()
#     valid_Y = torch.from_numpy(Y.iloc[n_test:X.shape[0], :].values).float()
#     return train_X, train_Y, valid_X, valid_Y


def show_result(Y, predicted_Y):
    rmse = mean_squared_error(Y, predicted_Y)
    r2 = r2_score(Y, predicted_Y)
    fig, ax = plt.subplots()
    ax.scatter(Y, predicted_Y, edgecolors=(0, 0, 0))
    ax.plot([-2.8, 2.8], [-2.8, 2.8], '--', lw=2)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    # ax.set_title(f'\nRMSE = {rmse:.6f}\nR2 score = {r2:.6f}')
    ax.text(-2.5, 2, f'RMSE = {rmse:.6f}\nR$^2$ score = {r2:.6f}')
    plt.xlim([-2.9, 2.9])
    plt.ylim([-2.9, 2.9])
    plt.show()


def test(name='ptfa', rm_dup=True):
    start_time = time.time()
    X = pd.read_csv(join(data_dir, f'{name}_fluxes.csv'), index_col=0)
    X = X[:-1].T.fillna(0)
    Y = pd.read_csv(join(data_dir, 'log_IC50.csv'), index_col=0)
    if rm_dup:
        X, Y = remove_duplicated1(X, Y)

    # train_X, train_Y, valid_X, valid_Y = split_data(X, Y)
    d_input, d_hidden, d_output = X.shape[1], [50, 50, 50], Y.shape[1]
    enmlp = ElasticNetMLPCV(d_input, d_hidden, d_output, opt_alg='gd-adam', max_iter=1000, alpha=0.05, cv=5)
    enmlp.train(X, Y)
    print(f"Training costs {time.time() - start_time:.2f} seconds!")

    predicted_Y = enmlp.forward(torch.from_numpy(X.values).float()).detach().numpy()
    show_result(Y, predicted_Y)
    coefs = pd.DataFrame(enmlp.get_weights().detach().numpy().T.tolist(), columns=['All_IC50'], index=X.columns)
    coefs = coefs.abs()
    # coefs[coefs < 1e-4] = 0
    coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
    coefs = coefs.round(decimals=6)
    coefs.to_csv("output/ptfa_mlp_coefs.csv")


if __name__ == '__main__':
    test('ptfa')

