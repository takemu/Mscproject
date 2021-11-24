import time

import numpy as np
import pandas as pd

from mscproject.ml.data import load_train_data, show_result, split_train_data, iml1515_excluded_conditions
from mscproject.ml.enmlp import ElasticNetMLPCV


# def test_enmlpcv2(name='ptfa', rm_dup=True):
#     st = time.time()
#     X, y = load_train_data(name=name, rm_dup=rm_dup)
#     train_X, train_y, test_X, test_y = split_train_data(X, y, train_ratio=0.9)
#     d_input, d_hidden, d_output = X.shape[1], [50, 50, 50], y.shape[1]
#     enmlp = ElasticNetMLPCV(d_input, d_hidden, d_output, opt_alg='gd-adam', max_iter=1000, alphas=[0.01, 0.05, 0.1],
#                             cv=10)
#     enmlp.train(train_X, train_y)
#     print(f"Training cost {time.time() - st:2f} seconds!")
#
#     coefs = pd.DataFrame(enmlp.weights.T, columns=['All_IC50'], index=X.columns)
#     # coefs = pd.DataFrame(enmlp.weights.tolist(), columns=X.columns).T.rename('All_IC50')
#     coefs = coefs.abs()
#     coefs[coefs < 1e-4] = 0
#     coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
#     coefs = coefs.round(decimals=6)
#     coefs = coefs.sort_values(by='All_IC50', ascending=False)
#     coefs.to_csv(f"output/{name}_mlp_coefs.csv")
#
#     predicted_y = enmlp.predict(test_X)
#     show_result(test_y, predicted_y)


def test_enmlpcv(name, rm_dup=True, excludes=iml1515_excluded_conditions, train_ratio=1, times=1):
    st = time.time()
    X, y = load_train_data(name=name, rm_dup=rm_dup, excludes=excludes)
    if train_ratio < 1:
        X, y, test_X, test_y = split_train_data(X, y, train_ratio=train_ratio)
    d_input, d_hidden, d_output = X.shape[1], [50, 50, 50], y.shape[1]
    coefs = []
    for i in range(times):
        enmlp = ElasticNetMLPCV(d_input, d_hidden, d_output, opt_alg='gd-adam', alphas=[0.01, 0.05, 0.1], cv=10)
        enmlp.train(X, y)
        coefs.append(enmlp.weights.tolist()[0])
    print(f"Training cost {time.time() - st:2f} seconds!")

    # coefs = pd.DataFrame(enmlp.weights.T, columns=['All_IC50'], index=X.columns)
    coefs = pd.DataFrame(coefs, columns=X.columns).mean(axis=0).T.rename('All_IC50')
    coefs = coefs.abs()
    coefs = coefs[coefs >= 1e-4]
    coefs = np.log10(coefs)
    coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
    coefs = coefs.round(decimals=2)
    coefs = coefs.sort_values(ascending=False)
    coefs.to_csv(f"output/{name}_mlp_coefs.csv")

    if times == 1:
        predicted_y = enmlp.predict(X)
        show_result(y, predicted_y)
        if train_ratio < 1:
            predicted_y = enmlp.predict(test_X)
            show_result(test_y, predicted_y)


if __name__ == '__main__':
    # test_enmlpcv(name='yangs', rm_dup=False, excludes=['galt', 'pser__L'])
    # test_enmlpcv(name='etfl', train_ratio=0.9)
    test_enmlpcv(name='etfl', times=1)
