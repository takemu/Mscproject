import time

import pandas as pd

from mscproject.ml.data import load_train_data, show_result, split_train_data
from mscproject.ml.enmlp import ElasticNetMLPCV


def test_enmlpcv2(name='ptfa', rm_dup=True):
    st = time.time()
    X, y = load_train_data(name=name, rm_dup=rm_dup)
    train_X, train_y, test_X, test_y = split_train_data(X, y, train_ratio=0.9)
    d_input, d_hidden, d_output = X.shape[1], [50, 50, 50], y.shape[1]
    enmlp = ElasticNetMLPCV(d_input, d_hidden, d_output, opt_alg='gd-adam', max_iter=1000, alphas=[0.01, 0.05, 0.1],
                            cv=10)
    enmlp.train(train_X, train_y)
    print(f"Training cost {time.time() - st:2f} seconds!")

    # coefs = pd.DataFrame(enmlp.weights.T, columns=['All_IC50'], index=X.columns)
    coefs = pd.DataFrame(enmlp.weights.tolist()[0], columns=X.columns).T.rename('All_IC50')
    coefs = coefs.abs()
    coefs[coefs < 1e-4] = 0
    coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
    coefs = coefs.round(decimals=6)
    coefs = coefs.sort_values(by='All_IC50', ascending=False)
    coefs.to_csv(f"output/{name}_mlp_coefs.csv")

    predicted_y = enmlp.predict(test_X)
    show_result(test_y, predicted_y)


def test_enmlpcv(name='ptfa', rm_dup=True, times=1):
    st = time.time()
    X, y = load_train_data(name=name, rm_dup=rm_dup)
    d_input, d_hidden, d_output = X.shape[1], [50, 50, 50], y.shape[1]
    coefs = []
    for i in range(times):
        enmlp = ElasticNetMLPCV(d_input, d_hidden, d_output, opt_alg='gd-adam', max_iter=1000, alphas=[0.01, 0.05, 0.1],
                                cv=10)
        enmlp.train(X, y)
        coefs.append(enmlp.weights.tolist()[0])
    print(f"Training cost {time.time() - st:2f} seconds!")

    # coefs = pd.DataFrame(enmlp.weights.T, columns=['All_IC50'], index=X.columns)
    coefs = pd.DataFrame(coefs, columns=X.columns).mean(axis=0).T.rename('All_IC50')
    coefs = coefs.abs()
    coefs[coefs < 1e-4] = 0
    coefs = (coefs - coefs.min()) / (coefs.max() - coefs.min()) * 100
    coefs = coefs.round(decimals=6)
    coefs = coefs.sort_values(ascending=False)
    coefs.to_csv(f"output/{name}_mlp_coefs.csv")
    #
    predicted_y = enmlp.predict(X)
    show_result(y, predicted_y)


if __name__ == '__main__':
    # test_enmlpcv('yangs')
    # test_enmlpcv2('ptfa', times=1)
    test_enmlpcv2('ptfa')
