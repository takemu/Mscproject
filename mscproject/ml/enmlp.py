import time

import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch import nn

from mscproject.ml.data import load_train_data, split_train_data, to_tensor
from mscproject.ml.mlp import MLP


def compute_l1_loss(w):
    return torch.abs(w).sum()


def compute_l2_loss(w):
    return torch.square(w).sum()


class ElasticNetMLP(MLP):

    def __init__(self, d_input, d_hidden, d_output, act_func='relu', opt_alg='sgd-adam', max_iter=1000, step_size=1e-3,
                 minibatch_size=100, l1_ratio=0.5, alpha=0.05):
        super().__init__(d_input, d_hidden, d_output, act_func, opt_alg, max_iter, step_size, minibatch_size)
        # self.mlp = MLP(d_input, d_hidden, d_output, act_func, opt_alg, max_iter, step_size, minibatch_size)
        self._weights = nn.Parameter(torch.zeros([1, d_input]))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step_size)
        # self.loss_func = torch.nn.MSELoss()
        # self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def regularization(self, loss_func):
        l1_weight = self.alpha * self.l1_ratio
        l2_weight = self.alpha * (1 - self.l1_ratio) / 2
        l1 = l1_weight * compute_l1_loss(self._weights)
        l2 = l2_weight * compute_l2_loss(self._weights)
        loss_func += l1
        loss_func += l2
        # parameters = []
        # for parameter in self.mlp.parameters():
        #     parameters.append(parameter.view(-1))
        # l1_weight *= 2
        # l2_weight *= 2
        # l1 = l1_weight * compute_l1_loss(torch.cat(parameters))
        # l2 = l2_weight * compute_l2_loss(torch.cat(parameters))
        # loss_func += l1
        # loss_func += l2
        return loss_func

    def forward(self, X):
        weighted_X = self._weights * X
        return super().forward(weighted_X)

    def train(self, train_X, train_y, valid_X=[], valid_y=[]):
        super().train(train_X, train_y, valid_X, valid_y)

        train_loss_hist = []
        valid_loss_hist = []
        for t in range(self.max_iter):
            fitted_train_y = self.forward(train_X)
            self.optimizer.zero_grad()
            self.regularization(self.loss_func(fitted_train_y, train_y)).backward()
            self.optimizer.step()
            train_loss_hist.append(self.total_loss(train_y, fitted_train_y))
            if len(valid_y) > 0:
                fitted_valid_y = self.forward(valid_X)
                valid_loss_hist.append(self.total_loss(valid_y, fitted_valid_y))
        return train_loss_hist, valid_loss_hist

    def predict(self, X):
        return self.forward(torch.from_numpy(X.values).float()).detach().numpy()

    @property
    def weights(self):
        # return self.weights
        return self._weights.detach().numpy()  # .T.tolist()


class ElasticNetMLPCV(ElasticNetMLP):
    def __init__(self, d_input, d_hidden, d_output, act_func='relu', opt_alg='sgd-adam', max_iter=1000, step_size=1e-3,
                 minibatch_size=100, l1_ratio=0.5, alpha=0.05, cv=10):
        super().__init__(d_input, d_hidden, d_output, act_func, opt_alg, max_iter, step_size, minibatch_size, l1_ratio,
                         alpha)
        self.cv = cv

    def train(self, X, y):
        kf = KFold(n_splits=self.cv)
        for t, v in kf.split(X):
            train_X = torch.from_numpy(X.iloc[t, :].values).float()
            train_y = torch.from_numpy(y.iloc[t, :].values).float()
            valid_X = torch.from_numpy(X.iloc[v, :].values).float()
            valid_y = torch.from_numpy(y.iloc[v, :].values).float()
            super().train(train_X, train_y, valid_X, valid_y)
            predicted_y = super().predict(valid_X)


if __name__ == "__main__":
    st = time.time()
    X, y = load_train_data()
    train_X, train_y, valid_X, valid_y = split_train_data(X, y)
    d_input, d_hidden, d_output = train_X.shape[1], [50, 50, 50], train_y.shape[1]
    en_mlp = ElasticNetMLP(d_input, d_hidden, d_output)
    train_loss, valid_loss = en_mlp.train(to_tensor(train_X), to_tensor(train_y), to_tensor(valid_X),
                                          to_tensor(valid_y))
    print(f"Training cost {time.time() - st:2f} seconds!")

    f, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.set_title(f"PyTorch MLP")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.plot([e.item() for e in train_loss], label="Training loss")
    ax.plot([e.item() for e in valid_loss], label="Validation loss")
    ax.legend()
    plt.show()