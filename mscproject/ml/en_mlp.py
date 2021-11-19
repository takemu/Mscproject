import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

from mscproject.ml.mlp import MLP, load_data


def compute_l1_loss(w):
    return torch.abs(w).sum()


def compute_l2_loss(w):
    return torch.square(w).sum()


class ElasticNetMLP(nn.Module):

    def __init__(self, d_input, d_hidden, d_output, act_func='relu', opt_alg='sgd-adam', max_iter=1000, step_size=0.05,
                 minibatch_size=100, l1_ratio=0.5, alpha=0.05):
        super().__init__()
        self.mlp = MLP(d_input, d_hidden, d_output, act_func, opt_alg, max_iter, step_size, minibatch_size)
        self.weights = nn.Parameter(torch.zeros([1, d_input]))
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_func = torch.nn.MSELoss()
        self.max_iter = max_iter
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def regularization(self, loss_func):
        l1_weight = self.alpha * self.l1_ratio
        l2_weight = self.alpha * (1 - self.l1_ratio) / 2
        l1 = l1_weight * compute_l1_loss(self.weights)
        l2 = l2_weight * compute_l2_loss(self.weights)
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
        weighted_X = self.weights * X
        return self.mlp.forward(weighted_X)

    def train(self, train_X, train_y, valid_X, valid_y):
        self.mlp.train(train_X, train_y, valid_X, valid_y)

        train_loss_hist = []
        valid_loss_hist = []
        for t in range(self.max_iter):
            fitted_train_y = self.forward(train_X)
            self.optimizer.zero_grad()
            self.regularization(self.loss_func(fitted_train_y, train_y)).backward()
            self.optimizer.step()
            train_loss_hist.append(self.mlp.total_loss(train_y, fitted_train_y))
            if len(valid_y) > 0:
                fitted_valid_y = self.forward(valid_X)
                valid_loss_hist.append(self.mlp.total_loss(valid_y, fitted_valid_y))
        return train_loss_hist, valid_loss_hist

    def get_weights(self):
        return self.weights


if __name__ == "__main__":
    st = time.time()

    train_X, train_y, valid_X, valid_y = load_data()
    d_input, d_hidden, d_output = train_X.shape[1], [30, 30, 30], train_y.shape[1]
    en_mlp = ElasticNetMLP(d_input, d_hidden, d_output)

    train_loss, valid_loss = en_mlp.train(train_X, train_y, valid_X, valid_y)

    f, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax.set_title(f"PyTorch MLP")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.plot([e.item() for e in train_loss], label="Training loss")
    ax.plot([e.item() for e in valid_loss], label="Validation loss")
    ax.legend()
    plt.show()

    print(f"Time elapsed {time.time() - st:2f} seconds!")
