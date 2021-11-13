import time

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MLP2(torch.nn.Module):
    def __init__(self, d_input, d_hidden, d_output, act_func='sigmoid', opt_alg='gd-adam', max_iter=100, step_size=0.1,
                 minibatch_size=100):
        super().__init__()
        if isinstance(d_hidden, int):
            d_hidden = [d_hidden]
        self.input_linear = torch.nn.Linear(d_input, d_hidden[0])
        self.n_hidden = len(d_hidden)
        self.hidden_linear = []
        if self.n_hidden > 1:
            for i in range(1, self.n_hidden):
                self.hidden_linear.append(torch.nn.Linear(d_hidden[i - 1], d_hidden[i]))
        self.output_linear = torch.nn.Linear(d_hidden[-1], d_output)

        self.act_func = act_func
        self.opt_alg = opt_alg
        self.max_iter = max_iter
        self.minibatch_size = minibatch_size

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=step_size)
        self.loss_func = torch.nn.MSELoss()

    def activation(self, z):
        if self.act_func.lower() == 'sigmoid':
            return self.sigmoid(z)
        elif self.act_func.lower() == 'relu':
            return self.relu(z)
        else:
            return z

    def total_loss(self, target_y, output_y, method='mean'):
        if method == 'mean':
            return torch.mean(0.5 * torch.sum((output_y - target_y) ** 2, axis=1))
        else:
            return torch.sum(0.5 * torch.sum((output_y - target_y) ** 2, axis=1))

    def forward(self, input_x):
        hidden_z = self.activation(self.input_linear(input_x))
        if self.n_hidden > 1:
            for i in range(0, self.n_hidden - 1):
                hidden_z = self.activation(self.hidden_linear[i](hidden_z))
        output_y = self.output_linear(hidden_z)
        return output_y

    def gd_adam(self, train_x, train_y, valid_x, valid_y):
        train_loss_hist = []
        valid_loss_hist = []

        for t in range(self.max_iter):
            fitted_train_y = self.forward(train_x)
            train_loss_hist.append(self.total_loss(train_y, fitted_train_y))
            if len(valid_y) > 0:
                fitted_valid_y = self.forward(valid_x)
                valid_loss_hist.append(self.total_loss(valid_y, fitted_valid_y))
            self.optimizer.zero_grad()
            self.loss_func(fitted_train_y, train_y).backward()
            self.optimizer.step()

        return train_loss_hist, valid_loss_hist

    def sgd_adam(self, train_x, train_y, valid_x, valid_y):
        train_loss_hist = []
        valid_loss_hist = []
        n_minibatches = max(int(len(train_x) / self.minibatch_size), 1)
        for t in range(self.max_iter):
            total_mb_loss = 0
            for j in range(n_minibatches):
                mb_pos = j * self.minibatch_size
                mb_x, mb_y = train_x[mb_pos:mb_pos + self.minibatch_size], train_y[mb_pos:mb_pos + self.minibatch_size]
                fitted_mb_y = self.forward(mb_x)
                total_mb_loss += self.total_loss(mb_y, fitted_mb_y)
                self.optimizer.zero_grad()
                self.loss_func(fitted_mb_y, mb_y).backward()
                self.optimizer.step()
            train_loss_hist.append(total_mb_loss / n_minibatches)
            if len(valid_y) > 0:
                fitted_valid_y = self.forward(valid_x)
                valid_loss_hist.append(self.total_loss(valid_y, fitted_valid_y))

        return train_loss_hist, valid_loss_hist

    def train(self, train_x, train_y, valid_x=[], valid_y=[]):
        if train_y.ndim == 1:
            train_y = train_y.reshape(len(train_y), 1)
        if len(valid_y) > 0 and valid_y.ndim == 1:
            valid_y = valid_y.reshape(len(valid_y), 1)

        if self.opt_alg.lower() == 'gd-adam':
            return self.gd_adam(train_x, train_y, valid_x, valid_y)
        elif self.opt_alg.lower() == 'sgd-adam':
            return self.sgd_adam(train_x, train_y, valid_x, valid_y)
        else:
            return [], []


def load_data2():
    train_df = pd.read_csv('problem2_train.csv', header=None)
    valid_df = pd.read_csv('problem2_valid.csv', header=None)
    train_x = torch.from_numpy(train_df.iloc[:, 0:10].values).float()
    train_y = torch.from_numpy(train_df.iloc[:, 10].values[:, np.newaxis]).float()
    valid_x = torch.from_numpy(valid_df.iloc[:, 0:10].values).float()
    valid_y = torch.from_numpy(valid_df.iloc[:, 10].values[:, np.newaxis]).float()
    return train_x, train_y, valid_x, valid_y


def test():
    d_input, d_hidden, d_output = 10, 15, 1
    step_size = 0.001
    minibatch_size = 1000
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    train_x, train_y, valid_x, valid_y = load_data2()
    mlp = MLP2(d_input, d_hidden, d_output, act_func='sigmoid', opt_alg='sgd-adam',
               max_iter=MAX_EPOCH, step_size=step_size, minibatch_size=minibatch_size)
    st = time.time()
    train_loss, valid_loss = mlp.train(train_x, train_y, valid_x, valid_y)
    print(time.time() - st)
    ax2.set_title(f"PyTorch MLP")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Epoch")
    ax2.plot(train_loss, label="Training loss")
    ax2.plot(valid_loss, label="Validation loss")
    ax2.legend()

    plt.show()


MAX_EPOCH = 300
if __name__ == "__main__":
    test()