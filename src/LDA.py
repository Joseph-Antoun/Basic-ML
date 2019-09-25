import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_cleaning


class LDA:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n, self.m = np.shape(X)

        self.w_0 = np.ones()
        self.w_1 = np.ones()

    def fit(self, X, y):
        n = self.n
        m = self.m

        tr_x = X
        tr_y = y

        n_0 = np.count_nonzero(tr_y == 0)
        n_1 = np.count_nonzero(tr_y)
        p_1 = n_1/(n_0 + n_1)
        p_0 = n_0/(n_0 + n_1)
        ind_0 = 1 - np.sign(tr_y)
        ind_1 = np.sign(tr_y)
        muo_0 = np.sum(np.multiply(ind_0, tr_x)/n_0, axis=0)
        muo_0 = muo_0.reshape(1, m)
        muo_1 = np.sum(np.multiply(ind_1, tr_x)/n_1, axis=0)
        muo_1 = muo_1.reshape(1, m)

        sigma = np.zeros((m, m))
        for i in range(0, n+1):
            sigma = sigma + (ind_0[i]*np.matmul(np.transpose(tr_x[i]-muo_0), (tr_x[i] - muo_0)))/(n_0 + n_1 - 2) +\
                    (ind_1[i]*np.matmul(np.transpose(tr_x[i]-muo_1), (tr_x[i] - muo_1)))/(n_0 + n_1 - 2)

        w_0 = np.log((p_1/p_0)) - (1/2) * np.matmul(muo_1, np.matmul(np.linalg.pinv(sigma), np.transpose(muo_1))) +\
            (1/2) * np.matmul(muo_0, np.matmul(np.linalg.pinv(sigma), np.transpose(muo_0)))
        w_1 = np.matmul(np.linalg.pinv(sigma), np.transpose(muo_1 - muo_0))

        self.w_0 = w_0
        self.w_1 = w_1

    def predict(self, d_x):
        w_0 = self.w_0
        w_1 = self.w_1

        d_x = d_x

        d = w_0 + np.matmul(np.transpose(d_x), w_1)

        d[d <= 0] = 0
        d[d > 0] = 1

        return d

    # def evaluate_acc(self, ev_x, ev_y):
    #     x = ev_x
    #     y = ev_y
    #     predicted_y = self.predict(x)
    #
    #     acc = (1 - (np.sum(abs(y - predicted_y)) / len(np.transpose(y)))) * 100
    #
    #     return acc


