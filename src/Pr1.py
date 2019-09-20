import math
import numpy as np
from random import random
import pandas as pd
import matplotlib.pyplot as plt


class Glob:
    """
   in this class we will have the global data such as the features and the raw data that will be used in all the other
    classes
    """
    n, m = np.shape(data)
    w_init = np.zeros(m, 1)



class LogisticRegression:

    def __init__(self, data, labels, alpha=1, num_iters=100, stopping_criteria = 10e-2):
        self.num_iters = num_iters
        self.alpha = alpha
        self.epsilon = stopping_criteria

    @staticmethod
    def sigmoid(t):

        """
        Calculate the sigmoid of the respective scalar
        :param t: w^t.x_i
        :return: sigmoid(t)
        """

        sig = 1/(1+np.exp(-t))
        return sig

    def fit(self, data, n_iters):

        x = data.x  # this should get the features matrix without the results y
        y = data.y  # This should get the results of  trained features
        alpha = self.alpha
        n, m = np.shape(data)
        w_init = np.random.random(m, 1)
        epsilon = self.epsilon
        stop = 1
        i = 0

        while i <= n_iters and stop > epsilon:
            z = np.matmul(np.transpose(w_init), np.transpose(x))  # calculating the inner input for the sigmoid
            q = np.transpose(y) - self.sigmoid(z)
            w = w_init + alpha * np.inner(np.transpose(x), q)
            stop = abs(w - w_init)
            w_init = w

        Glob.w_init = w_init

    def predict(self, t_data):
        x = t_data.x
        w = Glob.w_init
        z = np.matmul(np.transpose(w), np.transpose(x))
        p = self.sigmoid(z)
        predicted_y = np.around(p)
        return predicted_y

    def evaluate_acc(self, data):
        x = data.x
        y = data.y
        predicted_y = self.predict(x)

        acc = (1 - (np.sum(abs(y - predicted_y)) / len(np.transpose(y)))) * 100











