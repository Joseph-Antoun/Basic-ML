import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use the clean_data module (in the same folder) to load the data
import data_cleaning


class LogisticRegression:

    def __init__(self, X, y, x_labels, y_label, alpha=1, num_iters=100, stopping_criteria=10e-2, ratio=[80.0, 10.0, 10.0]):
        self.n, self.m = np.shape(X)
        """ 
        Here we might want to add an extra column filled with ones to X for the intercept ?
        """
        self.X = X 
        self.y = y

        self.x_labels = x_labels,
        self.y_label  = y_label,

        self.w = np.random.rand(self.m, 1)  # random weights initialization
        self.num_iters = num_iters
        self.alpha = alpha
        self.epsilon = stopping_criteria
        self.w_learned = np.ones()

        self.split_ratio = ratio
        self.tr_x = np.ones()
        self.tr_y = np.ones()
        self.vl_x = np.ones()
        self.vl_y = np.ones()
        self.ts_x = np.ones()
        self.ts_y = np.ones()

        self.split_data()

    def __repr__(self):
        """
        This method overrides print(LogisticRegression) - as in line 129
        """
        str_ = """
        LogisticRegression\n
        n = %s
        m = %s
        weights     = %s
        #iterations = %s
        alpha       = %s
        epsilon     = %s
        x_labels    = %s
        y_label     = %s
        X = %s,
        y = %s
        """ % (self.n, self.m, self.w, self.num_iters, self.alpha, self.epsilon,
                self.x_labels, self.y_label, self.X, self.y)
        return str_

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

        self.w_learned = w_init

    def predict(self, t_data):
        x = t_data.x
        w = self.w_learned
        z = np.matmul(np.transpose(w), np.transpose(x))
        p = self.sigmoid(z)
        predicted_y = np.around(p)
        return predicted_y

    def evaluate_acc(self, data):
        x = data.x
        y = data.y
        predicted_y = self.predict(x)

        acc = (1 - (np.sum(abs(y - predicted_y)) / len(np.transpose(y)))) * 100

        return acc

    def split_data(self):
        n = self.n
        tr_ind = int((self.split_ratio[0]*n) / 100)
        vl_ind = int((self.split_ratio[1]*n) / 100)
        ts_ind = int((self.split_ratio[2]*n) / 100)
        self.tr_x = self.X[0:tr_ind, :]
        self.tr_y = self.y[0:tr_ind, :]
        self.vl_x = self.X[tr_ind:vl_ind, :]
        self.vl_y = self.y[tr_ind:vl_ind, :]
        self.ts_x = self.X[vl_ind:ts_ind, :]
        self.ts_y = self.y[vl_ind:ts_ind, :]


def dataframe_to_narray(df, x_vars, y_var):
    """
        Returns numpy arrays for X and y to be used in the logistic regression
    """
    X = df[x_vars].to_numpy()
    y = df[y_var].to_numpy()
    return X,y


def main():

    # Load & clean the data
    file_path   = '../data/wine/winequality-red.csv'
    raw_data    = pd.read_csv(file_path, delimiter=';')
    clean_data  = data_cleaning.get_clean_data(raw_data, verbose=False)

    # Create categoric y column
    # this is to skip SettingWithCopyWarning from Pandas
    clean_df   = data_cleaning.get_clean_data(raw_data, verbose=False)
    clean_data = clean_df.copy()
    # Create the binary y column
    clean_data['y'] = np.where(clean_df['quality'] >= 6.0, 1.0, 0.0)
    # Drop the 'quality' column as it shouldn't be used to predict the wine binary rating
    clean_data.drop('quality', axis=1, inplace=True)


    # Split between X and y and create the numpy arrays
    y_vars = ['quality', 'y']
    x_vars = [var for var in clean_data.columns.tolist() if not var in y_vars]
    X, y = dataframe_to_narray(clean_data, x_vars, 'y')

    # Instanciate LogisticRegression
    lr = LogisticRegression(X, y, x_vars, 'y')
    print(lr)    


if __name__ == "__main__":
    main()





