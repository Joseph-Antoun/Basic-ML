import numpy as np


class LogisticRegression:

    def __init__(self, X, y, x_labels, y_label, alpha=1, num_iters=100, stopping_criteria=10e-2):
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
        self.w_learned = np.ones((self.m, 1))
        self.iter_err = []

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

    def fit(self, X, y):

        x = X  # this should get the features matrix without the results y
        y = y  # This should get the results of  trained features
        alpha = self.alpha
        n, m = np.shape(X)
        w_init = np.zeros((m, 1))
        epsilon = self.epsilon
        stop = 1

        for i in range(0, self.num_iters):
            z = y - self.sigmoid(np.matmul(x, w_init))
            w_init = w_init + ((alpha / n) * (np.matmul(x.T, z)))
            # w_init = w_init + ((alpha/n) * (np.sum(np.matmul(x.T, z), axis=1)))
            # z1 = np.matmul(np.transpose(w), np.transpose(x))
            # q1 = np.transpose(y) - self.sigmoid(z1)
            # stop = np.linalg.norm(np.subtract(q1, q), 2)
            # if stop < self.epsilon:
            #     break
            # stop = np.linalg.norm(np.subtract(w, w_init), 2)
            # self.inter_predict(w)

        self.w_learned = w_init

    def predict(self, X):
        x = X
        w = self.w_learned
        # z = np.matmul(x, w)
        # p = self.sigmoid(z)
        predicted_y = np.around(self.sigmoid(np.matmul(x, w)))
        return predicted_y

    def inter_predict(self, w):
        z = np.matmul(np.transpose(w), np.transpose(self.X))
        p = self.sigmoid(z)
        predicted_y = np.around(p)
        err = (np.sum(abs(self.y - predicted_y)) / len(np.transpose(self.y)))
        self.iter_err.append(err)


    # def evaluate_acc(self, data):
    #     x = data.x
    #     y = data.y
    #     predicted_y = self.predict(x)
    #
    #     acc = (1 - (np.sum(abs(y - predicted_y)) / len(np.transpose(y)))) * 100
    #
    #     return acc


# def main():
#
#     # Load & clean the data
#     file_path   = '../data/wine/winequality-red.csv'
#     raw_data    = pd.read_csv(file_path, delimiter=';')
#     clean_data  = data_cleaning.get_clean_data(raw_data, verbose=False)
#
#     # Create categoric y column
#     # this is to skip SettingWithCopyWarning from Pandas
#     clean_df   = data_cleaning.get_clean_data(raw_data, verbose=False)
#     clean_data = clean_df.copy()
#     # Create the binary y column
#     clean_data['y'] = np.where(clean_df['quality'] >= 6.0, 1.0, 0.0)
#     # Drop the 'quality' column as it shouldn't be used to predict the wine binary rating
#     clean_data.drop('quality', axis=1, inplace=True)
#
#
#     # Split between X and y and create the numpy arrays
#     y_vars = ['quality', 'y']
#     x_vars = [var for var in clean_data.columns.tolist() if not var in y_vars]
#     X, y = dataframe_to_narray(clean_data, x_vars, 'y')
#
#     # Instanciate LogisticRegression
#     lr = LogisticRegression(X, y, x_vars, 'y')
#     print(lr)
#
#
# if __name__ == "__main__":
#     main()





