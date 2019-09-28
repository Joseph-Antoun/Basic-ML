import numpy as np


class LogisticRegression:

    def __init__(self, X, y, x_labels, y_label, alpha=1, num_iters=100, stopping_criteria=1e-50):
        """
        Constructor for the logistic regression class

        :param X: the features data used for train
        :param y: the classification data used for train
        :param x_labels: the label of the features
        :param y_label: the label for the results
        :param alpha: learning rate
        :param num_iters: number of iterations
        :param stopping_criteria: threshold to stop the iterations of gradient descent
        """
        self.n, self.m = np.shape(X)
        self.X = X 
        self.y = y

        self.x_labels = x_labels,
        self.y_label  = y_label,

        self.w = np.random.rand(self.m, 1)  # random weights initialization
        self.num_iters = num_iters
        self.alpha = alpha
        self.epsilon = stopping_criteria
        self.w_learned = np.zeros((self.m, 1))
        self.h_cost = np.zeros((self.num_iters, 1))

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
        """
        This method is used to train (fit) the weight of the logistic regression model
        :param X:  training features data
        :param y: classification training data
        """
        x = X  # this should get the features matrix without the results y
        y = y  # This should get the results of  trained features
        alpha = self.alpha
        n, m = np.shape(X)
        w_init = np.zeros((m, 1))
        epsilon = self.epsilon
        stop = 1

        for i in range(0, self.num_iters):
            z = y - self.sigmoid(np.matmul(x, w_init))
            w = w_init + ((alpha / n) * (np.matmul(x.T, z)))
            self.h_cost[i] = self.cost_computation(x, y, w)
            if i != 0:
                ar_stop = abs(w_init - w)
                stop = np.amax(ar_stop)
            w_init = w
            if stop < epsilon:
                print("threshold reached.\n")
                break


        self.w_learned = w_init

    def predict(self, X):
        """
        This method is used to predict the classification of the test data or new acquired features
        :param X: features test or newly acquired
        :return: the predicted classification of each set of features
        """
        x = X
        w = self.w_learned
        predicted_y = np.around(self.sigmoid(np.matmul(x, w)))
        return predicted_y

    def cost_computation(self, X, y, w):
        """
        This method is used to calculate the cost function for each iteration and record it
        :param X: the data used to train
        :param y: the classification used to train
        :param w: the weight parameters  calculated in that iteration
        :return: the cost
        """
        n, m = np.shape(X)
        corr = 1e-5
        q = self.sigmoid(np.matmul(X, w))
        cost = (1/n) * ((np.matmul((-y).T, np.log(q + corr)))-(np.matmul((1-y).T, np.log(1-q + corr))))
        return cost





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





