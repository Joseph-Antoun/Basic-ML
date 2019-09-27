import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


import data_cleaning as dc
import Pr1 as logR
import LogisticRegression as logR2


def evaluate_acc(ts_x, ts_y, model):
    """
    This method is used to evaluate the accuracy and error of the trained model.
    :param ts_x: features data to test/ evaluate our model
    :param ts_y: the real classification of the test data
    :param model: the model under evaluation
    :return: acc: accuracy of the model, err: the error of the model
    """
    x = ts_x
    y = ts_y
    predicted_y = model.predict(x)
    acc = float(sum(predicted_y == y))/float(len(y))
    err = 1 - acc


    return err, acc


def model_selection(ev_x, ev_y, model, k=1, costs=[]):
    """
    This method is used for model selection, it applies K-Fold cross validation.
    :param ev_x: The features data used to train
    :param ev_y: the classification data used to train
    :param model: The model that need to apply the k-fold for
    :param k: the number of folds, if k=1 it will train and validate on the same data
    :param costs: the list that will hold the cost function results for each fold.
    :return: return the average error and accuracy of the model with k-fold
    """
    tot_err = []
    tot_acc = []

    if k == 1:
        n, m = np.shape(ev_x)
        tr_x = ev_x
        tr_y = ev_y

        model.fit(tr_x, tr_y)
        err, acc = evaluate_acc(tr_x, tr_y, model)
        tot_err.append(err)
        tot_acc.append(acc)
        costs.append(model.h_cost)

    elif k > 1:
        x = np.array_split(ev_x, k)
        y = np.array_split(ev_y, k)

        for i in range(0, k):
            temp_x = x.copy()
            temp_y = y.copy()
            x_vl = x[i]
            y_vl = y[i]
            del(temp_x[i])
            del(temp_y[i])
            x_tr = np.concatenate(temp_x)
            y_tr = np.concatenate(temp_y)
            model.fit(x_tr, y_tr)
            err, acc = evaluate_acc(x_vl, y_vl, model)
            tot_err.append(err)
            tot_acc.append(acc)
            costs.append(model.h_cost)

    avg_err = sum(tot_err)/len(tot_err)
    avg_acc = sum(tot_acc)/len(tot_acc)

    return avg_err, avg_acc


def main():
    start_time = datetime.now()

    file_path = '../data/wine/winequality-red.csv'
    raw_data = pd.read_csv(file_path, delimiter=';')
    clean_data = dc.get_clean_data(raw_data)

    # this is to skip SettingWithCopyWarning from Pandas
    clean_df = dc.get_clean_data(raw_data, verbose=False)
    clean_data = clean_df.copy()
    # Create the binary y column
    clean_data['y'] = np.where(clean_df['quality'] >= 6.0, 1.0, 0.0)
    # Drop the 'quality' column as it shouldn't be used to predict the wine binary rating
    clean_data.drop('quality', axis=1, inplace=True)

    # Split between X and y and create the numpy arrays
    y_var = 'y'
    x_vars = [var for var in clean_data.columns.tolist() if not var in y_var]
    X_train, X_test, y_train, y_test = dc.train_test_split(clean_data, x_vars, y_var)

    np.seterr(over='ignore')

    X_tr_lr = X_train
    X_ts_lr = X_test

    n_tr, m_tr = np.shape(X_train)
    n_ts, m_ts = np.shape(X_test)

    X_tr_lr = np.hstack((np.ones((n_tr, 1)), X_tr_lr))
    X_ts_lr = np.hstack((np.ones((n_ts, 1)), X_ts_lr))

    y_train = y_train.reshape(n_tr, 1)
    y_test = y_test.reshape(n_ts, 1)



    lr = logR.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.0023, num_iters=1500,
                                 stopping_criteria=1e-15)
    lr2 = logR2.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.001, reg_val=0.001, num_iters=1000,
                                   stopping_criteria=1e-20)
    cost_fcs = []
    cost_fcs2 = []

    lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr, 5, cost_fcs)
    print("Logistic Regression")
    print(lr_avg_err)
    print(lr_avg_acc)

    err, acc = evaluate_acc(X_ts_lr, y_test, lr)
    print(err)
    print(acc)

    lr_avg_err2, lr_avg_acc2 = model_selection(X_tr_lr, y_train, lr2, 5, cost_fcs2)
    print("Logistic Regression2")
    print(lr_avg_err2)
    print(lr_avg_acc2)


    print("Time lapsed = ", datetime.now() - start_time)
    plt.figure(1)
    sns.set_style('white')
    plt.plot(range(len(cost_fcs[0])), cost_fcs[0])
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.figure(2)
    sns.set_style('white')
    plt.plot(range(len(cost_fcs2[0])), cost_fcs2[0])
    plt.title("Convergence Graph of Cost Function2")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()





if __name__ == "__main__":
    main()




