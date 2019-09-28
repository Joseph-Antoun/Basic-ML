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


def plot_cost_func(cost_fnc):
    plt.figure()
    sns.set_style('white')

    plt.plot(range(len(cost_fnc[0])), cost_fnc[0], label="\u03B1 = 0.0013, \u03BB =0.001")
    plt.plot(range(len(cost_fnc[1])), cost_fnc[1], label="\u03B1 = 0.0014, \u03BB =0.001")
    plt.plot(range(len(cost_fnc[2])), cost_fnc[2], label="\u03B1 = 0.0016, \u03BB =0.001")
    plt.plot(range(len(cost_fnc[3])), cost_fnc[3], label="\u03B1 = 0.0018, \u03BB =0.001")
    plt.plot(range(len(cost_fnc[4])), cost_fnc[4], label="\u03B1 = 0.0020, \u03BB =0.001")
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()


def main():


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

    cost_fcs = []
    start_time = datetime.now()
    lr1 = logR2.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.0013, reg_val=0, num_iters=500,
                                   stopping_criteria=1e-15)
    # lr2 = logR2.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.0014, reg_val=0.001, num_iters=500,
    #                                stopping_criteria=1e-15)
    # lr3 = logR2.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.0016, reg_val=0.001, num_iters=500,
    #                                stopping_criteria=1e-15)
    # lr4 = logR2.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.0018, reg_val=0.001, num_iters=500,
    #                                stopping_criteria=1e-15)
    # lr5 = logR2.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.0020, reg_val=0.001, num_iters=500,
    #                                stopping_criteria=1e-15)

    lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr1, 1, cost_fcs)
    # lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr2, 1, cost_fcs)
    # lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr3, 1, cost_fcs)
    # lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr4, 1, cost_fcs)
    # lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr5, 1, cost_fcs)

    # plot_cost_func(cost_fcs)

    # lr2 = logR2.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.001, reg_val=0.001, num_iters=1000,
    #                                stopping_criteria=1e-20)

    # cost_fcs2 = []
    #
    # lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr, 5, cost_fcs)
    print("Time lapsed = ", datetime.now() - start_time)
    print("Logistic Regression")
    print("Average Error on validation: ", lr_avg_err)
    print("Average Accuracy on validation: ", lr_avg_acc)
    #
    err, acc = evaluate_acc(X_ts_lr, y_test, lr1)
    print("Average Test Error: ", err)
    print("Average Test Error: ", acc)
    #
    # lr_avg_err2, lr_avg_acc2 = model_selection(X_tr_lr, y_train, lr2, 5, cost_fcs2)
    # print("Logistic Regression2")
    # print("Average Error on validation with regularization: ", lr_avg_err2)
    # print("Average accuracy on validation with regularization: ", lr_avg_acc2)
    #
    # err2, acc2 = evaluate_acc(X_ts_lr, y_test, lr2)
    # print("Average Test Error with regularization: ", err2)
    # print("Average Test Error with regularization: ", acc2)
    #
    # print("Time lapsed = ", datetime.now() - start_time)


if __name__ == "__main__":
    main()




