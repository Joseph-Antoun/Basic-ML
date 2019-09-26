import numpy as np
import pandas as pd
from datetime import datetime

import data_cleaning as dc
import LDA as lda
import Pr1 as logR


def evaluate_acc(ts_x, ts_y, model):
    x = ts_x
    y = ts_y
    predicted_y = model.predict(x)
    acc = float(sum(predicted_y == y))/float(len(y))
    err = 1 - acc


    return err, acc


def model_selection(ev_x, ev_y, model, k=1):
    tot_err = []
    tot_acc = []
    # avg_err = 0.0
    # avg_acc = 0.0

    if k == 1:
        n, m = np.shape(ev_x)
        ind = int((80.0*n)/100)
        tr_x = ev_x[0:ind, :]
        tr_y = ev_y[0:ind, :]
        evl_x = ev_x[ind::, :]
        evl_y = ev_y[ind::, :]

        model.fit(tr_x, tr_y)
        err, acc = evaluate_acc(evl_x, evl_y, model)
        tot_err.append(err)
        tot_acc.append(acc)

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
    X_tr_lr = np.hstack((np.ones((n_tr, 1)), X_tr_lr))
    y_train = y_train.reshape(n_tr, 1)

    n_ts, m_ts = np.shape(X_test)
    X_ts_lr = np.hstack((np.ones((n_ts, 1)), X_ts_lr))


    lr = logR.LogisticRegression(X_tr_lr, y_train, x_vars, y_var, alpha=0.03, num_iters=2000, stopping_criteria=0.001)


    lr_avg_err, lr_avg_acc = model_selection(X_tr_lr, y_train, lr, 5)
    print("Logistic Regression")
    print(lr_avg_err)
    print(lr_avg_acc)



    ld = lda.LDA(X_train, y_train)
    ld_avg_err, ld_avg_acc = model_selection(X_train, y_train, ld, 5)
    print("LDA")
    print(ld_avg_err)
    print(ld_avg_acc)

    print("Time lapsed = ", datetime.now() - start_time)

if __name__ == "__main__":
    main()




