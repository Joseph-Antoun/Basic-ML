import numpy as np
import pandas as pd


def evaluate_acc(ts_x, ts_y, model):
    x = ts_x
    y = ts_y
    predicted_y = model.predict(y)

    err = (np.sum(abs(y - predicted_y)) / len(np.transpose(y)))
    acc = (1 - err)*100

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





