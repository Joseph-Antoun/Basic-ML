# coding: utf-8
import numpy as np
import pandas as pd
import sys

# Sklearn functions for validation purposes
from sklearn.model_selection import train_test_split as sklearn_train_test_split

# Import custom data cleaning module
import data_cleaning



def compute_y_column_wine(clean_data):
    """
    Transform 'quality'[0,1,...,10] => 'y' [0,1]
    Wine dataset only
    """
    # this is to skip SettingWithCopyWarning from Pandas
    clean_df = clean_data.copy()
    # Create the binary y column
    clean_df['y'] = np.where(clean_data['quality'] >= 6.0, 1.0, 0.0)
    # Drop the 'quality' column
    return clean_df.drop('quality', axis=1)


def validate_train_test_split(clean_data, x_vars, y_var):
    """
    Validates our custom train_test_split function against the
    one from sklearn
    """
    X_train,X_test,y_train,y_test = data_cleaning.train_test_split(
        clean_data, 
        x_vars, y_var,
        train_ratio=0.8,
        random_seed=42)

    # Sklearn's function required a numpy array
    X = clean_data[x_vars].to_numpy()
    y = clean_data[y_var].to_numpy()

    X_train_sk,X_test_sk,y_train_sk,y_test_sk = sklearn_train_test_split(
        X, y, train_size=0.8, random_state=42)

    print("custom X_train")
    print(X_train)
    print("sklearn X_train")
    print(X_train_sk)
    print(X_train.shape)
    print(X_train_sk.shape)


def main():

    # Load the wine dataset and clean it
    file_path   = '../data/wine/winequality-red.csv'
    raw_data    = pd.read_csv(file_path, delimiter=';')
    clean_data  = data_cleaning.get_clean_data(raw_data, verbose=False)
    clean_data  = compute_y_column_wine(clean_data)

    # Split between X and y and create the numpy arrays
    x_vars = clean_data.columns.tolist()
    y_var  = 'y'
    x_vars.remove(y_var)
    print("Input variables: %s" % x_vars)
    print("Variable to predict: %s" % y_var)

    #--------------------------------------------------------------------------
    # Train test split
    #--------------------------------------------------------------------------
    X_train,X_test,y_train, y_test = data_cleaning.train_test_split(
            clean_data,
            x_vars,
            y_var,
            random_seed=42)
    # Validate our custom function
    validate_train_test_split(clean_data, x_vars, y_var) 



if __name__ == "__main__":
    main()

