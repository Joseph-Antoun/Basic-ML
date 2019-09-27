# coding: utf-8
import numpy as np
import pandas as pd
import sys

# Sklearn functions for validation purposes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Sklean_LDA

# Import custom data cleaning module
import data_cleaning
import data_visualization
import LDA as Custom_LDA



        

def main():

    # Load the wine dataset and clean it
    file_path   = '../data/wine/winequality-red.csv'
    raw_data    = pd.read_csv(file_path, delimiter=';')
    clean_data  = data_cleaning.get_clean_data(raw_data, verbose=False)
    clean_data  = data_cleaning.compute_y_column_wine(clean_data)

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

    #--------------------------------------------------------------------------
    # LDA
    #--------------------------------------------------------------------------
    # Sklearn version of LDA
    sklearn_lda = Sklean_LDA()
    sklearn_lda.fit(X_train, y_train)
    sklearn_pred = sklearn_lda.predict(X_test)

    # Custom version of LDA
    custom_lda = Custom_LDA.LDA(X_train, y_train)
    custom_lda.fit(X_train, y_train)
    custom_pred = custom_lda.predict(X_test)



if __name__ == "__main__":
    main()

