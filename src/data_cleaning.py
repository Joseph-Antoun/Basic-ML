# coding: utf-8
import numpy as np
import pandas as pd
import sys


def isFloat(string):
    """
    Checks if string is a float
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def flag_not_float(df, column):
    """
    flag for non numerical values
    """
    for i in df.index:
        if not isFloat(df[column][i]):
            df.at[i, 'flag'] = 'not a float'
    return df


def flag_duplicates(df):
    """
    Identify potential duplicate rows
    """
    duplicate_rows_df = df[df.duplicated()]
    if duplicate_rows_df.shape[0] == 0:
        print("No duplicates found in dataframe")
        return df
    # If we reach this part of the function then we do have duplicates in the data
    for i in duplicate_rows_df.index:
        df.at[i, 'flag'] = 'duplicate'
    return df


def flag_outliers(df, columns_to_clean, n_std=3):
    """
    n_std: number of standard deviations above which a value is considered to be an outlier
    If a given row will be flagged if it contains at least one outlier
    """
    for col in columns_to_clean:
        # sumamry statistics
        col_mean, col_std = np.mean(df[col]), np.std(df[col])
        # identify outliers
        cut_off = col_std * n_std
        # this is the range of acceptable data
        lower, upper = col_mean - cut_off, col_mean + cut_off
        # outliers
        outliers = df[(df[col]<lower) | (df[col]>upper)]
        for i in outliers.index:
            df.at[i, 'flag'] = 'potential outlier'
    return df


def flag_data(raw_data):
    """
        The different flags will be:
        - 'ok' (no issue found with row)
        - 'not a float'
        - 'duplicate'
        - 'potential outlier'
    """ 
    # by default, flag everything as OK, this may change as problems are found
    columns_to_clean = raw_data.columns
    raw_data['flag'] = 'ok'

    # Test that all columns are made of float values
    for column in columns_to_clean:
        raw_data = flag_not_float(raw_data, column)

    # Flag duplicated data if any
    raw_data = flag_duplicates(raw_data)
    # Flag potential outliers
    raw_data = flag_outliers(raw_data, columns_to_clean)
    return raw_data


def get_clean_data(raw_data, delete_outliers=False, verbose=True):
    """
    Flag potential errors and delete them. 
    Deleting outliers is optionnal
    """

    # Drop rows containing missing values first
    if verbose:
        print("Number of missing values in the data\n%s\n" % raw_data.isna().sum())
    raw_data = raw_data.dropna()

    # First flag the data
    flagged_data = flag_data(raw_data)

    if verbose:
        # Summary of what as flagged
        print("SUMMARY OF DATA FLAGGING")
        print(raw_data.groupby('flag').count()['quality'])

    # Then clean the data
    if delete_outliers is True:
        clean_data = flagged_data[flagged_data['flag']=='ok']
    else:
        clean_data = flagged_data[(flagged_data['flag']=='ok') | (flagged_data['flag']=='potential outlier')]

    # Drop the flag column before returning the clean data
    return clean_data.drop('flag', axis=1)


def main():
    
    file_path   = '../data/wine/winequality-red.csv'
    raw_data    = pd.read_csv(file_path, delimiter=';')
    clean_data  = get_clean_data(raw_data)

    print(clean_data)


if __name__ == "__main__":
    main()



