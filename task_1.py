import pandas as pd
import numpy as np

def read_file(filename, column, header):
    """
    Reads the given 'pca_dataset.txt' file and returns a grouped pandas object, grouped by ""

    :param filename: name of the file as string to read from
    :param column: column index (0-based) to read
    :param header: the header that the user wants to assign to the column that is read
    :returns: reads and returns the given column from the file 

    """
    data = pd.read_csv(filename, delimiter=" ", usecols=[column], header=None, names=[header])
    return data

def center_data(X):
    """
    :param X: data matrix of shape (N,D)
    :returns: Centered data matrix X_centered of shape (N,D)
    """
    N, D = X.shape
    
    # calculate the mean of every dimension
    x_mean = (X.T @ np.ones(N)) / N
    
    # substract this mean which centers the data
    X_centered = X - x_mean
    return X - np.mean(X, axis=0)
    #assert np.isclose(X_centered, X - np.mean(X, axis=0)) == True
    #return X_centered

def reduce_dim_pca(X, k):
    # apply SVD 
    U, S, Vh = np.linalg.svd(X)
    N,D = X.shape
        
    # number of columns that we want to truncate
    trunc_cols = D - k
    
    # the lower dimension representation should be at least 1 
    assert trunc_cols > 0
    
    U_trunc = U[:,:trunc_cols]
    S_trunc = S[:trunc_cols]
    
    X_trunc = U_trunc[:, :trunc_cols] * S_trunc
    
    return X_trunc