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
    """
    :param X: input data matrix X of shape (N,D)
    :param k: the reduced dimension space 
    :returns: reduced data matrix X_reduced of shape (N,k)
    """
    # apply SVD 
    U, S, Vh = np.linalg.svd(X)
    N,D = X.shape
        
    # number of columns that we want to truncate
    trunc_cols = D - k
    
    # the lower dimension representation should be at least 1 
    assert trunc_cols > 0
    
    U_trunc = U[:,:-trunc_cols]
    S_trunc = S[:-trunc_cols]
    
    X_reduced = U_trunc[:, :k] * S_trunc
    
    return X_reduced

def get_contained_energy_in_each_component(X):
    """
    :param X: input data matrix X of shape (N,D)
    :returns: the energy contained in each principal component
    """
    # apply SVD 
    U, S, Vh = np.linalg.svd(X)
    N,D = X.shape
    
    S_squared = S * S
    components_sum = np.sum(S_squared)
    return S_squared / components_sum

def approximate_pca(X, k):
    """
    :param X: input data matrix X of shape (N,D)
    :param k: the reduced dimension space size
    :returns: approximated data matrix X_reduced of shape (N,D) using only k - principal components
    """
    # apply SVD 
    U, S, Vh = np.linalg.svd(X)
    N,D = X.shape
        
    # number of columns that we want to truncate
    trunc_cols = D - k
    
    # the lower dimension representation should be at least 1 
    assert trunc_cols >= 0
    
    if trunc_cols == 0:
        X_approxed = np.dot(U[:, :D] * S, Vh)
        return X_approxed
    
    S_shaped_zeros = np.zeros(trunc_cols, dtype=np.float64)
    S_trunc = S[:-trunc_cols]
    S_ = np.concatenate((S_trunc, S_shaped_zeros), axis=0)
    
    X_approxed = np.dot(U[:, :D] * S_, Vh)
    return X_approxed

def get_energy_loss(X, k):
    """
    :param X: input data matrix X of shape (N,D) 
    :param k: the reduced dimension space size
    :returns: the energy loss in percentage 
    """
    N,D = X.shape
    contained_energy_in_each_component = get_contained_energy_in_each_component(X)
    kept_energy = np.sum(contained_energy_in_each_component[:k])
    return 1 - kept_energy