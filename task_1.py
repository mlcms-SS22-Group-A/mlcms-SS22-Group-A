from random import randint
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def dataloader(dir):
    """

    :param dir: directory of text file
    :return: return np array of data
    """
    return np.loadtxt(dir)

def approximate_lstsq(X):
    """

    :param X: given data with x and y coordinates
    :return: solution vector
    """
    x = X[:, 0]
    y = X[:, 1]
    M = x[:, np.newaxis]
    p, res, rnk, s = lstsq(M, y)
    return p

def process_data_and_show(raw_data, approximation):
    """

    :param raw_data: raw data matrix
    :param approximation: approximation matrix
    :return:
    """
    plt.plot(raw_data[:, 0], raw_data[:, 1], 'o', label='data')
    p = approximation
    xx = np.linspace(np.amin(raw_data), np.amax(raw_data), 101)
    yy = p[0] * xx
    plt.plot(xx, yy, label='least squares fit, $y = Ax$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(framealpha=1, shadow=True)
    plt.grid(alpha=0.25)
    plt.show()


def get_radial_function(x, epsilon):
    """

    :param x: given data point, creates a radial basis function
    :param epsilon: epsilon value, free searched
    :return: returns the found basis function with created index l
    """
    max_lim = x.shape[0]
    ind = randint(0, max_lim)
    column = np.empty(x.shape)
    for i in range(0, x.shape[0]):
        column[i] = np.exp(-np.linalg.norm(x[ind] - x[i]) / (2 * (epsilon ** 2)))
    return column, ind

def evaluate_func(xx, xl, epsilon):
    """

    :param xx: sample space
    :param xl: the x value with which basis func created
    :param epsilon:  epsilon
    :return: evaluated result
    """
    y = np.empty(xx.shape)
    for i in range(0, xx.shape[0]):
        y[i] = np.exp(-np.linalg.norm(xl - xx[i]) / ( 2 * (epsilon ** 2)))
    return y


def approximate_with_radial_basisfunc(X, L=3, epsilon=0.5):
    """

    :param X: input data
    :param L: number of basis functions
    :param epsilon: epsilon
    :return: plots the approximation with data itself.
    """
    x = X[:, 0]
    y = X[:, 1]
    columns = np.empty((L, x.shape[0]))
    indices = np.empty(L)

    for i in range(0, L):
        column, ind = get_radial_function(x, epsilon)
        columns[i] = column
        indices[i] = ind

    M = columns.T
    p, res, rnk, s = lstsq(M, y)

    L = p.shape[0]

    xx = np.linspace(np.amin(x), np.amax(x), 101)
    yy = np.zeros(xx.shape)
    for i in range(0, L):
        ind = int(indices[i])

        yy += p[i] * evaluate_func(xx, x[ind], epsilon)

    plt.plot(X[:, 0], X[:, 1], 'o', label='data')
    plt.plot(xx, yy, label='least squares fit, with radial basis functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(framealpha=1, shadow=True)
    plt.grid(alpha=0.25)
    plt.show()
