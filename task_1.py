import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def read_and_split_function(file_path: str, test_size=0.2, random_state=42):
    """
    Reads file and splits the data into training and test set
    :param file_path: relative path to the file to be read, first column of the file should contain x values, followed
    by a delimiter of a SPACE character, in the second column should be the function values of according x values
    :param test_size: the percentage of the data that should be seperated as the test data
    :param random_state: this is used to create the same output when shuffling the data
    :returns x_train: x values of the train set
            x_test: x values of the test set
            f_x_train: function values of the train set
            f_x_test: function values of the test set
    """
    # read file as a numpy array
    function = np.loadtxt(file_path)

    # split the x and f_x values from the two columns
    x = function[:, 0]
    f_x = function[:, 1]

    # split data into train and test data
    x_train, x_test, f_x_train, f_x_test = train_test_split(x, f_x, test_size=test_size, random_state=random_state)
    return x_train, x_test, f_x_train, f_x_test


def linear_fit(x, f_x, cond=1e-2):
    """
    Fits a linear function to the given x and function values
    :param x: x values
    :param f_x: function values of the given x values
    :param cond: condition for the least squares method
    :returns approximated function values of train data
             residual: error between the approximated and original function values
    """
    # solve Ax = b where A = x and b = f_x, x will contain the coefficient m of linear function
    solution = np.linalg.lstsq(x, f_x, cond)[0]
    return x @ solution, solution


def nonlinear_fit(x, f_x, num_rbf: int, eps_helper=0.05, cond=1e-2):
    """
    Fits a nonlinear function to the given x and function values
    :param x: x values
    :param f_x: function values of the given x values
    :param num_rbf: number of radial functions to be used
    :param eps_helper: helper for the bandwidth calculation
    :param cond: condition for the least squares method
    :returns approximated function values of train data
             residual: error between the approximated and original function values
    """
    # linearly space num_rbf different x_l points
    x_l_list = np.linspace(np.min(x), np.max(x), num_rbf)
    # compute epsilon similarly to Diffusion Map algorithm
    distance = np.linalg.norm(x[:, None] - x_l_list[None], axis=-1)
    eps = eps_helper * np.max(distance)
    # compute radial basis functions according to different x_ls and bandwidth
    phi_l_list = get_radial_basis_function(x_l_list, x, eps)
    # use the least squares solution to obtain coefficients of the radial basis functions
    solution = np.linalg.lstsq(phi_l_list, f_x, cond)[0]
    return phi_l_list @ solution, solution, x_l_list, eps


def get_radial_basis_function(x_l_list, x, eps):
    return np.array([np.exp(-(x_l - x) ** 2 / eps ** 2) for x_l in x_l_list]).T


def compute_function(x_l_list, x, eps, coefficient):
    phi_l_list = get_radial_basis_function(x_l_list, x, eps)
    return phi_l_list @ coefficient


def plot_approximate_function(x_test, f_x_test, x_train, f_hat, figure_save_path: str, figure_title: str,
                              save_figure=False):
    """
    Plots the approximated function on top of the test data to show the fit of the function to the dataset, prints out
    the residual of the fit and saves the figure if requested
    :param x_test: x values of the test set
    :param f_x_test: function values of the test set
    :param x_train: x values of the train set
    :param f_hat: approximated function values of train data
    :param figure_save_path: relative path to the location where the figure will be saved
    :param figure_title: title of the figure
    :param save_figure: boolean, if set to True, figure will be saved to the given figure_save_path parameter
    """
    # prepare the figure
    plt.figure()
    plt.title(figure_title)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$ / $\hat{f}(x)$")
    # scatter the test data
    plt.scatter(x_test, f_x_test)
    # show the approximated function
    x_train_sorted, f_hat_sorted = zip(*sorted(zip(x_train, f_hat)))
    plt.plot(x_train_sorted, f_hat_sorted, c="orange")
    # show the legend to label original and approximated data
    plt.legend([r"$f(x)$", r"$\hat{f}(x)$"])
    # save figure if necessary
    if save_figure:
        plt.savefig(figure_save_path)
