import numpy as np
import scipy


def generate_periodic_data(n):
    """
    generates a periodic data that contains n points
    :param n: number of points in data
    :return: x_k and t_k
    """
    # create linear space for time
    t_k = np.linspace(0, 2 * np.pi, n)
    # apply formula given in the exercise sheet to obtain x_k
    return np.column_stack((np.cos(t_k), np.sin(t_k))), t_k


def diffusion_map(x, num_pc):
    """
    applies diffusion map algorithm step by step
    :param x: data, on which the algorithm will be applied on
    :param num_pc: number of principal components
    :return lambda_l: eigenvalues
            phi_l   : corresponding eigenfunctions
    """
    # form a distance matrix
    d = np.linalg.norm(x[None, :, :] - x[:, None, :], axis=-1)
    # set epsilon to 5% of the diameter of the dataset
    eps = 0.05 * np.max(d)
    # form the kernel matrix
    w = np.exp(- np.square(d) / eps)
    # form the diagonal normalization matrix
    p = np.diag(np.sum(w, axis=-1))
    # normalize w to form the kernel matrix
    k = np.linalg.inv(p) @ w @ np.linalg.inv(p)
    # form the diagonal normalization matrix
    q = np.diag(np.sum(k, axis=-1))
    # form the symmetric matrix
    t_hat = np.linalg.inv(scipy.linalg.sqrtm(q)) @ k @ np.linalg.inv(scipy.linalg.sqrtm(q))
    # find the num_pc + 1 largest eigenvalues and associated eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(t_hat)
    a_l = eigenvalues[-(num_pc + 1):]
    v_l = eigenvectors[:, -(num_pc + 1):]
    # compute the eigenvalues of t_hat^{1/eps}
    lambda_l = np.sqrt(np.power(a_l, 1 / eps))
    # compute the eigenvectors ϕ_l of the matrix t
    phi_l = np.linalg.inv(scipy.linalg.sqrtm(q)) @ v_l
    return lambda_l, phi_l
