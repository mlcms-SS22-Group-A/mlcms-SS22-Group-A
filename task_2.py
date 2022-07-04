import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


def read_data(file_path):
    """
    Reads data and returns it as a numpy array
    @param file_path: Path to the file
    @return: Context of the file saved as a numpy array
    """
    return np.loadtxt(file_path)


def plot_phase_portrait_and_trajectory(A, X, Y, title, sol=None, figure_save_path="", save_figure=False):
    """
    Plots phase portrait and a trajectory on top of it
    @param A: Matrix that defines the evolution operator
    @param X: Meshgrid
    @param Y: Meshgrid
    @param title: Title of the plot
    @param sol: Contains trajectory data to be plotted
    @param figure_save_path: Path where the figure will be saved
    @param save_figure: boolean, if set to True, plotted figure will be saved
    """
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)

    # Plot the figure
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot()
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    if sol is not None:
        ax0.plot(sol.y[0], sol.y[1])
    ax0.set_title(title)
    ax0.set_aspect(1)
    if save_figure:
        fig.savefig(figure_save_path)


def mse(x_truth, x_prediction):
    """
    Calculates the mse loss between the ground truth data and the prediction
    @param x_truth: Ground truth data
    @param x_prediction: Predicted/approximated data
    @return: MSE loss
    """
    return np.mean(np.sum((x_truth - x_prediction)**2, axis=-1))


def search_for_optimal_dt(dt_list, linear_vectorfield_x0, linear_vectorfield_x1):
    """
    Searches for an optimal timestep that minimizes the approximation error
    @param dt_list: list of timesteps, where the minimum error will be searched
    @param linear_vectorfield_x0: data x0
    @param linear_vectorfield_x1: data x1
    @return err_list: list containing the mse error for a specific timestep (order is
                      same as dt_list)
            dt: the timestep in dt_list that caused the minimal approximation error
            min_idx: the index of dt in dt_list
            A_t: linear approximation of the vector-field
    """
    err_list = []
    A_t = None

    for dt in dt_list:
        # Compute vectors
        v_k = (linear_vectorfield_x1 - linear_vectorfield_x0) / dt
        # Approximate vectorfield
        A_t, residual = np.linalg.lstsq(linear_vectorfield_x0, v_k, rcond=1e-5)[:2]

        # Estimate the x1 points by solving the linear system for each point in x0
        f_hat = np.zeros(linear_vectorfield_x0.shape)
        for i in range(linear_vectorfield_x0.shape[0]):
            f_hat[i, :] = solve_ivp(lambda t, x: x @ A_t, t_span=(0.0, 0.1), t_eval=[0.1],
                                    y0=linear_vectorfield_x0[i, :]).y.T

        # compute mse error
        err_list.append(mse(linear_vectorfield_x1, f_hat))

    # find the dt that produces minimum approximation error
    min_idx = np.argmin(np.array(err_list))
    dt = dt_list[min_idx]
    return err_list, dt, min_idx, A_t


def solve_linear_system_and_plot(A_t, dt, initial_location, figure_save_path="", save_figure=False):
    """
    Solve linear system dx/dt = Ax using the calculated timestep and the initial location, then plot
    the phase portrait and the trajectory
    @param A_t: linear approximation of the vectorfield
    @param dt: timestep
    @param initial_location: initial location of the trajectory
    @param figure_save_path: Path where the figure will be saved
    @param save_figure: boolean, if set to True, plotted figure will be saved
    """
    # Solve the linear system
    sol = solve_ivp(lambda t, x: x @ A_t, t_span=(0, 100), t_eval=np.arange(0, 100, dt), y0=initial_location)

    # Plot phase portrait and trajectory
    Y, X = np.mgrid[-10.5:10.5:0.1, -10.5:10.5:0.1]
    plot_phase_portrait_and_trajectory(A_t.T, X, Y, "", sol, figure_save_path, save_figure)
