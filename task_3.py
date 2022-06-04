import matplotlib.pyplot as plt
import scipy.integrate
import sympy as sy
import numpy as np

# this is done to show a background grid while using sympy.plot
plt.style.use('ggplot')


def andronov_hopf(alpha, x1, x2):
    """
    Andronov-Hopf Bifurcation 
    d/dt x1 = alpha * x1 − x2 − x1 * (x1**2 + x2**2)
    d/dt x2 = x1 + alpha * x2 - x2 * (x1**2 + x2**2)
    :param alpha: Parameter of the equation 
    :param x1: First dimension input 
    :param x2: Second dimension input
    :returns: (d/dt x1, d/dt x2) time derivatives of the dimensions
    """
    return alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2), x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)


def task_3_plot_andronov_hopf_phase_diagrams(X, Y, alphas, save_figure):
    """
    Plots the phase portraits of the Andronov-Hopf with different alpha values.
    :param X: notebook parameter X
    :param Y: notebook parameter Y
    :param alphas: parameters that we want to evaluate and plot the phase diagrams
    :param save_figure: boolean, if True the figures are saved 
    """
    # loop over the three different alpha values and plot them
    for alpha in alphas:
        # instead of creating matrix alpha fill the U and V values directly with the given equations
        U, V = andronov_hopf(alpha, X, Y)

        # add parameter value to the title
        title = "\u03B1 = " + str(alpha) + "\n"

        # plot the figure
        fig = plt.figure(figsize=(10, 10))
        ax0 = fig.add_subplot(1, 1, 1)
        ax0.streamplot(X, Y, U, V, density=[0.5, 1], color="b")
        ax0.set_title(title)
        ax0.set_aspect(1)

        # save figure if according parameter is set
        if save_figure:
            fig.savefig("./figures/task3_andronov_hopf_alpha_" + str(alpha) + ".pdf")


def plot_andronov_hopf_orbits(start_position_1, start_position_2):
    # solve the ODEs w.r.t to time and position
    sol = scipy.integrate.solve_ivp(lambda t, y: andronov_hopf(1.0, y[0], y[1]), (0, 10), start_position_1)
    sol_ = scipy.integrate.solve_ivp(lambda t, y: andronov_hopf(1.0, y[0], y[1]), (0, 10), start_position_2)

    # plot both trajectories in 3D
    fig = plt.figure(figsize=(10, 10))
    ax0 = plt.axes(projection="3d")
    ax0.plot(sol.t, sol.y[0], sol.y[1], label=r"Trajectory with starting point $(2, 0)$", color="r")
    ax0.plot(sol_.t, sol_.y[0], sol_.y[1], label=r"Trajectory with starting point $(0.5, 0)$", color="y")
    ax0.set_xlabel(r"$t$")
    ax0.set_ylabel(r"$x_1$")
    ax0.set_zlabel(r"$x_2$")
    ax0.legend()
    ax0.set_aspect(aspect="auto")


def plot_cusp_bifurcation():
    x = sy.symbols("x")
    # create sample points (alpha_1, alpha_2)
    sample_points = [(x_, y_) for x_ in np.arange(-5.0, 5.0, 0.5) for y_ in np.arange(-5.0, 5.0, 0.5)]

    # prepare a 3d plot (alpha_1, alpha_2 bottom plane, x third direction)
    plt.figure(figsize=(10, 10))
    ax0 = plt.axes(projection='3d')
    ax0.set_xlabel(r"$\alpha_1$")
    ax0.set_ylabel(r"$\alpha_2$")
    ax0.set_zlabel("x")

    # for each sample point solve equation = 0 w.r.t x and plot it
    for idx, _ in enumerate(sample_points):
        solution = sy.solveset(sample_points[idx][0] + sample_points[idx][1] * x - x ** 3, x)
        ax0.scatter(sample_points[idx][0], sample_points[idx][1], list(solution)[0], color="blue")
    ax0.view_init(azim=95)
