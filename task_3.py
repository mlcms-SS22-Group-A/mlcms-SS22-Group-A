import matplotlib.pyplot as plt


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
        if save_figure:
            fig.savefig("/figures/task3_andronov_hopf_alpha_" + str(alpha) + ".pdf")
