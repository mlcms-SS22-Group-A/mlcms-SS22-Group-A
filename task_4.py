import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# this is done to show a background grid while using sympy.plot
plt.style.use('ggplot')


def logistic_map(r_, x_n):
    """

    :param r_:
    :param x_n:
    :return:
    """
    return r_ * x_n * (1 - x_n)


def lorenz_attractor(state, t, sigma, rho, beta):
    """

    :param state:
    :param t:
    :param sigma:
    :param rho:
    :param beta:
    :return:
    """
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def plot_logistic_map(iter_count):
    """

    :param iter_count:
    :return:
    """
    r_list = np.linspace(0.0, 4.0, 4000)
    r_plot = []
    x_list = []

    for r in r_list:
        x0 = np.random.random(25)
        for _ in range(iter_count):
            x0 = logistic_map(r, x0)
        for x in x0:
            r_plot.append(r)
            x_list.append(x)

    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.set_xlabel("r")
    ax0.set_ylabel("x")
    ax0.scatter(r_plot, x_list, s=0.01)
    ax0.set_title("Iteration=" + str(iter_count))
    fig.savefig("./figures/task3-logistic_map_" + str(iter_count) + ".pdf")


def plot_lorenz_attractor(sigma, rho, beta, plot):
    """

    :param sigma:
    :param rho:
    :param beta:
    :param plot:
    :return:
    """
    state0 = [10.0, 10.0, 10.0]
    state0_ = [10.0 + 10 ** (-8), 10.0, 10.0]
    t_end = 1000.0
    t = np.linspace(0.0, t_end, 100000)

    states = scipy.integrate.odeint(lorenz_attractor, state0, t, args=(sigma, rho, beta))
    states_ = scipy.integrate.odeint(lorenz_attractor, state0_, t, args=(sigma, rho, beta))

    states_diff = np.sum(np.power(states - states_, 2), axis=1)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax_ = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax_.set_xlabel("x")
    ax_.set_ylabel("y")
    ax_.set_zlabel("z")
    if plot:
        ax.plot(states[:, 0], states[:, 1], states[:, 2], label=r"$x_0$")
        ax_.plot(states_[:, 0], states_[:, 1], states_[:, 2], label=r"$\hat{x}_0$", color="orange")
        ax.legend()
        ax_.legend()
    else:
        ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=t, cmap="autumn", s=0.5)
        ax_.scatter(states_[:, 0], states_[:, 1], states_[:, 2], c=t, cmap="winter", s=0.5)
        ax.set_title(r"Initial position $x_0 = (10.0, 10.0, 10.0)$")
        ax_.set_title(r"Initial position $\hat{x}_0 = (10.0 + 10^{-8}, 10.0, 10.0)$")

    return states_diff, t
