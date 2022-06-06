import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# this is done to show a background grid while using sympy.plot
plt.style.use('ggplot')


def logistic_map(r_, x_n):
    """
    This is the logistic map update scheme, right side of ODE
    :param r_: parameter of the dynamic system
    :param x_n: current state
    :return x_(n+1): updated state
    """
    return r_ * x_n * (1 - x_n)


def lorenz_attractor(state, t, sigma, rho, beta):
    """
    This method calculates dx/dt, dy/dt, dz/dt ODEs
    :param state: tuple containing the spatial coordinates of current state
    :param t: time
    :param sigma: parameter of dynamic system
    :param rho: parameter of dynamic system
    :param beta: parameter of dynamic system
    :return:
    """
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def plot_logistic_map(iter_count):
    """
    This method samples 25 points for 4000 different r values each of those points is updated iter_count times using
    the logistic_map() method, then they are added to a list so that they can be plotted afterwards
    :param iter_count: Number of logistic_map calls for each of the randomly sampled point x
    """
    # sample 4000 r values between 0 and 4 linearly
    r_list = np.linspace(0.0, 4.0, 4000)
    # helper lists for plotting
    r_plot = []
    x_list = []

    for r in r_list:
        # create 25 random x points between 0 and 1
        x0 = np.random.random(25)
        # apply them the logistic map updating iter_count times
        for _ in range(iter_count):
            x0 = logistic_map(r, x0)
        # save the results into the list
        for x in x0:
            r_plot.append(r)
            x_list.append(x)

    # plot the bifurcation map
    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.set_xlabel("r")
    ax0.set_ylabel("x")
    ax0.scatter(r_plot, x_list, s=0.01)
    ax0.set_title("Iteration=" + str(iter_count))


def plot_lorenz_attractor(sigma, rho, beta, plot):
    """
    This method plots two different trajectories of Lorenz system, one with starting position [10.0, 10.0, 10.0],
    other one [10.0 + 10 ** (-8), 10.0, 10.0]. To do that we first solve lorenz systems ODE using the given parameters
    and starting points, then use the result to plot trajectories.
    :param sigma: parameter of dynamic system
    :param rho: parameter of dynamic system
    :param beta: parameter of dynamic system
    :param plot: if set to True, plot() method is used for displaying the trajectories, this feature is added, since
    the lorenz attractor (rho = 28) does not create visually reasonable and easy to understand outputs when plot()
    method is used
    :return states_diff: array holding the L2-distance between trajectories over time
                      t: time-steps stored in an array
    """
    # initialize two different starting positions
    state0 = [10.0, 10.0, 10.0]
    state0_ = [10.0 + 10 ** (-8), 10.0, 10.0]
    # end time
    t_end = 1000.0
    # time is represented by a linear space between 0 and time_end, there are 100000 time-steps, meaning each time-step
    # is 0.01 seconds
    t = np.linspace(0.0, t_end, 100000)

    # solve ODEs using scipy
    states = scipy.integrate.odeint(lorenz_attractor, state0, t, args=(sigma, rho, beta))
    states_ = scipy.integrate.odeint(lorenz_attractor, state0_, t, args=(sigma, rho, beta))

    # calculate L2-distance between trajectories for each time-step
    states_diff = np.sum(np.power(states - states_, 2), axis=1)

    # plot trajectories in different subplots
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
    
    # return these so that the difference can be plotted against the time in later tasks
    return states_diff, t
