import matplotlib.pyplot as plt
import numpy as np

# this is done to show a background grid while using sympy.plot
plt.style.use('ggplot')


def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1] - time[0]
    for k in range(1, len(time)):
        yt[k, :] = yt[k - 1, :] + step_size * f_ode(yt[k - 1, :])
    return yt, time


def plot_phase_portrait(A, X, Y, title):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    :param title:
    :param A: the matrix that defines the linear dynamical system
    :param X: x-coordinates of the linear system
    :param Y: y-coordinates of the linear system
    """
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)

    fig = plt.figure(figsize=(10, 10))
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title(title)
    ax0.set_aspect(1)
    return fig, ax0
