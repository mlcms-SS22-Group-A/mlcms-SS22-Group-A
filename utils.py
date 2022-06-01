import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import Symbol, Derivative


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
    :param A: the matrix that defines the linear dynamical system
    :param X: x-coordinates of the linear system
    :param Y: y-coordinates of the linear system
    """
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title(title)
    ax0.set_aspect(1)
    return fig, ax0


def plot_bifurcation_diagram(equation, var, alpha , x_lim , numsys):
    """
    Plots a bifurcation diagram of the given equation, w.r.t var and within the given x_lim.
    :param equation: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param var: the variable that we want to solve the equation w.r.t.
    :param x_lim: The range of the plot on the x-axis for alpha variable.
    :returns: sy plot to save the figure in the notebook
    """
    # solve equation = 0 w.r.t var
    solutions = sy.solveset(equation, var)

    # plot the solutions
    first = True
    plot = None
    x= var
    for solution in solutions:
        deriv = Derivative(equation, x)
        if plot is None:
            if deriv.doit().subs({x:solution.evalf(subs={alpha:4}) }) > 0:

                plot = sy.plot(solution, line_color = "red",title="Dynamical System "+numsys, show=False, xlim=x_lim, ylim=[-1.25, 1.25], axis_center=(x_lim[0], -1.25),
                           xlabel=r"$\alpha$", ylabel=r"$x_0$", label="unstable")
            else:
                plot = sy.plot(solution, line_color="blue", title="Dynamical System "+numsys, show=False, xlim=x_lim, ylim=[-1.25, 1.25], axis_center=(x_lim[0], -1.25),
                              xlabel=r"$\alpha$", ylabel=r"$x_0$", label="stable")

        else:
            if deriv.doit().subs({x: solution.evalf(subs={alpha: 4})}) > 0:
                plot.extend(sy.plot(solution, line_color = "red", show=False, label="unstable"))
            else:
                plot.extend(sy.plot(solution, line_color="blue", show=False, label="stable"))

    plot.legend = True
    plot.show()
    return plot
