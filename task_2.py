import sympy as sy
import matplotlib.pyplot as plt
from sympy import Derivative

# this is done to show a background grid while using sympy.plot
plt.style.use('ggplot')

def task_2_plot_bifurcation_diagram(equation, var, alpha , x_lim , numsys):
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