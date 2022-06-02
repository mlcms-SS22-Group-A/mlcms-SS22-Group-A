import numpy as np
from utils import plot_phase_portrait


def task_1_plot_reconstructions(X, Y, alphas, save_figure):
    """
    Plots the phase portraits given in the figre 2.5 in the book of Kuznetsov 
    also described in the exercise sheet.
    :param  X: notebook parameter X 
    :param Y: notebook parameter Y
    :param alphas: alpha variables of the parameterized matrix given on the exercise sheet. 
                    A_alpha = [ [alpha, (-)alpha], [-0.25, 0] ]
    :param save_figure: boolean parameter, saves the figure if True
    """
    if len(alphas) != 5:
        raise ValueError("alphas should include an alpha for all the 5 constructions!")
    # define titles for different cases
    titles = ["Stable node", "Stable focus", "Unstable saddle", "Unstable focus", "Unstable node"]
    # since more than one parametrized matrix is used, this list helps to differentiate between them
    helpers = [-1, -1, -1, 1, 1]

    # loop over the five different cases and plot them
    for (alpha, title, helper) in zip(alphas, titles, helpers):
        # create the matrix according to the case
        A = np.array([[alpha, helper * alpha], [-.25, 0]])
        # get the eigenvalues of the matrix
        eigenvalues = np.linalg.eigvals(A)

        # add parameter value to the title
        title += "\n\u03B1 = " + str(alpha)
        if helper == -1:
            title += ", " + "\u03B2 = " + str(alpha * -1)
        title += "\n"
        # subscript for multiple eigenvalues
        subscript = ord('\u2081')
        # loop over the list of eigenvalues and add their values to title
        for idx, eig in enumerate(eigenvalues):
            title += "\u03BB" + chr(subscript) + " = " + "{:.2f}".format(eig)
            if idx < len(eigenvalues) - 1:
                title += ", "
            subscript += 1

        # plot the figure
        fig, ax0 = plot_phase_portrait(A, X, Y, title)
        ax0.set_aspect(1)
        if save_figure:
            fig.savefig(title + ".pdf")
