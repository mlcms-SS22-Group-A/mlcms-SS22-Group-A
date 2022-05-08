import numpy as np


def parser(x):
    """
    parses the given state matrix and returns the number of rows, columns and the coordinates of the pedestrians,
    obstacles and targets
    :param x: state of the cellular automaton as a matrix of {0,1,2,3} which indicate {E,P,O,T} respectively
    :returns: rows, columns, pedestrians, obstacles, targets
    """
    rows, cols = x.shape
    pedestrians = np.argwhere(x == 1)
    obstacles = np.argwhere(x == 2)
    targets = np.argwhere(x == 3)

    num_targets, _ = targets.shape

    if num_targets != 1:
        raise ValueError("Our implementation does not support multiple targets.")

    return rows, cols, pedestrians, obstacles, targets
