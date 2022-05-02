import numpy as np

def parser(X):
    """
    Parses the given state matrix and returns the number of rows, columns and
    the coordinates of the pedestrians, obstacles and targets.

    :param X: state of the cellular automata as matrix of {0,1,2,3}
              which indicate {E,P,O,T} respectively
    :returns: rows, columns, pedestrians, obstacles, targets
    """
    rows, cols = X.shape
    pedestrians = np.argwhere(X == 1)
    obstacles = np.argwhere(X == 2)
    targets = np.argwhere(X == 3) # TODO support multiple targets

    numTargets, _ = targets.shape

    if numTargets != 1:
        raise ValueError("Our implementation does not support multiple targets yet.")

    return rows, cols, pedestrians, obstacles, targets
