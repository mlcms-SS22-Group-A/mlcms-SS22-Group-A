import numpy as np


def read_file(filename, dtype):
    """
    Reads the given file and returns a matrix which contains the cells for the crowd simulation with entries in space
    {E,P,O,T} (E := empty, P := pedestrian, O := obstacle, T := target)
    :param filename: the name of the file to read the initial state
    :param dtype: the type of data that is contained in the given file
    :returns: numpy array of the initial cellular automaton state
    """
    return np.loadtxt(filename, dtype=dtype, delimiter=',')
