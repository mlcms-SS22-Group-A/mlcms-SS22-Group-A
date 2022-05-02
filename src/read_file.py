import numpy as np

def read_file(filename):
    """
    Reads the given file and returns a matrix which contains the cells
    for the crowd simulation with entries in space {E,P,O,T}
    E := empty, P := pedestrian, O := obstacle, T := target

    :param filename: the name of the file to read the initial state
    :returns: numpy array of the initial cellular automata state
    """
    return np.loadtxt(filename, dtype='i', delimiter=',')
