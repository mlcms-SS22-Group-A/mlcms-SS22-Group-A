from queue import PriorityQueue

import numpy as np

from src.neighbours import *

INFINITY = np.inf


def dijkstra(X, target_coordinate):
    """
    Computes the distance of
    :param X: state matrix of the cellular automata
    :param target_coordinate: (x,y) coordinate of the target cell
    :returns: matrix D that contains distance of every node
              from the given target coordinate
    """

    # Set infinite distance for all cells
    D = np.matrix(np.ones(shape=X.shape) * INFINITY).tolist()

    # Start with the zero distance on the target cell
    t_x, t_y = target_coordinate
    D[t_x][t_y] = 0

    # Build empty priority queue
    Q = PriorityQueue()

    # Insert the target to the priority queue (initial insert with the key 0)
    Q.put((0, target_coordinate))

    while not Q.empty():
        _, cell = Q.get()
        x, y = cell
        neighbouring_cells = get_neighbours(X, x, y, True)
        for neighbour in neighbouring_cells:
            n_x, n_y = neighbour
            if D[n_x][n_y] > D[x][y] + 1:
                # the distance to the neighbours is always 1 in this setup
                Q.put((D[x][y] + 1, neighbour))
                # update the new distance
                D[n_x][n_y] = D[x][y] + 1

    return D
