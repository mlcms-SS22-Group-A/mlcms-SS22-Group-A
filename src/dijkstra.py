import numpy as np

from queue import PriorityQueue
from src.neighbours import *

INFINITY = np.inf


def dijkstra(state_matrix, target_coordinate):
    """
    computes a matrix that contains the shortest distance for each node in cellular automaton using a priority queue
    containing the target cell at initialization. at each step available neighbours of current cell will be added to
    the queue and distance will be updated
    :param state_matrix: state matrix of the cellular automaton
    :param target_coordinate: (x,y) coordinate of the target cell
    :returns: matrix d that contains distance of every node to the given target coordinate
    """

    # set infinite distance for all cells
    d = np.matrix(np.ones(shape=state_matrix.shape) * INFINITY).tolist()

    # start with the zero distance on the target cell
    t_x, t_y = target_coordinate
    d[t_x][t_y] = 0

    # build empty priority queue
    q = PriorityQueue()

    # insert the target to the priority queue (initial insert with the key 0)
    q.put((0, target_coordinate))

    while not q.empty():
        # get current cell
        _, cell = q.get()
        x, y = cell
        # find all available neighbours (pedestrians count as available, but obstacles does not)
        neighbouring_cells = np.array(get_neighbours(state_matrix, [x, y], True, False))
        for neighbour in neighbouring_cells:
            n_x, n_y = neighbour
            # check if the neighbour is a diagonal one
            is_diagonal = (x - n_x != 0) and (y - n_y != 0)
            if not is_diagonal:
                if d[n_x][n_y] > d[x][y] + 1:
                    # the distance to the neighbours is 1
                    q.put((d[x][y] + 1, list(neighbour)))
                    # update the new distance
                    d[n_x][n_y] = d[x][y] + 1
            else:
                if d[n_x][n_y] > d[x][y] + 1.4:
                    # the distance to the neighbour is 1.4, since it is diagonal
                    q.put((d[x][y] + 1.4, list(neighbour)))
                    # update the new distance
                    d[n_x][n_y] = d[x][y] + 1.4

    return d
