import math
import numpy as np


def update_distances(available_neighbours, target, distance_matrix):
    """
    Updates the distance matrix by storing the distances of each cell in the available_neighbours list to the
    target cell. This method is only used if euclidean distances are used in update scheme, since dijkstra is
    precomputed once and distances are read from a matrix. But for efficiency reasons, we update the matrix that
    holds euclidean distances throughout the simulation (as it is needed)
    :param available_neighbours: neighbours list, which contains their coordinates, i.e: [x, y]
    :param target: target cell [t.x, t.y]
    :param distance_matrix: distance matrix that is to be updated
    """
    for neighbour in available_neighbours:
        [n_x, n_y] = neighbour
        if np.array_equal(neighbour, target):
            # target is a neighbour
            distance_matrix[n_x, n_y] = 0
        else:
            # not computed yet, we compute and store the distance
            if distance_matrix[n_x, n_y] == 0:
                [target_x, target_y] = target
                distance_matrix[n_x, n_y] = math.sqrt((n_x - target_x) ** 2 + (n_y - target_y) ** 2)
