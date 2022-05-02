import numpy as np
from queue import PriorityQueue

INFINITY = np.inf

def getNeighbours(X, x, y):
    """
    Returns the valid neighbouring cells (no obstacles)
    :param X: the state matrix of the cellular automaton
    :param x: x-coordinate of the target cell
    :param y: y-coordinate of the target cell
    :returns: neighbouring_cells
    """
    neighbouring_cells = []
    rows = len(X)
    cols = len(X[0])

    # for better readability TODO can we compute this using np arrays?
    isBoundaryCell = x == 0 or y == 0 or x == rows - 1 or y == cols - 1

    if isBoundaryCell:
        if x == 0 and y == cols - 1:
            # upper right corner
            if X[x][y - 1] != 2:
                # left
                neighbouring_cells.append((x, y - 1))
            if X[x + 1][y] != 2:
                # down
                neighbouring_cells.append((x + 1, y))
        elif x == rows - 1 and y == cols - 1:
            # down right corner
            if X[x - 1][y] != 2:
                # up
                neighbouring_cells.append((x - 1, y))
            if X[x][y - 1] != 2:
                # left
                neighbouring_cells.append((x, y - 1))
        elif x == rows - 1 and y == 0:
            # down left corner
            if X[x - 1][y] != 2:
                # up
                neighbouring_cells.append((x - 1, y))
            if X[x][y + 1] != 2:
                # right
                neighbouring_cells.append((x, y + 1))
        elif x == 0 and y == 0:
            # upper left corner
            if X[x][y + 1] != 2:
                # right
                neighbouring_cells.append((x, y + 1))
            if X[x - 1][y] != 2:
                # down
                neighbouring_cells.append((x - 1, y))
        elif y == cols - 1:
            # right column boundary
            if X[x + 1][y] != 2:
                # up
                neighbouring_cells.append((x + 1, y))
            if X[x][y - 1] != 2:
                # left
                neighbouring_cells.append((x, y - 1))
            if X[x - 1][y] != 2:
                # down
                neighbouring_cells.append((x - 1, y))
        elif x == rows - 1:
            # down row boundary
            if X[x][y - 1] != 2:
                # left
                neighbouring_cells.append((x, y - 1))
            if X[x - 1][y] != 2:
                # up
                neighbouring_cells.append((x - 1, y))
            if X[x][y + 1] != 2:
                # right
                neighbouring_cells.append((x, y + 1))
        elif y == 0:
            # left column boundary
            if X[x - 1][y] != 2:
                # up
                neighbouring_cells.append((x - 1, y))
            if X[x][y + 1] != 2:
                # right
                neighbouring_cells.append((x, y + 1))
            if X[x - 1][y] != 2:
                # down
                neighbouring_cells.append((x - 1, y))
        elif x == 0:
            # top row boundary
            if X[x][y - 1] != 2:
                # left
                neighbouring_cells.append((x, y - 1))
            if X[x - 1][y] != 2:
                # down
                neighbouring_cells.append((x - 1, y))
            if X[x][y + 1] != 2:
                # right
                neighbouring_cells.append((x, y + 1))
    else:
        if X[x][y + 1] != 2:
            # right
            neighbouring_cells.append((x, y + 1))
        if X[x + 1][y] != 2:
            # down
            neighbouring_cells.append((x + 1, y))
        if X[x][y - 1] != 2:
            # left
            neighbouring_cells.append((x, y - 1))
        if X[x - 1][y] != 2:
            # up
            neighbouring_cells.append((x - 1, y))

    return neighbouring_cells

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
    D[t_x][t_y] = 0;

    # Build empty priority queue
    Q = PriorityQueue()

    # Insert the target to the priority queue (initial insert with the key 0)
    Q.put((0, target_coordinate))

    while not Q.empty():
        _, cell = Q.get()
        x, y = cell
        neighbouring_cells = getNeighbours(X, x, y)
        for neighbour in neighbouring_cells:
            n_x, n_y = neighbour
            if D[n_x][n_y] > D[x][y] + 1:
                # the distance to the neighbours is always 1 in this setup
                Q.put((D[x][y] + 1, neighbour))
                # update the new distance
                D[n_x][n_y] = D[x][y] + 1

    return D
