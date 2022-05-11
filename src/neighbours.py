def get_neighbours(state_matrix, cell, check_obstacle=False, check_pedestrian=False):
    """
    computes and creates a list of neighbours of the given cell, which at most contains 8 elements to deal with the
    boundary cells of the grid, many checks are done and only the appropriate neighbours are added to the list
    (for example if a left upper corner is given to compute its neighbours, the list contains bottom, bottom right and
    right neighbours only)
    :param state_matrix: state matrix of cellular automaton ({0,1,2,3} == {E,P,O,T})
    :param cell: coordinates of the cell as an array of length 2 : [x_coord, y_coord]
    :param check_obstacle: if set to True, neighbour list does not include obstacle cells
    :param check_pedestrian: if set to True, neighbour list does not include pedestrian cells
    :returns: (valid) neighbour list of the given cell
    """
    # get the row and column index of given cell in the state matrix
    [row, col] = cell
    neighbours = []
    m_row, m_col = state_matrix.shape

    if row != 0:
        # not top boundary, we add the upper cell
        neighbours.append([state_matrix[row - 1, col], row - 1, col])
        if col != 0:
            # not left boundary, we add the upper-left cell
            neighbours.append([state_matrix[row - 1, col - 1], row - 1, col - 1])
        if col != m_col - 1:
            # not right boundary, we add the upper-right cell
            neighbours.append([state_matrix[row - 1][col + 1], row - 1, col + 1])
    if row != m_row - 1:
        # not bottom boundary, we add the bottom cell
        neighbours.append([state_matrix[row + 1][col], row + 1, col])
        if col != 0:
            # not left boundary, we add the bottom-left cell
            neighbours.append([state_matrix[row + 1, col - 1], row + 1, col - 1])
        if col != m_col - 1:
            # not right boundary, we add the bottom-right cell
            neighbours.append([state_matrix[row + 1, col + 1], row + 1, col + 1])
    if col != 0:
        # not left boundary, we add the left cell
        neighbours.append([state_matrix[row][col - 1], row, col - 1])
    if col != m_col - 1:
        # not right boundary, we add the right cell
        neighbours.append([state_matrix[row][col + 1], row, col + 1])
    if check_obstacle:
        # filter the obstacle neighbours
        neighbours = list(filter(lambda neighbour: neighbour[0] != 2, neighbours))
    if check_pedestrian:
        # filter the pedestrian neighbours
        neighbours = list(filter(lambda neighbour: neighbour[0] != 1, neighbours))
    # map the list such that it only contains the coordinates of the neighbours as a tuple
    neighbours = list(map(lambda neighbour: (neighbour[1], neighbour[2]), neighbours))
    return neighbours
