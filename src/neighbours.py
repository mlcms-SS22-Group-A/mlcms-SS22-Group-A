def get_neighbours(matrix, row, column, checkobstacle):
    neighbours = []
    m_row = len(matrix)
    m_col = len(matrix[0])
    if row != 0:
        neighbours.append([matrix[row - 1][column], row - 1, column])
    if row != m_row - 1:
        neighbours.append([matrix[row + 1][column], row + 1, column])
    if column != 0:
        neighbours.append([matrix[row][column - 1], row, column - 1])
    if column != m_col - 1:
        neighbours.append([matrix[row][column + 1], row, column + 1])
    if checkobstacle:
        neighbours = list(filter(lambda neighbour: neighbour[0] != 2, neighbours))
        neighbours = list(map(lambda neighbour: (neighbour[1], neighbour[2]), neighbours))
    return neighbours
