import math
import numpy as np


def pedestrian(matrix, row, column, trow, tcolumn):
    neighbours = []
    if row != 0:
        neighbours.append([matrix[row - 1][column], row - 1, column])
    elif row != matrix.shape - 1:
        neighbours.append([matrix[row + 1][column], row + 1, column])
    elif column != 0:
        neighbours.append([matrix[row][column - 1], row, column - 1])
    elif column != matrix[0].shape - 1:
        neighbours.append([matrix[row][column + 1], row, column + 1])

    for i in range(0, len(neighbours)):
        if neighbours[i][0] == 2:
            return

    distances = []
    for i in range(0, len(neighbours)):
        distances.append(math.sqrt((neighbours[i][1] - trow) ** 2 + (neighbours[i][2] - tcolumn) ** 2))

    min_distance_index = np.array(distances).argmin()

    matrix[row][column] = 0
    matrix[neighbours[min_distance_index][1]][neighbours[min_distance_index][2]] = 1


def update(matrix, pedestrians, trow, tcolumn):
    for i in range(0, pedestrians.shape):
        pedestrian(matrix, pedestrians[i][0], pedestrians[i][1], trow, tcolumn)
    return
