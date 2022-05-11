import math


def euclidean(x1, x2, y1, y2):
    """
    computes the euclidean distance between the points (x1, y1) and (x2, y2) using the formula given in the
    exercise sheet
    :param x1: x coordinate of first point
    :param x2: y coordinate of first point
    :param y1: x coordinate of second point
    :param y2: y coordinate of second point
    :return: the euclidean distance between the given points
    """
    return math.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)
