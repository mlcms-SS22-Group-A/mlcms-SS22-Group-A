import math

from src.dijkstra import *

max_distance = 1.0


class Simulator:

    def __init__(self, matrix, target):
        self.target = target
        self.distances = dijkstra(matrix, target)

    def pedestrianupdate(self, matrix, row, column, pedestrians):
        """
        This function serves as a helper to the update() function (see below) to determine the state of cellular
        automaton at next time step, given the current state, position of pedestrian and target.

        :param matrix: Current state of cellular automata
        :param row: Current row of pedestrian
        :param column: Current column of pedestrian
        :param pedestrians: List of all pedestrians
        """
        neighbours = get_neighbours(self.distances, row, column, False)

        for i in range(0, len(neighbours)):
            if neighbours[i][0] == 0.0:
                matrix[row][column] = 0
                return

        neighbours.append([self.distances[row][column], row, column])
        for i in range(0, len(neighbours)):
            for j in range(0, len(pedestrians)):
                if pedestrians[j][0] == row and pedestrians[j][1] == column:
                    continue
                distance = math.sqrt(
                    (neighbours[i][1] - pedestrians[j][0]) ** 2 + (neighbours[i][2] - pedestrians[j][1]) ** 2)
                if distance < max_distance:
                    neighbours[i][0] += math.exp(1 / (distance ** 2 - max_distance ** 2))

        neighbours_distance = list(map(lambda n: n[0], neighbours))
        min_distance_index = np.array(neighbours_distance).argmin()

        matrix[row][column] = 0
        matrix[neighbours[min_distance_index][1]][neighbours[min_distance_index][2]] = 1

    def update(self, matrix, pedestrians):
        """
         This function updates the state of cellular automaton such that all pedestrians move towards the target.

         :param matrix: Current state of cellular automata
         :param pedestrians: List of pedestrians
       """
        row, _ = pedestrians.shape
        for i in range(0, row):
            self.pedestrianupdate(matrix, pedestrians[i][0], pedestrians[i][1], pedestrians)
