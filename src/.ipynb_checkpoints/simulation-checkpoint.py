import math
import numpy as np

class Simulator:
    
    def __init__(self):
        self.target=False
    def pedestrianupdate(self,matrix, row, column, trow, tcolumn):
        """
        This function serves as a helper to the update() function (see below) to determine the state of cellular
        automaton at next time step, given the current state, position of pedestrian and target.

        :param matrix: Current state of cellular automaton
        :param row: Current row of pedestrian
        :param column: Current column of pedestrian
        :param trow: The row at which target is stored
        :param tcolumn: The column at which target is stored
        """
        neighbours = []
        m_row, m_col = matrix.shape
        if row != 0:
            neighbours.append([matrix[row - 1][column], row - 1, column])
        if row != m_row - 1:
            neighbours.append([matrix[row + 1][column], row + 1, column])
        if column != 0:
            neighbours.append([matrix[row][column - 1], row, column - 1])
        if column != m_col - 1:
            neighbours.append([matrix[row][column + 1], row, column + 1])

        for i in range(0, len(neighbours)):
            if neighbours[i][0] == 3:
                if not self.target:
                      matrix[row][column]=0  
                self.target=True
                return 

        distances = []
        for i in range(0, len(neighbours)):
            distances.append(math.sqrt((neighbours[i][1] - trow) ** 2 + (neighbours[i][2] - tcolumn) ** 2))

        min_distance_index = np.array(distances).argmin()
        minNeighbor =  matrix[neighbours[min_distance_index][1]][neighbours[min_distance_index][2]]
        while minNeighbor == 2:
            distances[min_distance_index] =distances[np.array(distances).argmax()]+1
            min_distance_index = np.array(distances).argmin()
            minNeighbor =  matrix[neighbours[min_distance_index][1]][neighbours[min_distance_index][2]]
        if minNeighbor == 1:
            return
        matrix[row][column] = 0
        matrix[neighbours[min_distance_index][1]][neighbours[min_distance_index][2]] = 1
    

    def update(self,matrix, pedestrians, trow, tcolumn):
        """
         This function updates the state of cellular automaton such that all pedestrians move towards the target.

         :param matrix: Current state of cellular automaton
         :param pedestrians: List of pedestrians
         :param trow: The row at which target is stored
         :param tcolumn: The column at which target is stored
       """
        row, _ = pedestrians.shape
        for i in range(0, row):
             self.pedestrianupdate(matrix, pedestrians[i][0], pedestrians[i][1], trow, tcolumn)
       
