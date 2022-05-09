import numpy as np


class Pedestrian:
    def __init__(self, cell, distance_to_target, speed=1.6, target_reached=False, age=0):
        """
        Initializes a new pedestrian object with the given parameters
        :param cell: the initial position [x, y] of pedestrian
        :param distance_to_target: the distance of the pedestrian to target
        :param speed: speed of the pedestrian in m/s
        :param target_reached: boolean to check whether the pedestrian has reached the target
        """
        [self.x, self.y] = cell
        self.age=age
        self.speeds_measured = []
        self.start_position = cell
        self.distance_to_target = distance_to_target
        self.target_reached = target_reached
        self.speed = speed
        # this variable holds the movement that is left from the last time step, which is added to the available
        # movement in the current step
        self.carry = 0.0

    # getter and setter methods for distance_to_target
    def update_distance_to_target(self, distance):
        self.distance_to_target = distance

    def get_distance_to_target(self):
        return self.distance_to_target
