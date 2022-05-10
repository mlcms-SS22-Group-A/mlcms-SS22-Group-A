import numpy as np

from src.neighbours import get_neighbours
from src.update_distances import update_distances
from src.euclidean_distance import euclidean

# the distance (cost) of a diagonal movement
DIAGONAL_DIST = 1.4
MEASURING_POINT_1 = 249
MEASURING_POINT_2 = 251


def move(state_matrix, ped, cell, available_movement, timestep):
    """
    moves the pedestrian (ped) towards the given cell. a check is done to assure that the current speed of the
    pedestrian is sufficient to make pedestrian traverse the distance between the current position of the pedestrian
    and the neighbouring cell, if not we pass the current available movement to the next timestep as carry
    :param state_matrix: current state of cellular automaton
    :param ped: the pedestrian to move
    :param cell: neighbouring cell to move
    :param available_movement: available movement that the pedestrian can do (speed + carry)
    :returns: (boolean, movement_left)
              boolean value: if the pedestrian could successfully move to that cell (speed was sufficient)
              movement_left: remaining movement that the current pedestrian can do
    """
    [n_x, n_y] = cell
    # check whether cell to move is diagonal
    is_diagonal = (ped.x - n_x != 0) and (ped.y - n_y != 0)
    able_to_move = False

    # check if ped is able to move to given cell by comparing the available movement and the cost for that
    # specific movement
    if is_diagonal:
        if available_movement >= DIAGONAL_DIST:
            able_to_move = True
    else:
        if available_movement >= 1.0:
            able_to_move = True

    # movement_left holds the available movement of the pedestrian after her movement in the current step,
    # if she is not able to move in the current timestep because of insufficient speed, all of her available movmement
    # will be passed to the next timestep
    movement_left = available_movement

    if able_to_move:
        # move the pedestrian
        prev_x = ped.x
        prev_y = ped.y
        [curr_x, curr_y] = cell
        ped.x = curr_x
        ped.y = curr_y

        # measuring point reached, append current speed to the speeds_measured list (speed = distance / time)
        if (ped.y == MEASURING_POINT_1 or ped.y == MEASURING_POINT_2) and (timestep > 10):
            ped.speeds_measured.append(euclidean(ped.x, ped.y, ped.start_position[0], ped.start_position[1]) /
                                       (timestep + 1))
        
        # empty the previous cell
        state_matrix[prev_x][prev_y] = 0
        # fill the current cell with pedestrian if it is not the target
        if state_matrix[curr_x][curr_y] != 3:
            state_matrix[curr_x][curr_y] = 1

        # subtract from the available movement the cost for the current movement
        if is_diagonal:
            movement_left = available_movement - DIAGONAL_DIST
        else:
            movement_left = available_movement - 1

    return able_to_move, movement_left


def update(state_matrix, ped, distance_matrix, target, available_movement, euclidean, timestep):
    """
    a discrete-time update scheme for the cellular automaton where for a given pedestrian all available neighbours are
    found, their distances to target are calculated (in case of dijkstra they are already precomputed and stored in a
    matrix), the best available neighbour is found and pedestrian moves towards that neighbour.
    :param state_matrix: current state of cellular automaton
    :param ped: current pedestrian to update
    :param distance_matrix: a matrix holding either the euclidean distance or the distances that are calculated
    using dijkstra algorithm (the type of data changes according to the task number)
    :param target: coordinates of the target
    :param available_movement: the distance that the pedestrian can traverse in the current step
    :param euclidean: boolean to check whether euclidean distances are used in the update scheme
    :return movement_left: the movement that is left available and that will be carried into the next timestep
    """
    # get available neighbours of current pedestrian (both pedestrians and obstacles excluded from the list)
    available_neighbours = get_neighbours(state_matrix, [ped.x, ped.y], True, True)
    if len(available_neighbours) == 0:
        # no available neighbour (surrounded with obstacles?)
        return 0
    # append current position of pedestrian as a neighbour since staying in place is also a valid option
    available_neighbours.append([ped.x, ped.y])

    # since we fill the euclidean distances matrix throughout the simulation, the update method is only required if
    # euclidean distances are used
    if euclidean:
        update_distances(available_neighbours, target, distance_matrix)

    # this commented out section refers to a try of adding a cost function to avoid pedestrian collision
    """
    # compute the euclidean distances from the available neighbouring cells to the target for
    # the current pedestrian
    costs = np.empty(shape=distance_matrix.shape, dtype=float) # we only use this for euclidean right now TODO
    # TODO TRYING TO INTEGRATE THE COST FUNCTION
    for i in range(0, len(available_neighbours)):
        for j in range(0, len(ped_list)):
            if ped_list[j].x == ped.x and ped_list[j].y == ped.y:
                continue
            distance = math.sqrt((available_neighbours[i][0] - ped_list[j].x) ** 2 + (available_neighbours[i][1] - ped_list[j].y) ** 2)
            if distance < MAX_DISTANCE:
                costs[available_neighbours[i][0]][available_neighbours[i][1]] += math.exp(1 / (distance ** 2 - MAX_DISTANCE     ** 2))
     """

    # for each neighbour take their distance and append them into a list where each element looks like following:
    # (distance_of_neighbour_to_target, [neighbour.x, neighbour.y])
    neighbours_and_dist = []
    for neighbour in available_neighbours:
        [n_x, n_y] = neighbour
        neighbours_and_dist.append((distance_matrix[n_x][n_y], neighbour))

    # sort the neighbours list according to their distances and find the best neighbour
    neighbours_and_dist_sorted = sorted(neighbours_and_dist, key=lambda tup: tup[0])
    neighbours_sorted = list(map(lambda tup: tup[1], neighbours_and_dist_sorted))

    # find the best available neighbour
    [x, y] = neighbours_sorted[0]

    # if the pedestrian stayed at her location, return 0 as movement_left since she did not try to move
    if np.array_equal([x, y], [ped.x, ped.y]):
        return 0

    # move towards the best neighbour
    (ped_moved, movement_left) = move(state_matrix, ped, [x, y], available_movement, timestep)

    # recursive call if the pedestrian has achieved to move in the current step to check whether she can still traverse
    # to another cell
    if ped_moved:
        # check if pedestrian has reached the target
        if np.array_equal([ped.x, ped.y], target):
            ped.target_reached = True
            return
        # recursive call with the movement_left from current step given as available_movement parameter
        return update(state_matrix, ped, distance_matrix, target, movement_left, euclidean, timestep)

    # return the movement that is left from current timestep
    return movement_left
