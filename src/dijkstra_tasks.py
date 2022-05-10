import numpy as np
import pygame
from src.read_file import read_file
from src.parser import parser
from src.pedestrian import Pedestrian
from src.dijkstra import dijkstra
from src.update import update

# average speeds according to ages
PED_MAP = {"0": 0.6,
           "1": 1.2,
           "2": 1.6,
           "3": 1.52,
           "4": 1.49,
           "5": 1.45,
           "6": 1.3,
           "7": 1.1,
           "8": 0.67
           }


def dijkstra_tasks(scenario_number, is_measuring):
    """
    simulates the given task (which has the number scenario_number) using the dijkstra distances in the update scheme.
    a pygame screen is generated and at each step pedestrians, target and the obstacles are drawn to that screen. at
    each timestep an update function is called for each pedestrian to simulate their movement
    :param scenario_number: the task to be simulated
    :param is_measuring: set to true to adjust the measuring time for test scenarios 4 and 7
    :return ped_list: list of pedestrians
    :return average_speed: average speed of pedestrians for test scenario 4
    """

    # read the initial state of the cellular automaton from a file
    state_matrix = read_file("input/task" + scenario_number + ".txt", 'i')
    # read the speed of each pedestrian from a file
    speeds = read_file("speeds.txt", 'f')

    # parse the input and get the required parameters
    rows, columns, pedestrians, obstacles, targets = parser(state_matrix)

    # defines the frames per second
    fps = 1

    if scenario_number == "5-6":
        # increase fps for faster run time
        fps = 60
        # random speed distribution between 1.2 and 1.4
        speeds = np.random.uniform(low=1.2, high=1.4, size=len(pedestrians))
    if scenario_number == "5-7":
        # increase fps for faster run time
        fps = 60
        # using the mod operation getting the corresponding average speeds from PED_MAP, speeds are sampled from a
        # gaussian distribution given the average speeds with variance 0.2
        speeds = np.empty(shape=len(pedestrians))
        for i in range(0, len(pedestrians)):
            avg = PED_MAP[str(i % 9)]
            speeds[i] = np.random.normal(loc=avg, scale=0.2, size=None)

    # initialize colors for further use
    # red for pedestrian, orange for target, purple for obstacle
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (255, 0, 0)
    orange = (250, 89, 8)
    purple = (210, 8, 250)

    # set width and height for grid rectangles and the distance between rectangles
    width = 25
    height = 25
    margin = 5

    # initialize pygame, font for P O T, and render them to place in cells
    pygame.init()
    font = pygame.font.SysFont('Calibri', 20, False, False)
    pds = font.render("P", True, black)
    obs = font.render("O", True, black)
    tgt = font.render("T", True, black)

    # window size is set according num of columns and rows, + 1 is added to ensure end and start lines are both visible
    window_size = [columns * width + (columns + 1) * margin, rows * height + (rows + 1) * margin]
    scr = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Simulator")
    clock = pygame.time.Clock()

    # extract the coordinates of the target
    [x_target, y_target] = targets[0]

    # compute the distance matrix using Dijkstra's algorithm
    distance_dijkstra = dijkstra(state_matrix, targets[0])

    # create a list that holds pedestrian objects
    ped_list = []
    # exceptions for the input file, from which the speed of each pedestrian is read
    if speeds.size == 0:
        speeds = np.array([])
    if speeds.size == 1:
        speeds = np.array([speeds])
    for i in range(0, len(pedestrians)):
        # if no specific speed is given for a pedestrian, create one with a default value of 1.6 m/s
        if i >= speeds.shape[0]:
            new_pedestrian = Pedestrian(pedestrians[i], distance_dijkstra[pedestrians[i][0]][pedestrians[i][1]],
                                        1.6, np.array_equal(pedestrians[i], targets[0]), i % 9)
        else:
            new_pedestrian = Pedestrian(pedestrians[i], distance_dijkstra[pedestrians[i][0]][pedestrians[i][1]],
                                        speeds[i], np.array_equal(pedestrians[i], targets[0]), i % 9)
        # add the created pedestrian object to the list
        ped_list.append(new_pedestrian)

    # sort pedestrians according to their distance to target
    ped_list = list(sorted(ped_list, key=lambda x: x.get_distance_to_target()))

    # this will be set to true if either every pedestrian has reached the target or user decided to close the window
    finished = False
    # counter for the timestep
    timestep = 0
    average_speed = 0

    # main loop for simulation
    while not finished:
        # user clicked the quit button on the screen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True

        # check whether all pedestrians have reached the target
        finished = True
        for ped in ped_list:
            finished = finished and ped.target_reached

        # color background to black, they will then become the lines between rectangles
        scr.fill(black)

        # grid holds the state of each cell
        grid = []
        # 0 = empty cell
        for row in range(rows):
            grid.append([])
            for column in range(columns):
                grid[row].append(0)

        # 1 = pedestrian in cell, 2 = obstacle in cell, 3 = target in cell
        for pedestrian in pedestrians:
            grid[pedestrian[0]][pedestrian[1]] = 1
        for obstacle in obstacles:
            grid[obstacle[0]][obstacle[1]] = 2
        for target in targets:
            grid[target[0]][target[1]] = 3

        # for each row and column draw white rectangles
        for row in range(rows):
            for column in range(columns):
                color = white

                if grid[row][column] == 1:
                    color = red
                if grid[row][column] == 2:
                    color = purple
                if grid[row][column] == 3:
                    color = orange

                pygame.draw.rect(scr,
                                 color,
                                 [(margin + width) * column + margin,
                                  (margin + height) * row + margin,
                                  width,
                                  height])

                # put P, O, T
                if color == red:
                    scr.blit(pds, [(margin + width) * column + margin, (margin + height) * row + margin])
                if color == purple:
                    scr.blit(obs, [(margin + width) * column + margin, (margin + height) * row + margin])
                if color == orange:
                    scr.blit(tgt, [(margin + width) * column + margin, (margin + height) * row + margin])

        # update positions of pedestrians, after that update their distance to target
        for ped in ped_list:
            if not ped.target_reached:
                ped.carry = update(state_matrix, ped, distance_dijkstra, [x_target, y_target], ped.carry + ped.speed,
                                   False, timestep)
                ped.update_distance_to_target(distance_dijkstra[ped.x][ped.y])

        # sort pedestrians according to their distance to target
        ped_list = list(sorted(ped_list, key=lambda x: x.get_distance_to_target()))

        # parse the state matrix to update pedestrians list
        _, _, pedestrians, _, _ = parser(state_matrix)

        # move to next time step
        clock.tick(fps)
        timestep += 1
        # bring the drawings to screen
        pygame.display.flip()
        # measuring for 60 seconds only
        if is_measuring and timestep > 70:
            finished = True
    # calculating average speed
    if scenario_number == "5-6":
        average_speed = 0
        counter = 0
        for ped in ped_list:
            # filter pedestrians whose speed is not measured
            if len(ped.speeds_measured) != 0:
                average_speed += np.mean(np.array(ped.speeds_measured))
                counter += 1
        average_speed /= counter
    # close the screen and shutdown simulation
    pygame.quit()
    return ped_list, average_speed
