from src.dijkstra_tasks import dijkstra_tasks
from src.create_input import *


def test_it(scenario_number):
    """
    prepares input for tests 4 and 7 and runs the tests
    :param scenario_number:
    :return: for test scenario 4 returns the flow and the density values,
             for test scenario 7 returns the ages and the average speed of pedestrians
    """
    if scenario_number == "5-6":
        x = []
        y = []
        for i in range(1, 7):
            # create input
            create_input_random(scenario_number, i)
            # run simulation
            (pedestrian_list, avg_speed) = dijkstra_tasks(scenario_number, True)
            # append density
            x.append(i / 10)
            # append flow = density * avg_speed
            y.append((i / 10) * avg_speed)
        return x, y
    else:
        # run simulation
        (pedestrian_list, avg_speed) = dijkstra_tasks(scenario_number, True)
        results = np.empty(9, dtype=np.object)
        for i in range(0, 9):
            results[i] = []
        # calculate average speed for each age
        for ped in pedestrian_list:
            if len(ped.speeds_measured) != 0:
                results[ped.age].append(np.mean(ped.speeds_measured))
        averages = np.empty(shape=9)
        for i in range(0, 9):
            averages[i] = np.mean(np.array(results[i]))
        # set up for axis of plotted figure
        ages = [5, 10, 20, 30, 40, 50, 60, 70, 80]
        return ages, averages
