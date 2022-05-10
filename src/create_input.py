import numpy as np


def create_input_random(scenario_number, density):
    """
    this function places pedestrians row-wise according to the given density randomly
    :param scenario_number: the scenario to run
    :param density: density of pedestrian (in percentile of row, i.e density = 5 -> almost 50% of each row consist
    pedestrians)
    """
    with open("input/task" + scenario_number + ".txt", 'w') as f:
        for j in range(0, 10):
      
            arr = np.random.binomial(n=1, p=density / 10, size=501)
            for i in range(0, 501):
                if i == 250 and j == 4:
                    f.write(str(3) + ",")
                elif i == 500:
                    f.write(str(arr[i]))
                else:
                    f.write(str(arr[i]) + ",")
            f.write('\n')
    f.close()
