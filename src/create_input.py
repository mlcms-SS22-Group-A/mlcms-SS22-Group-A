import random
import numpy as np


def create_input_random(scenario_number,density):
    with open("input/task" + scenario_number + ".txt", 'w') as f:
        for j in range(0, 10):
      
            arr = np.random.binomial(n=1, p= density / 10, size=501)
            for i in range(0, 501):
                if i == 250 and j == 4:
                    f.write(str(3) + ",")
                elif i == 500:
                    f.write(str(arr[i]))
                else:
                    f.write(str(arr[i]) + ",")
            f.write('\n')
    f.close()
