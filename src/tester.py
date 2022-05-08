from src.dijkstra_tasks import dijkstra_tasks
from src.create_input import *


def test_it(scenario_number):
    x=[]
    y=[]
    for i in range(1,7):
        create_input_random(scenario_number,i)
        avg_speed=dijkstra_tasks(scenario_number,True)
        x.append(i/10)
        y.append((i/10)*avg_speed)
    return x,y

        

        
        