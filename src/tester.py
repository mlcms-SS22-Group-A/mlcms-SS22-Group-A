from src.dijkstra_tasks import dijkstra_tasks
from src.create_input import *
import random


def test_it(scenario_number):
    if(scenario_number=="5-6"):   
        x=[]
        y=[]
        for i in range(1,7):
            create_input_random(scenario_number,i)
            (pedestrian_list,avg_speed)=dijkstra_tasks(scenario_number,True)
            x.append(i/10)
            y.append((i/10)*avg_speed)
        return x,y
    else :
        
        (pedestrian_list,avg_speed)=dijkstra_tasks(scenario_number,True)
        results = np.empty(9, dtype=np.object)
        for i in range(0,9):
            results[i]=[]
        for ped in pedestrian_list:
            if(len(ped.speeds_measured)!=0):
                results[ped.age].append(np.mean(ped.speeds_measured))
        averages = np.empty(shape=9)
        for i in range(0,9):
            averages[i] = np.mean(np.array(results[i]))
        ages=[5,10,20,30,40,50,60,70,80]
        return ages,averages
        