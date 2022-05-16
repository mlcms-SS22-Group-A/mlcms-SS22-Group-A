# Imports
import sys
import getopt
import json

def read_opts():
    argv = sys.argv[ 1: ]

    try:
        opts, args = getopt.getopt(argv, " f:x:y: ")
    except:
        print("Error")

    for opt, arg in opts:
        if opt in [ '-f' ]:
            file_location = arg
        elif opt in [ '-x' ]:
            x_location = arg
        elif opt in [ '-y' ]:
            y_location = arg

    place_ped(file_location, x_location, y_location)

def place_ped(file_location, x_location, y_location):

    # read
    with open(file_location, 'r') as f_read:
        scenario_dict = json.load(f_read)
    f_read.close()

    # scenario_dict["scenario"]["topography"]["dynamicElements"].append("mert")
    # scenario_dict

    # arr =  scenario_dict["scenario"]["topography"]["dynamicElements"]
    # arr.append("mert")
    # scenario_dict["scenario"]["topography"]["dynamicElements"] = arr

    with open(file_location, 'w') as f_write:
        json.dump(scenario_dict, f_write, indent=2)

    f_write.close()

read_opts()
