import sys
import getopt
import json


def read_opts():
    argv = sys.argv[1:]
    opts = []
    file_location = None
    x_location = None
    y_location = None

    try:
        opts, args = getopt.getopt(argv, " f:x:y: ")
    except getopt.GetoptError as err:
        print(err)

    for opt, arg in opts:
        if opt in ['-f']:
            file_location = arg
        elif opt in ['-x']:
            x_location = arg
        elif opt in ['-y']:
            y_location = arg

    place_ped(file_location, x_location, y_location)


def place_ped(file_location, x_location, y_location):
    with open(file_location, 'r') as f_read:
        scenario_dict = json.load(f_read)
    f_read.close()

    with open("pedestrian.json", 'w') as f_read:
        ped = json.load(f_read)
    f_read.close()

    ped["position"]["x"] = float(x_location)
    ped["position"]["y"] = float(y_location)

    scenario_dict["scenario"]["topography"]["dynamicElements"].append(ped)

    with open("Sample.scenario", "w") as outfile:
        json.dump(scenario_dict, outfile, indent=2)
    outfile.close()


read_opts()
