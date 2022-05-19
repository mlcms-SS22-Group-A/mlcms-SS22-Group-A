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
    targetId = scenario_dict["scenario"]["topography"]["targets"][0]["id"]

    topography_ids = []
    for obstacle in scenario_dict["scenario"]["topography"]["obstacles"]:
        topography_ids.append(obstacle["id"])
    for target in scenario_dict["scenario"]["topography"]["targets"]:
        topography_ids.append(target["id"])
    for source in scenario_dict["scenario"]["topography"]["sources"]:
        topography_ids.append(source["id"])
    for ped in scenario_dict["scenario"]["topography"]["dynamicElements"]:
        topography_ids.append(ped["attributes"]["id"])

    topography_ids.sort()
    numItems = len(topography_ids)
    pedId = topography_ids[numItems - 1] + 1

    with open("pedestrian.json", 'r+') as f_read:
        ped = json.load(f_read)

    f_read.close()

    ped["attributes"]["id"] = pedId
    ped["position"]["x"] = float(x_location)
    ped["position"]["y"] = float(y_location)
    ped["targetIds"] = [ int(targetId) ]

    scenario_dict["scenario"]["topography"]["dynamicElements"].append(ped)

    f_outputName = file_location.replace(".scenario", "_pedAdded.scenario")
    with open(f_outputName, "w") as outfile:
        json.dump(scenario_dict, outfile, indent=2)
    outfile.close()


read_opts()
