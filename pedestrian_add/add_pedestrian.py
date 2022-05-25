import sys
import getopt
import json

def usage():
    """
    To print the usage in case the user gives the opt '-h'
    """
    print("-------------------------------------------------------------------------------------")
    print("python add_pedestrian -f [.scenario file path] -x [x-coordinate] -y [y-coordinate]")
    print()
    print("Adds a single pedestrian at the given position and to the given .scenario file.")
    print("But does not modify the given scenario file. Saves the new scenario to a new file.")
    print("New file is the old name and '_pedAdded' string appended to the end.")
    print("There should be a 'pedestrian.json' file in the same directory.")
    print()
    print("\t-h\tshow this help message")
    print("\t-f\tspecify the .scenario file, which you want to modify and add a pedestrian")
    print("\t-x\tthe x-coordinate of the pedestrian")
    print("\t-y\tthe y-coordinate of the pedestrian")
    print("-------------------------------------------------------------------------------------")

def read_opts():
    """
    Reads command line arguments and parses them using getopt.
    After parsing the arguments, this method calls the actual method that adds a single
    pedestrian object into the given scenario file.
    """
    argv = sys.argv[1:]
    opts = []

    # params that we need to add a single pedestrian to the file
    # specified by the path 'file_location'
    file_location = None
    x_location = None
    y_location = None

    try:
        opts, args = getopt.getopt(argv, " f:x:y:h ")
    except getopt.GetoptError as err:
        print(err)

    # parse the given options
    for opt, arg in opts:
        if opt in ['-h']:
            usage()
            sys.exit(0)
        elif opt in ['-f']:
            file_location = arg
        elif opt in ['-x']:
            x_location = arg
        elif opt in ['-y']:
            y_location = arg
        else:
            # print help if another opt is given
            usage()
            sys.exit(1)

    # mandatory parameters, if not given show usage and abort
    if file_location is None or x_location is None or y_location is None:
        usage()
        sys.exit(1)

    place_ped(file_location, x_location, y_location)


def place_ped(file_location, x_location, y_location):
    """
    Adds a single pedestrian at (x,y) position to the given scenario file by saving the
    modified scenario file into another file with the same name and '_pedAdded' string
    appended to the end. There is no option for choosing multiple targets, so the user
    cannot specify the id of the target. The target id is read from the scenario file
    and set accordingly for the pedestrian. Each item in a scenario has a unique id.
    We avoid any collisions by looking at the ids of the source, targets, obstacles and
    other dynamicElements, and setting the id of the new pedestrian uniquely.

    There should be a 'pedestrian.json' file in the same directory.

    :param file_location: Absolute or relative path of the scenario file, to which the user wants
                          to add a pedestrian object.
    :param x_location: The x-coordinate of the pedestrian to be added.
    :param y_location: The y-coordinate of the pedestrian to be added.
    """
    with open(file_location, 'r') as f_read:
        scenario_dict = json.load(f_read)

    # we assume a single target scenario.
    targetId = scenario_dict["scenario"]["topography"]["targets"][0]["id"]

    # here we get all the ids in the scenario.
    topography_ids = []
    for obstacle in scenario_dict["scenario"]["topography"]["obstacles"]:
        topography_ids.append(obstacle["id"])
    for target in scenario_dict["scenario"]["topography"]["targets"]:
        topography_ids.append(target["id"])
    for source in scenario_dict["scenario"]["topography"]["sources"]:
        topography_ids.append(source["id"])
    for ped in scenario_dict["scenario"]["topography"]["dynamicElements"]:
        topography_ids.append(ped["attributes"]["id"])

    # we set the id of the pedestrian to be unique
    topography_ids.sort()
    numItems = len(topography_ids)
    pedId = topography_ids[numItems - 1] + 1

    with open("pedestrian.json", 'r+') as f_read:
        ped = json.load(f_read)

    f_read.close()

    # modify the fields in the pedestrian json file
    ped["attributes"]["id"] = pedId
    ped["position"]["x"] = float(x_location)
    ped["position"]["y"] = float(y_location)
    ped["targetIds"] = [ int(targetId) ]

    scenario_dict["scenario"]["topography"]["dynamicElements"].append(ped)

    # save the scenario with a new name (_pedAdded appended)
    f_outputName = file_location.replace(".scenario", "_pedAdded.scenario")
    with open(f_outputName, "w") as outfile:
        json.dump(scenario_dict, outfile, indent=2)
    outfile.close()

# Start of program
read_opts()
