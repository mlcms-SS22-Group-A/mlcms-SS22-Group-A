# mlcms-SS22-Group-A

The codes that are changed and added during tasks 3, 4 and 5 are in this branch. They are seperated into different 
directories, which are listed in the following:

* All the files that are required to add a pedestrian to a given scenario is the directory /pedestrian_add. 

* The files that are modified in the Vadere source code (the integration of SIR model,
coloring both during the simulation and post-visualization,
usage of LinkedCellsGrid class for efficient neighbour finding,
decoupling of infection rate and step length as much as possible and
the addition of recovered state) 
are located in the directory /vadere.

* Changes made to the provided visualization code, so that it can show the recovered state as well, is in
the /sir_visualization directory. 

* The directory /scenarios contains some scenarios that we have run to compare both our cellular
automaton with Vadere and also different models such as OSM, SFM and GNM with each other.