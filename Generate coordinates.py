from project1 import spherical
import random

# read molecules from descriptor file
Molecules = spherical.ReadMolecules(F='MoleculesDescriptor.')  
# align molecules that its center of mass it in 0,0,0 and principal axis aligned along x, y, z
Water = spherical.align_molecule(Molecules[0])
CO2 = spherical.align_molecule(Molecules[1])
Ethane = spherical.align_molecule(Molecules[2])
prototype = Water # which molecule will be used
# generate coordinates
RandomSeed = None # if none use seed()
SphereRadius = 10 # Radius of sphere where water molecules will be placed
nMolecules = 3 # number of molecules in the system
nRecords = 10000 # number of records
# if we need to generate molecules far from each others, we can specify minimal
# distance between atoms by additional_gap variable
additional_gap = 11 # to be added to r1 + r2
PrototypeFirst = False
if RandomSeed is not None:
    random.seed(RandomSeed)
else:
    random.seed()
records = spherical.generate_molecule_coordinates_list(prototype,\
    nMolecules=nMolecules, nRecords=nRecords, SphereRadius=SphereRadius,\
    additional_gap=additional_gap, PrototypeFirst=PrototypeFirst,\
    max_gen_trials=100, max_inner_trials=100, max_outer_trials=100,\
    verbose=True, n_verbose=10)
# store coordinates in file
if type(records) is list:
    if additional_gap == 0:
        str1 = 'No additional gap'
    else:
        str1 = 'Additional gap = ' + str(additional_gap)
    f = open('Coordinates. ' + str(nMolecules) + ' ' + prototype.Name + ' Molecules. ' + str(nRecords) + ' Records. ' + str1 + '.x', "w")
    f.writelines(records)
    f.close()