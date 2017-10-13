from project1 import spherical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# read molecules from descriptor file
Molecules = spherical.ReadMolecules(F='MoleculesDescriptor.')  
# align molecules that its center of mass it in 0,0,0 and principal axis aligned along x, y, z
Water = spherical.align_molecule(Molecules[0])
CO2 = spherical.align_molecule(Molecules[1])
Ethane = spherical.align_molecule(Molecules[2])
prototype = Ethane # assign molecule for calculation and plot
# this is the example of using the function for your recursion
A = spherical.f1(Ethane, 5, 15, 25, 35, 45, 55, CoordinateSystem='Cartesian', AngleType='Degree') 
# A is numpy array. it has 3 columns (x, y, z) and many rows that correspond to atom coordinates

# generate and plot molecules
RandomSeed = 101
SphereRadius = 10 # Radius of sphere where water molecules will be placed
nMolecules = 5 # number of molecules in the system
additional_gap = 6 # to be added to r1 + r2
PrototypeFirst = False # first molecule - aligned prototype

molecules = spherical.generate_molecule_coordinates_list(prototype,\
    nMolecules=nMolecules, nRecords=None, SphereRadius=SphereRadius,\
    additional_gap=additional_gap, PrototypeFirst=PrototypeFirst,\
    max_gen_trials=100, max_inner_trials=100, max_outer_trials=10,\
    verbose=False, n_verbose=1)
# molecules.append(molecule) # add original molecule
if type(molecules) is not int:
    fig1 = plt.figure(100, figsize=(19,10))
    cx = fig1.gca(projection='3d')
    cx.set_xlabel('X Label')
    cx.set_ylabel('Y Label')
    cx.set_zlabel('Z Label')
    spherical.plot_molecules(cx, molecules)

    
    