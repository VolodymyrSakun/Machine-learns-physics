from structure import spherical
import random
from math import pi
import numpy as np

# read molecules from descriptor file
Molecules = spherical.ReadMolecules(F='MoleculesDescriptor.')  
# align molecules that its center of mass it in 0,0,0 and principal axis aligned along x, y, z
Water = spherical.align_molecule(Molecules[0])
Ethane = spherical.align_molecule(Molecules[1])
# this is the example of using the function for your recursion
A = spherical.f1(Ethane, 5, 15, 25, 35, 45, 55, CoordinateSystem='Cartesian', AngleType='Degree') 
# A is numpy array. it has 3 columns (x, y, z) and many rows that correspond to atom coordinates
theta = 90 # angle between x axis and projection of point to xy plane
phi = 90 # angle between z axis and O-P vector
psi = 90 # angle of rotation around axis of rotation specified by angles theta and phi
new_center2 = spherical.Point(3, 3, 3)
new_center3 = spherical.Point(-2, -3, -4)
# rorate molecule according to angles
Water1 = spherical.rotate_molecule_angles(Water, theta, phi, psi, AngleType='Degree')
# translate molecule to new center
Water2 = spherical.translate_molecule_to_new_center(Water1, new_center2)
Water3 = spherical.translate_molecule_to_new_center(Water1, new_center3)
Ethane1 = spherical.rotate_molecule_angles(Ethane, theta, phi, psi, AngleType='Degree')
Ethane2 = spherical.translate_molecule_to_new_center(Ethane1, new_center2)
Ethane3 = spherical.translate_molecule_to_new_center(Ethane1, new_center3)
rotation_axis = spherical.get_axis_of_rotation(theta, phi, AngleType='Degree')

# plot molecules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig1 = plt.figure(100, figsize=(19,10))
ax = fig1.gca(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# ax.auto_scale_xyz([0, 6], [0, 6], [0, 6])
# example of ethane rotation / translation
spherical.plot_molecule(ax, Ethane)
spherical.plot_molecule(ax, Ethane1, dot_color='black', bond_color='green')
spherical.plot_molecule(ax, Ethane2, dot_color='grey', bond_color='yellow')
spherical.plot_molecule(ax, Ethane3, dot_color='pink', bond_color='magenta')
# ax.plot([0, 5*rotation_axis.x], [0, 5*rotation_axis.y], [0, 5*rotation_axis.z])
ax.legend()

fig2 = plt.figure(101, figsize=(19,10))
bx = fig2.gca(projection='3d')
bx.set_xlabel('X Label')
bx.set_ylabel('Y Label')
bx.set_zlabel('Z Label')
# bx.set_aspect('equal')
# bx.auto_scale_xyz([0, 10], [0, 10], [0, 10])
# example of water rotation / translation
spherical.plot_molecule(bx, Water)
spherical.plot_molecule(bx, Water1, dot_color='black', bond_color='green')
spherical.plot_molecule(bx, Water2, dot_color='grey', bond_color='yellow')
spherical.plot_molecule(bx, Water3, dot_color='pink', bond_color='magenta')
bx.legend()

# generate coordinates
RandomSeed = 101
Radius = 10 # Radius of sphere where water molecules will be placed
nMolecules = 3 # number of molecules in the system
nRecords = 10000 # number of records
max_trials = 1000
# random.seed(RandomSeed)
random.seed()
Molecules = []
records = []
rec = 0
empty_loop = 0
Saturated = 0
additional_gap = 0 # to be added to r1 + r2

while (rec <  nRecords) and (empty_loop < max_trials):
    if (rec % 10 == 0) and (Saturated == 0):
        print(rec)
    Saturated = 0
    while (len(Molecules) < nMolecules) and (Saturated < max_trials):
        theta = 2*pi*random.random()
        phi = pi*random.random()
        psi = 2*pi*random.random()
        x = 2 * Radius * (random.random() - 0.5)
        y = 2 * Radius * (random.random() - 0.5)
        z = 2 * Radius * (random.random() - 0.5)
        new_center = spherical.Point(x, y, z)
        Water1 = spherical.rotate_molecule_angles(Water, theta, phi, psi, AngleType='Radian')
        Water2 = spherical.translate_molecule_to_new_center(Water1, new_center)
        Good = True
        for j in Molecules:
            for k in j.Atoms:
                x1 = k.x
                y1 = k.y
                z1 = k.z
                r1 = k.Atom.Radius
                for l in Water2.Atoms:
                    x2 = l.x
                    y2 = l.y
                    z2 = l.z
                    r2 = l.Atom.Radius
                    critical_distance = r1 + r2 + additional_gap
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)            
                    if distance < critical_distance:
                        Good = False
                        Saturated += 1
                        break
                if not Good:
                    break
            if not Good:
                break        
        if Good:
            Molecules.append(Water2)
            Saturated = 0
    record = []
    if Saturated == 0:
        for j in Molecules:
            for k in j.Atoms:
                S = k.Atom.Symbol
                x = str(k.x)
                y = str(k.y)
                z = str(k.z)
                line = S + ': ' + x + '\t' + y + '\t' + z + '\n'
                record.append(line)
        record.append('\n')
        for line in record:
            records.append(line)
        rec += 1
        empty_loop = 0
    else:
        empty_loop += 1
    Molecules = []

f = open('Coord.x', "w")
f.writelines(records)
f.close()