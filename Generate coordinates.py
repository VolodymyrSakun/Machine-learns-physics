from structure import spherical
# read molecules from descriptor file
Molecules = spherical.ReadMolecules(F='MoleculesDescriptor.')  
# align molecules that its center of mass it in 0,0,0 and principal axis aligned along x, y, z
Water = spherical.align_molecule(Molecules[0])
Ethane = spherical.align_molecule(Molecules[1])
# this is the example of using the function for your recursion
A = spherical.f1(Ethane, 5, 5, 5, 45, 45, 45, CoordinateSystem='Cartesian', AngleType='Degree') 
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
ax.auto_scale_xyz([0, 10], [0, 10], [0, 10])
# example of ethane rotation / translation
spherical.plot_molecule(ax, Ethane)
spherical.plot_molecule(ax, Ethane1, dot_color='black', bond_color='green')
spherical.plot_molecule(ax, Ethane2, dot_color='grey', bond_color='yellow')
spherical.plot_molecule(ax, Ethane3, dot_color='pink', bond_color='magenta')
# ax.plot([0, 5*rotation_axis.x], [0, 5*rotation_axis.y], [0, 5*rotation_axis.z])
ax.legend()
# plt.show()

fig2 = plt.figure(101, figsize=(19,10))
bx = fig2.gca(projection='3d')
bx.set_xlabel('X Label')
bx.set_ylabel('Y Label')
bx.set_zlabel('Z Label')
# bx.set_aspect('equal')
bx.auto_scale_xyz([0, 10], [0, 10], [0, 10])
# example of water rotation / translation
spherical.plot_molecule(bx, Water)
spherical.plot_molecule(bx, Water1, dot_color='black', bond_color='green')
spherical.plot_molecule(bx, Water2, dot_color='grey', bond_color='yellow')
spherical.plot_molecule(bx, Water3, dot_color='pink', bond_color='magenta')
bx.legend()
# bx.show()









