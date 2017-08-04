from structure import spherical

Molecules = spherical.ReadMolecules()  

Water0 = spherical.align_molecule(Molecules[0])
Ethane0 = spherical.align_molecule(Molecules[1])
A = spherical.f1(Ethane0, 5, 5, 5, 45, 45, 45, CoordinateSystem='Cartesian', AngleType='Degree') 

theta = 45
phi = 45
psi = 180
new_center = spherical.Point(3, 3, 3)
Water1 = spherical.rotate_molecule_angles(Water0, theta, phi, psi, AngleType='Degree')
Water2 = spherical.translate_molecule_to_new_center(Water1, new_center)
Ethane1 = spherical.rotate_molecule_angles(Ethane0, theta, phi, psi, AngleType='Degree')
Ethane2 = spherical.translate_molecule_to_new_center(Ethane1, new_center)
W0 = spherical.get_atom_coordinates(Water0)
W1 = spherical.get_atom_coordinates(Water1)
W2 = spherical.get_atom_coordinates(Water2)
E0 = spherical.get_atom_coordinates(Ethane0)
E1 = spherical.get_atom_coordinates(Ethane1)
E2 = spherical.get_atom_coordinates(Ethane2)
rotation_axis = spherical.get_axis_of_rotation(theta, phi, AngleType='Degree')

# plot molecules
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
#    ax.scatter(W0[:, 0], W0[:, 1], W0[:, 2], zdir='z', c='r', depthshade=False)
#    ax.scatter(W1[:, 0], W1[:, 1], W1[:, 2], zdir='z', c='b', depthshade=False)
#    ax.scatter(W2[:, 0], W2[:, 1], W2[:, 2], zdir='z', c='y', depthshade=False)    
ax.scatter(E0[:, 0], E0[:, 1], E0[:, 2], zdir='z', c='r', depthshade=False)
#    ax.scatter(E1[:, 0], E1[:, 1], E1[:, 2], zdir='z', c='b', depthshade=False)
ax.scatter(E2[:, 0], E2[:, 1], E2[:, 2], zdir='z', c='y', depthshade=False)    

ax.plot([0, rotation_axis.x], [0, rotation_axis.y], [0, rotation_axis.z])
plt.show()







