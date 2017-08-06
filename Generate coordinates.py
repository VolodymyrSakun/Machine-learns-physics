from structure import spherical

Molecules = spherical.ReadMolecules()  

Water0 = spherical.align_molecule(Molecules[0])
Ethane0 = spherical.align_molecule(Molecules[1])

A = spherical.f1(Ethane0, 5, 5, 5, 45, 45, 45, CoordinateSystem='Cartesian', AngleType='Degree') 

theta = 30
phi = 50
psi = 40
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
ax.scatter(W0[:, 0], W0[:, 1], W0[:, 2], zdir='z', c='r', depthshade=False)
#    ax.scatter(W1[:, 0], W1[:, 1], W1[:, 2], zdir='z', c='b', depthshade=False)
ax.scatter(W2[:, 0], W2[:, 1], W2[:, 2], zdir='z', c='y', depthshade=False)    
ax.scatter(E0[:, 0], E0[:, 1], E0[:, 2], zdir='z', c='r', depthshade=False)
#    ax.scatter(E1[:, 0], E1[:, 1], E1[:, 2], zdir='z', c='b', depthshade=False)
ax.scatter(E2[:, 0], E2[:, 1], E2[:, 2], zdir='z', c='y', depthshade=False)    
for i in range(0, len(Water0.Bonds), 1):
    i1 = Water0.AtomIndex.index(Water0.Bonds[i][0])
    i2 = Water0.AtomIndex.index(Water0.Bonds[i][1])
    x1 = Water0.Atoms[i1].x
    x2 = Water0.Atoms[i2].x
    y1 = Water0.Atoms[i1].y
    y2 = Water0.Atoms[i2].y
    z1 = Water0.Atoms[i1].z
    z2 = Water0.Atoms[i2].z
    ax.plot([x1, x2], [y1, y2], [z1, z2])
for i in range(0, len(Water2.Bonds), 1):
    i1 = Water2.AtomIndex.index(Water2.Bonds[i][0])
    i2 = Water2.AtomIndex.index(Water2.Bonds[i][1])
    x1 = Water2.Atoms[i1].x
    x2 = Water2.Atoms[i2].x
    y1 = Water2.Atoms[i1].y
    y2 = Water2.Atoms[i2].y
    z1 = Water2.Atoms[i1].z
    z2 = Water2.Atoms[i2].z
    ax.plot([x1, x2], [y1, y2], [z1, z2])
for i in range(0, len(Ethane2.Bonds), 1):
    i1 = Ethane2.AtomIndex.index(Ethane2.Bonds[i][0])
    i2 = Ethane2.AtomIndex.index(Ethane2.Bonds[i][1])
    x1 = Ethane2.Atoms[i1].x
    x2 = Ethane2.Atoms[i2].x
    y1 = Ethane2.Atoms[i1].y
    y2 = Ethane2.Atoms[i2].y
    z1 = Ethane2.Atoms[i1].z
    z2 = Ethane2.Atoms[i2].z
    ax.plot([x1, x2], [y1, y2], [z1, z2])
for i in range(0, len(Ethane0.Bonds), 1):
    i1 = Ethane0.AtomIndex.index(Ethane0.Bonds[i][0])
    i2 = Ethane0.AtomIndex.index(Ethane0.Bonds[i][1])
    x1 = Ethane0.Atoms[i1].x
    x2 = Ethane0.Atoms[i2].x
    y1 = Ethane0.Atoms[i1].y
    y2 = Ethane0.Atoms[i2].y
    z1 = Ethane0.Atoms[i1].z
    z2 = Ethane0.Atoms[i2].z
    ax.plot([x1, x2], [y1, y2], [z1, z2])
# ax.plot([0, 5*rotation_axis.x], [0, 5*rotation_axis.y], [0, 5*rotation_axis.z])
plt.show()







