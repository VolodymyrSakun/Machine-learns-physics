# class Point
# class Vector
# class Plane
# function plane_from3points
# function vector_from_points
# function vector_from_coordinates
# function UnitVector
# function vector_multiply_by_number
# function vector_add
# function vector_substract
# function cross_product
# function dot_product3
# function determinant_3x3
# function projection
# function unit_vector_from_coordinates
# function get_bisection
# function get_directing_point_O2_O3
# function get_transformation_matrix
# function transform_coordinates
# function get_angles
# function get_real_form1, 2, 3
# function center_of_mass
# function translate_molecule_to_new_coordinate_system
# function translate_molecule_to_new_center
# function get_change_of_cordinates_matrix
# function rotate_molecule_axis
# function rotate_point_axes
# function get_axis_of_rotation
# function rotate_point_about_axis
# function rotate_point_angles
# function rotate_molecule_angles
# function get_inertia_tensor
# function get_atom_coordinates
# function align_molecule
# function spherical_to_cartesian
# function f1
# function check_overlap
# function ReadMolecules
# function Molecule_to_plot
# function plot_molecules
# function generate_random_molecule
# function check_new_molecule
# function generate_molecule_coordinates_list

import numpy as np
import math
from scipy.special import sph_harm
from project1 import structure
import re
import random

# n - degree. 0, 1, 2, 3 ...
# m - order. ... -3, -2, -1, 0, 1, 2, 3 ...   abs(m) <= n
# theta = [0..2pi]
# phi = [0..pi]
# s = sph_harm(m, n, theta, phi).real
        
class Point(dict):
#    x = None
#    y = None 
#    z = None
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
    
class Vector:
    x = None
    y = None 
    z = None
    length = None
    def __init__(self, X, Y, Z):
        self.x = X
        self.y = Y
        self.z = Z
        self.length = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        
class Plane:
# a*x + b*y + c*z = d
    a = None
    b = None
    c = None
    d = None
    def __init__(self, A, B, C, D):
        self.a = A
        self.b = B
        self.c = C
        self.d = D
  
def plane_from3points(p1, p2, p3):
    M = np.array([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z]])
    d = determinant_3x3(M)
    M = np.array([[1, p1.y, p1.z], [1, p2.y, p2.z], [1, p3.y, p3.z]])
    a = determinant_3x3(M)
    M = np.array([[p1.x, 1, p1.z], [p2.x, 1, p2.z], [p3.x, 1, p3.z]])
    b = determinant_3x3(M)
    M = np.array([[p1.x, p1.y, 1], [p2.x, p2.y, 1], [p3.x, p3.y, 1]])
    c = determinant_3x3(M)
    return Plane(a, b, c, d)

def vector_from_points(p_origin, p_arrow):
    v = Vector(p_arrow.x - p_origin.x, p_arrow.y - p_origin.y, p_arrow.z - p_origin.z)
    return v

def vector_from_coordinates(X1, Y1, Z1, X2, Y2, Z2):
# X1, Y1, Z1 - coordinates of origin
# X2, Y2, Z2 - coordinates of arrow
    v = Vector(X2 - X1, Y2 - Y1, Z2 - Z1)
    return v
    
def UnitVector(Vector):
    Vector.x = Vector.x / Vector.length
    Vector.y = Vector.y / Vector.length
    Vector.z = Vector.z / Vector.length
    Vector.length = 1
    return Vector

def vector_multiply_by_number(vector, number):
    v = Vector(vector.x * number, vector.y * number, vector.z * number)
    return v

def vector_add(u, v):
# u + v
    return Vector(u.x+v.x, u.y+v.y, u.z+v.z)

def vector_substract(u, v):
# u - v
    return Vector(u.x-v.x, u.y-v.y, u.z-v.z)

def cross_product(u, v):
# Return u x v
    x = u.y*v.z - u.z*v.y
    y = u.z*v.x - u.x*v.z
    z = u.x*v.y - u.y*v.x
    w = Vector(x, y, z)
    return w

def dot_product3(Vector1, Vector2):
# dot product of vectors in 3D space
    return Vector1.x * Vector2.x + Vector1.y * Vector2.y + Vector1.z * Vector2.z

def determinant_3x3(M):
    if (M.shape[0] != 3) or (M.shape[1] != 3):
        return False
    det = M[0,0]*M[1,1]*M[2,2] + M[0,1]*M[1,2]*M[2,0] + M[0,2]*M[1,0]*M[2,1] -\
        M[0,2]*M[1,1]*M[2,0] - M[0,1]*M[1,0]*M[2,2] - M[0,0]*M[1,2]*M[2,1]
    return det

def projection(u, v):
# projection u onto v
    const = dot_product3(u, v) / dot_product3(v, v)
    proj = vector_multiply_by_number(v, const)
    return proj

def unit_vector_from_coordinates(x1, y1, z1, x2, y2, z2):        
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    x = x2 - x1
    y = y2 - y1
    z = z2 - z1
    x = x / length
    y = y / length
    z = z / length
    return x, y, z

def get_bisection(p1, origin, p3):
# origin - origin of bisection
# bisection between p1 and p3 on the same plane
    p_bisection = Point((p1.x+p3.x)/2, (p1.y+p3.y)/2, (p1.z+p3.z)/2)
    bisection = Vector(p_bisection.x-origin.x, p_bisection.y-origin.y, p_bisection.z-origin.z)
    bisection = UnitVector(bisection)
    return bisection

def get_directing_point_O2_O3(O2, O3):
    directing_point = Point((O2.x+O3.x)/2, (O2.y+O3.y)/2, (O2.z+O3.z)/2)
    return directing_point
    
def get_transformation_matrix(new_origin_O, H1, H2, directing_point):
# calculate new basis and transformation matrix
# e1 is the bisection between O-H1 and O-H2
# e2 is the normal to plane O-H1-H2 pointed towards external atom
# e3 is cross product e1 x e2  
    M_translation = np.eye(4)
    M_rotation = np.eye(4)
    M_translation[0,3] = new_origin_O.x
    M_translation[1,3] = new_origin_O.y
    M_translation[2,3] = new_origin_O.z
    plane = plane_from3points(new_origin_O, H1, H2)
    e1 = get_bisection(H1, new_origin_O, H2) # new X axis. bisection unit vector
    e2 = Vector(plane.a, plane.b, plane.c) # new Y axis. normal to plane towards external atom
    e2 = UnitVector(e2)
    y_direction = vector_from_points(new_origin_O, directing_point) 
    cos = dot_product3(e2, y_direction) / y_direction.length
    if cos < 0:
        e2 = vector_multiply_by_number(e2, -1) # new Y axis
    e3 = cross_product(e1, e2) # new Z axis. right hand rule
    e3 = UnitVector(e3)
    M_rotation[0,0], M_rotation[0,1], M_rotation[0,2] = e1.x, e1.y, e1.z
    M_rotation[1,0], M_rotation[1,1], M_rotation[1,2] = e2.x, e2.y, e2.z
    M_rotation[2,0], M_rotation[2,1], M_rotation[2,2] = e3.x, e3.y, e3.z
    return e1, e2, e3, M_translation, M_rotation, 

def transform_coordinates(M, point):
# use transformation matrix M from get_transformation_matrix to transform point to new_point
    x = M[0,0]*point.x + M[0,1]*point.y + M[0,2]*point.z + M[0,3]*1
    y = M[1,0]*point.x + M[1,1]*point.y + M[1,2]*point.z + M[1,3]*1
    z = M[2,0]*point.x + M[2,1]*point.y + M[2,2]*point.z + M[0,3]*1
    new_point = Point(x, y, z)
    return new_point

def get_angles(new_origin, H1, H2, external_atom, directing_point):
    p1 = Vector(external_atom.x + new_origin.x, external_atom.y + new_origin.y, external_atom.z + new_origin.z)
    e1, e2, e3, M_translation, M_rotation = get_transformation_matrix(new_origin, H1, H2, directing_point)
    cos_phi = dot_product3(p1, e3) / p1.length
    phi = math.acos(cos_phi)
    projection_on_z = projection(p1, e3)
    projection_on_xy = vector_substract(p1, projection_on_z)
    cos_theta = dot_product3(projection_on_xy, e1) / projection_on_xy.length
    theta = math.acos(cos_theta)
    return theta, phi
    
def get_real_form1(m, n, theta, phi):
    if m == 0:
        s = sph_harm(0, n, theta, phi)
        return s
    s_m = sph_harm(m, n, theta, phi)
    s_minus_m = sph_harm((-1)*m, n, theta, phi)
    if m < 0:
        s = (1j/np.sqrt(2))*(s_m - (-1)**m * s_minus_m)
    if m > 0:
        s = (1/np.sqrt(2))*(s_minus_m + (-1)**m * s_m)
    return s.real
  
def get_real_form2(m, n, theta, phi):
    if m == 0:
        s = sph_harm(0, n, theta, phi)
        return s
    s_abs_m = sph_harm(abs(m), n, theta, phi)
    s_minus_abs_m = sph_harm((-1)*abs(m), n, theta, phi)
    if m < 0:
        s = 1j/np.sqrt(2)*(s_minus_abs_m - (-1)**m * s_abs_m)
    if m > 0:
        s = 1/np.sqrt(2)*(s_minus_abs_m + (-1)**m * s_abs_m)
    return s.real


def get_real_form3(m, n, theta, phi):
    if m == 0:
        s = sph_harm(0, n, theta, phi)
        return s
    if m < 0:
        s = np.sqrt(2) * (-1)**m * sph_harm(abs(m), n, theta, phi).imag
    if m > 0:
        s = np.sqrt(2) * (-1)**m * sph_harm(m, n, theta, phi).real
    return s.real

def center_of_mass(Object):
    if type(Object) is not list:
        if type(Object) is structure.AtomCoordinates:
            return Point(Object.x, Object.y, Object.z)
        elif type(Object) is structure.Molecule:
            if Object.nAtoms == 1: # only one atom in molecule
                return Point(Object.Atom[0].x, Object.Atom[0].y, Object.Atom[0].z)
            else:
                return center_of_mass(Object.Atoms)
    if type(Object) is not list:
        return False
    Mi = 0
    MiXi = 0
    MiYi = 0
    MiZi = 0    
    if type(Object[0]) is structure.AtomCoordinates:
        try:
            for i in range(0, len(Object), 1):
                Mi += Object[i].Atom.Mass
                MiXi += Object[i].Atom.Mass * Object[i].x
                MiYi += Object[i].Atom.Mass * Object[i].y
                MiZi += Object[i].Atom.Mass * Object[i].z
            X = MiXi / Mi
            Y = MiYi / Mi
            Z = MiZi / Mi
            return Point(X, Y, Z)
        except:
            return False
    elif type(Object[0]) is structure.Molecule:
        try:
            for i in range(0, len(Object), 1):
                Mi += Object[i].Mass
                MiXi += Object[i].Mass * Object[i].CenterOfMass.x
                MiYi += Object[i].Mass * Object[i].CenterOfMass.y
                MiZi += Object[i].Mass * Object[i].CenterOfMass.z
            X = MiXi / Mi
            Y = MiYi / Mi
            Z = MiZi / Mi
            return Point(X, Y, Z)
        except:
            return False        
        
        
        
# molecule - class Molecule; new_origin - class Point
# change coordinate system to have new origine = new_origin
def translate_molecule_to_new_coordinate_system(molecule, new_origin):
    Atoms = []
    for i in range(0, len(molecule.Atoms), 1):
        atom = molecule.Atoms[i].Atom
        X = molecule.Atoms[i].x - new_origin.x
        Y = molecule.Atoms[i].y - new_origin.y
        Z = molecule.Atoms[i].z - new_origin.z
        Atoms.append(structure.AtomCoordinates(atom, X, Y, Z))
    new_molecule = structure.Molecule(Atoms, Name=molecule.Name)
    return new_molecule

# molecule - class Molecule; new_origin - class Point
# same coordinate system, origin does not change
def translate_molecule_to_new_center(molecule, new_center):
    Atoms = []
    for i in range(0, len(molecule.Atoms), 1):
        atom = molecule.Atoms[i].Atom
        X = molecule.Atoms[i].x + new_center.x
        Y = molecule.Atoms[i].y + new_center.y
        Z = molecule.Atoms[i].z + new_center.z
        Atoms.append(structure.AtomCoordinates(atom, X, Y, Z))
    new_molecule = structure.Molecule(Atoms, Name=molecule.Name)
    return new_molecule

def get_change_of_cordinates_matrix(e1,e2,e3):
# e1, e2, e3 - new coordinate system axis
    M_rotation = np.eye(4)    
    M_rotation[0,0], M_rotation[0,1], M_rotation[0,2] = e1.x, e1.y, e1.z
    M_rotation[1,0], M_rotation[1,1], M_rotation[1,2] = e2.x, e2.y, e2.z
    M_rotation[2,0], M_rotation[2,1], M_rotation[2,2] = e3.x, e3.y, e3.z
    return M_rotation

def rotate_molecule_axis(molecule, e1, e2, e3):
    Atoms = []
    for i in range(0, len(molecule.Atoms), 1):
        atom = molecule.Atoms[i].Atom
        M_rotation = get_change_of_cordinates_matrix(e1, e2, e3)
        old_coordinates = np.array([[molecule.Atoms[i].x], [molecule.Atoms[i].y], [molecule.Atoms[i].z], [1]])
        new_coordinates = np.dot(M_rotation, old_coordinates)
        Atoms.append(structure.AtomCoordinates(atom, new_coordinates[0, 0], new_coordinates[1, 0], new_coordinates[2, 0]))
    new_molecule = structure.Molecule(Atoms, Name=molecule.Name)        
    return new_molecule

def rotate_point_axes(point, e1, e2, e3):
    M_rotation = get_change_of_cordinates_matrix(e1, e2, e3)
    old_coordinates = np.array([[point.x], [point.y], [point.z], [1]])
    new_coordinates = np.dot(M_rotation, old_coordinates)
    new_point = Point(new_coordinates[0, 0], new_coordinates[1, 0], new_coordinates[2, 0])
    return new_point

def get_axis_of_rotation(theta, phi, AngleType='Radian'):
    if AngleType == 'Degree':
        theta = theta * math.pi/180
        phi = phi * math.pi/180
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi) 
    v = Vector(x, y, z)
    unit_vector = UnitVector(v)
    return unit_vector
    
def rotate_point_about_axis(axis, psi, point, AngleType='Radian'):
    if AngleType == 'Degree':
        psi = psi * math.pi/180
    a = axis.x
    b = axis.y
    c = axis.z
    d = np.sqrt(b**2 + c**2) # length of the projection onto the yz plane
    Rx = np.eye(4)
    RxInverse = np.eye(4)
    Ry = np.eye(4)
    RyInverse = np.eye(4)
    Rz = np.eye(4)
    Rx[1, 1] = c/d # cos(t)
    Rx[1, 2] = -b/d # -sin(t)
    Rx[2, 1] = b/d # sin(t)
    Rx[2, 2] = c/d # cos(t)   
    RxInverse[1, 1] = c/d # cos(t)
    RxInverse[1, 2] = b/d # sin(t)
    RxInverse[2, 1] = -b/d # -sin(t)
    RxInverse[2, 2] = c/d # cos(t)     
    Ry[0, 0] = d
    Ry[0, 2] = -a
    Ry[2, 0] = a
    Ry[2, 2] = d
    RyInverse[0, 0] = d
    RyInverse[0, 2] = a
    RyInverse[2, 0] = -a
    RyInverse[2, 2] = d
    Rz[0, 0] = math.cos(psi)
    Rz[0, 1] = -math.sin(psi)    
    Rz[1, 0] = math.sin(psi)
    Rz[1, 1] = math.cos(psi)    
    old_coordinates = np.array([[point.x], [point.y], [point.z], [1]])
    new_coordinates = RxInverse.dot(RyInverse).dot(Rz).dot(Ry).dot(Rx).dot(old_coordinates)
    new_point = Point(new_coordinates[0, 0], new_coordinates[1, 0], new_coordinates[2, 0])
    return new_point
    
def rotate_point_angles(theta, phi, psi, point, AngleType='Radian'):
    if AngleType == 'Degree':
        theta = theta * math.pi/180
        phi = phi * math.pi/180
        psi = psi * math.pi/180
    rotation_axis = get_axis_of_rotation(theta, phi, AngleType='Radian')
    new_point = rotate_point_about_axis(rotation_axis, psi, point, AngleType='Radian')
    return new_point
    
def rotate_molecule_angles(molecule, theta, phi, psi, AngleType='Radian'):
    if AngleType == 'Degree':
        theta = theta * math.pi/180
        phi = phi * math.pi/180
        psi = psi * math.pi/180
    Atoms = []
    for i in range(0, len(molecule.Atoms), 1):
        atom = molecule.Atoms[i].Atom
        point = Point(molecule.Atoms[i].x, molecule.Atoms[i].y, molecule.Atoms[i].z)
        new_point = rotate_point_angles(theta, phi, psi, point, AngleType='Radian')
        Atoms.append(structure.AtomCoordinates(atom, new_point.x, new_point.y, new_point.z))
    new_molecule = structure.Molecule(Atoms, Name=molecule.Name)        
    return new_molecule
    
def get_inertia_tensor(molecule):
    Ixx = 0
    Iyy = 0
    Izz = 0
    Ixy = 0
    Iyz = 0
    Ixz = 0
    for i in range(0, len(molecule.Atoms), 1):
        Ixx += (molecule.Atoms[i].y**2 + molecule.Atoms[i].z**2) * molecule.Atoms[i].Atom.Mass
        Iyy += (molecule.Atoms[i].x**2 + molecule.Atoms[i].z**2) * molecule.Atoms[i].Atom.Mass
        Izz += (molecule.Atoms[i].x**2 + molecule.Atoms[i].y**2) * molecule.Atoms[i].Atom.Mass
        Ixy += -(molecule.Atoms[i].x * molecule.Atoms[i].y * molecule.Atoms[i].Atom.Mass)
        Iyx = Ixy
        Iyz += -(molecule.Atoms[i].y * molecule.Atoms[i].z * molecule.Atoms[i].Atom.Mass)
        Izy = Iyz
        Ixz += -(molecule.Atoms[i].x * molecule.Atoms[i].z * molecule.Atoms[i].Atom.Mass)
        Izx = Ixz
    I = np.zeros(shape=(3, 3), dtype=float)
    I[0, 0] = Ixx
    I[0, 1] = Ixy
    I[0, 2] = Ixz
    I[1, 0] = Iyx
    I[1, 1] = Iyy
    I[1, 2] = Iyz
    I[2, 0] = Izx
    I[2, 1] = Izy
    I[2, 2] = Izz
    return I

def get_atom_coordinates(molecule):
    nAtoms = len(molecule.Atoms)
    A = np.zeros(shape=(nAtoms, 3), dtype=float)
    R = np.zeros(shape=(nAtoms), dtype=float)
    for i in range(0, nAtoms, 1):
        A[i, 0] = molecule.Atoms[i].x
        A[i, 1] = molecule.Atoms[i].y
        A[i, 2] = molecule.Atoms[i].z
        R[i] = molecule.Atoms[i].Atom.Radius
    return A, R

def align_molecule(molecule):
    molecule = translate_molecule_to_new_coordinate_system(molecule, molecule.CenterOfMass)    
    I = get_inertia_tensor(molecule)
    w, v = np.linalg.eig(I)
    e1 = Vector(v[0, 0], v[1, 0], v[2, 0])
    e2 = Vector(v[0, 1], v[1, 1], v[2, 1])
    e3 = Vector(v[0, 2], v[1, 2], v[2, 2]) 
    molecule = rotate_molecule_axis(molecule, e1, e2, e3)
    return molecule
    
def spherical_to_cartesian(R, theta, phi, AngleType='Radian'):
    if AngleType == 'Degree':
        Theta = theta * math.pi/180
        Phi = phi * math.pi/180
    else:
        Theta = theta
        Phi = phi
    x = R * math.sin(Phi) * math.cos(Theta)
    y = R * math.sin(Phi) * math.sin(Theta)
    z = R * math.cos(Phi) 
    p = Point(x, y, z)
    return p
    
def f1(molecule, arg1, arg2, arg3, theta, phi, psi, CoordinateSystem='Cartesian',\
       AngleType='Radian'):
# AngleType can be 'Radian' or 'Degree'
# CoordinateSystem can be 'Cartesian' or 'Spherical'
# if CoordinateSystem='Cartesian' arg1 = x coordinate of center of mass of molecule
# arg2 = y coordinate of center of mass of molecule
# arg3 = z coordinate of center of mass of molecule
# if CoordinateSystem='Spherical' arg1 = R (distance from origin to center of mass of molecule)
# arg2 = theta - angle between projection of vector R on XY plane and X axis
# arg3 = phi - angle between Vector R and Z axis
    if AngleType == 'Degree':
        Theta1 = theta * math.pi/180
        Phi1 = phi * math.pi/180
        Psi = psi * math.pi/180
    else:
        Theta1 = theta
        Phi1 = phi
        Psi = psi
    if CoordinateSystem == 'Spherical':
        R = arg1
        if AngleType == 'Degree':
            Theta2 = arg2 * math.pi/180
            Phi2 = arg3 * math.pi/180
        COM_coordinates = spherical_to_cartesian(R, Theta2, Phi2, AngleType='Radian')
    if CoordinateSystem == 'Cartesian':
        COM_coordinates = Point(arg1, arg2, arg3) # x, y, z
    molecule0 = align_molecule(molecule) # molecule aligned at 0,0,0
    molecule1 = rotate_molecule_angles(molecule0, Theta1, Phi1, Psi, AngleType='Radian')
    molecule2 = translate_molecule_to_new_center(molecule1, COM_coordinates)
    A0, _ = get_atom_coordinates(molecule0)
    A1, _ = get_atom_coordinates(molecule2)
    A = np.concatenate((A0, A1), axis=0)
    return A        
            
def check_overlap(molecule1, molecule2):
# returns True if atoms occupy same space, otherwise returns False
    for i in range(0, len(molecule1.Atoms), 1):
        for j in range(0, len(molecule2.Atoms), 1):
            v = vector_from_coordinates(molecule1.Atoms[i].x, molecule1.Atoms[i].y,\
                molecule1.Atoms[i].z, molecule2.Atoms[j].x, molecule2.Atoms[j].y,\
                molecule2.Atoms[j].z)
            if v.length > (molecule1.Atoms[i].Atom.Radius + molecule2.Atoms[j].Atom.Radius):
                return True
    return False
    
def ReadMolecules(F='MoleculesDescriptor.'): # inactive
    F = F # file with info about system structure
    Separators = '=|,| |:|;|: '
# read descriptor from file
    with open(F) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string
    i = 0
    nMolecule = 0
    Atoms = []
    while i < len(lines):
        x = lines[i]
        if len(x) == 0:
            del(lines[i])
            continue
        if x[0] == '#':
            del(lines[i])
            continue
        i += 1
    i = 0
    j = 0
    types_list = []
    idx_list = []
    MoleculeNames = []
    k = 0
    while i < len(lines):    
        x = lines[i]
        if (x.find('Molecule') != -1):
#            x = lines[i]
            s = re.split(Separators, x)
            del(s[0])
            s0 = s[0]
            for l in range(1, len(s), 1):
                s0 = s0 + ' ' + s[l]
            MoleculeNames.append(s0)
            i += 1
            while i < len(lines):
                x = lines[i]
                if x.find('endMolecule') != -1:
                    break
                s = re.split(Separators, x)
                s = list(filter(bool, s))
                symbol = s[0]
                bond_idx = s.index("Bond")
                mass_idx = s.index("Mass")   
                radius_idx = s.index("Radius")  
                bond_idx += 1
                Bonds = []
                while bond_idx < mass_idx:
                    Bonds.append(s[bond_idx])
                    bond_idx += 1
                mass_idx += 1
                Mass = float(s[mass_idx])
                radius_idx += 1
                Radius = float(s[radius_idx])
                X = float(s[radius_idx+1])
                Y = float(s[radius_idx+2])
                Z = float(s[radius_idx+3])
                if symbol not in types_list:
                    types_list.append(symbol)
                    idx_list.append(k)
                    k += 1
                idx = types_list.index(symbol)
                atom = structure.Atom(symbol, j, idx_list[idx], nMolecule, Mass, Radius, Bonds)
                Atoms.append(structure.AtomCoordinates(atom, X, Y, Z))
                j += 1
                i += 1
            nMolecule += 1
        i += 1
    Molecules = []
    for i in range(0, nMolecule, 1):
        MoleculeAtoms = []
        for j in range(0, len(Atoms), 1):
            if Atoms[j].Atom.MolecularIndex == i:
                MoleculeAtoms.append(Atoms[j])
        Molecules.append(structure.Molecule(MoleculeAtoms, Name=MoleculeNames[i]))
    return Molecules
    
def Molecule_to_plot(molecule):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    z1 = []
    z2 = []
    for i in range(0, len(molecule.Bonds), 1):
        i1 = molecule.AtomIndex.index(molecule.Bonds[i][0])
        i2 = molecule.AtomIndex.index(molecule.Bonds[i][1])
        x1.append(molecule.Atoms[i1].x)
        x2.append(molecule.Atoms[i2].x)
        y1.append(molecule.Atoms[i1].y)
        y2.append(molecule.Atoms[i2].y)
        z1.append(molecule.Atoms[i1].z)
        z2.append(molecule.Atoms[i2].z)
    return x1, x2, y1, y2, z1, z2
   
def plot_molecules(Plot, molecule, dot_color='red', bond_color='blue'):
# molecule either one molecule or list of molecules
    if type(molecule) is list:
        for j in molecule:
            plot_molecules(Plot, j, dot_color=dot_color, bond_color=bond_color)
    else:
        S, R = get_atom_coordinates(molecule)
        for i in range(0, S.shape[0], 1):
            Plot.scatter(S[i, 0], S[i, 1], S[i, 2], s=R[i]*100, zdir='z', color=dot_color, depthshade=False)
        x1, x2, y1, y2, z1, z2 = Molecule_to_plot(molecule)
        for i in range(0, len(x1), 1):
            if i == 0:
                Plot.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], label=molecule.Name, color=bond_color, linewidth=1)
            else:
                Plot.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], color=bond_color, linewidth=1)
    return

def generate_random_molecule(prototype, SphereRadius=10, max_trials=100):
# molecule prototype must be aligned by function align_molecule
    n_trials = 0
    while n_trials < max_trials:
        Good = True
        theta = 2*math.pi*random.random()
        phi = math.pi*random.random()
        psi = 2*math.pi*random.random()
        x = 2 * SphereRadius * (random.random() - 0.5)
        y = 2 * SphereRadius * (random.random() - 0.5)
        z = 2 * SphereRadius * (random.random() - 0.5)
        new_center = Point(x, y, z)
        molecule_rotated = rotate_molecule_angles(prototype, theta, phi, psi, AngleType='Radian')
        molecule = translate_molecule_to_new_center(molecule_rotated, new_center)
        for i in range(0, len(molecule.Atoms), 1):
            x = abs(molecule.Atoms[i].x)
            y = abs(molecule.Atoms[i].y)
            z = abs(molecule.Atoms[i].z)
            r = abs(molecule.Atoms[i].Atom.Radius)
            distance_from_origin = np.sqrt(x**2 + y**2 + z**2) + r
            if (distance_from_origin > SphereRadius):
                Good = False
                break
        if Good:
            return molecule
        n_trials += 1
    print('Unable to generate molecule with provided configuration')
    return -1

def generate_random_molecules2(prototype, DMin=0, DMax=10, max_trials=100):
# molecule prototype must be aligned by function align_molecule
# generates one molecule that has center of mass farther than DMin and closer than DMax from COM of prototype
    n_trials = 0
    while n_trials < max_trials:
# angles for molecule rotation        
        theta = 2*math.pi*random.random()
        phi = math.pi*random.random()
        psi = 2*math.pi*random.random()      
# spherical coordinates of new center of mass        
        theta1 = 2*math.pi*random.random()
        phi1 = math.pi*random.random()
        rho = DMin + (DMax - DMin) * random.random()
        x = rho * math.sin(phi1) * math.cos(theta1)
        y = rho * math.sin(phi1) * math.sin(theta1)
        z = rho * math.cos(phi1)
        new_center = Point(x, y, z)
        molecule_rotated = rotate_molecule_angles(prototype, theta, phi, psi, AngleType='Radian')
        molecule = translate_molecule_to_new_center(molecule_rotated, new_center)
        if check_new_molecule2(prototype, molecule, additional_gap=0):
            return molecule
        n_trials += 1
    print('Cannot generate', prototype.Name, 'in interval [', DMin, '..', DMax, ']')
    return None
   
def check_new_molecule2(molecules, molecule, additional_gap=0):
    if type(molecules) is not list:
        molecules = [molecules]
    for j in molecules:
        for k in j.Atoms:
            x1 = k.x
            y1 = k.y
            z1 = k.z
            r1 = k.Atom.Radius
            for l in molecule.Atoms:
                x2 = l.x
                y2 = l.y
                z2 = l.z
                r2 = l.Atom.Radius
                critical_distance = r1 + r2 + additional_gap
                distance = float(np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2))
                if (distance < critical_distance):
                    return False
    return True

def check_new_molecule(molecules, molecule, additional_gap=0):
    if type(molecules) is not list:
        molecules = [molecules]
    Good = True
    for j in molecules:
        for k in j.Atoms:
            x1 = k.x
            y1 = k.y
            z1 = k.z
            r1 = k.Atom.Radius
            for l in molecule.Atoms:
                x2 = l.x
                y2 = l.y
                z2 = l.z
                r2 = l.Atom.Radius
                critical_distance = r1 + r2 + additional_gap
                distance = float(np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2))
                if (distance < critical_distance):
                    Good = False
                    break
            if not Good:
                break
        if not Good:
            break        
    if Good:
        molecules.append(molecule)
    return molecules

def generate_molecule_coordinates_list(prototype, nMolecules=2, nRecords=100,\
        SphereRadius=10, additional_gap=0, PrototypeFirst=False, max_gen_trials=10,\
        max_inner_trials=100, max_outer_trials=10, verbose=False, n_verbose=10):
# generate molecule coordinates inside sphere with radius = Radius
# if nRecords=None - returns molecules list with nMolecules molecules
    records = []
    rec = 0
    if nRecords is None:
        generate_records = False
        nRecords = 1
    else:
        generate_records = True
    while (rec <  nRecords):
        n_outer_trials = 0            
        molecules = []
        if verbose:
            if (rec % n_verbose == 0):
                print(rec)        
        while (len(molecules) < nMolecules) and (n_outer_trials < max_outer_trials):
            molecules = []
            if PrototypeFirst:
                molecules.append(prototype)
            else:
                molecules.append(generate_random_molecule(prototype, SphereRadius=SphereRadius, max_trials=max_gen_trials))
            n_inner_trials = 0
            while (len(molecules) < nMolecules) and (n_inner_trials < max_inner_trials):
                new_molecule = generate_random_molecule(prototype, SphereRadius=SphereRadius, max_trials=max_gen_trials)
                if type(new_molecule) is int:
                    return -1
                molecules = check_new_molecule(molecules, new_molecule, additional_gap=additional_gap)
                n_inner_trials += 1
            n_outer_trials += 1
        if len(molecules) < nMolecules:
            print('Unable to generate moleculeS with provided configuration')
            return -1
        if (not generate_records):
            return molecules
        record = []
        for j in molecules:
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
    return records

def get_distance(x1, y1, z1, x2, y2, z2):
    v = vector_from_coordinates(x1, y1, z1, x2, y2, z2)
    return v.length

def check_molecules(molecules, RMin=0, RMax=100, CheckCenterOfMass=True, nDistances=-1):
# RMin - minimal distance
# EMax - maximal distance 
# nDistances - how many distances must follow the selection criteria
    if RMin is None:
        RMin = 0
    if RMax is None:
        RMax = 1e100
    if nDistances is None:
        CheckAtomDistances = False
    else:
        CheckAtomDistances = True
        Distances = 0
        for i in range(0, len(molecules), 1):
            for k in range(0, len(molecules[i].Atoms), 1):
                for j in range(i+1, len(molecules), 1):
                    for l in range(0, len(molecules[j].Atoms), 1):
                        Distances += 1
        if (nDistances == -1) or (nDistances > Distances):
            nDistances = Distances
    if CheckCenterOfMass:
# check center of mass
        for i in range(0, len(molecules), 1):
            for j in range(i+1, len(molecules), 1):
                r = get_distance(molecules[i].CenterOfMass.x, molecules[i].CenterOfMass.y,\
                    molecules[i].CenterOfMass.z, molecules[j].CenterOfMass.x,\
                    molecules[j].CenterOfMass.y, molecules[j].CenterOfMass.z)
                if r < RMin:
                    return False
                if r > RMax:
                    return False                
    if CheckAtomDistances:
# check all atoms
        good = 0
        for i in range(0, len(molecules), 1):
            for k in range(0, len(molecules[i].Atoms), 1):
                for j in range(i+1, len(molecules), 1):
                    for l in range(0, len(molecules[j].Atoms), 1):
                        r = get_distance(molecules[i].Atoms[k].x, molecules[i].Atoms[k].y,\
                            molecules[i].Atoms[k].z, molecules[j].Atoms[l].x,\
                            molecules[j].Atoms[l].y, molecules[j].Atoms[l].z)
                        if (r > RMin) and (r < RMax): # good distance
                            good += 1
        if good < nDistances:
            return False
    return True              
                
def get_molecules_from_records(system, records, idx=None):
    if type(records) is structure.Record:
        idx = [0]
        rec = [records]
    else:
        if idx is None:
            idx = list(range(0, len(records), 1))
        rec = records
    molecules = []
    for i in idx: # record index
        l = 0 # atom index in records
        for j in range(0, len(system.Molecules), 1): # molecule index in records
            atoms = []
            MoleculeName = system.Molecules[j].Name
            for k in range(0, len(system.Molecules[j].Atoms), 1): # atom index in molecule
                atom = system.Molecules[j].Atoms[k].Atom # type Atom
                atom_c = structure.AtomCoordinates(atom, rec[i].atoms[l].x, rec[i].atoms[l].y, rec[i].atoms[l].z)
                atoms.append(atom_c)
                l += 1
            molecules.append(structure.Molecule(atoms, MoleculeName))
    return molecules
            

                    

    
    