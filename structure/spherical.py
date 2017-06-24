import numpy as np
from math import acos

# n - degree. 0, 1, 2, 3 ...
# m - order. ... -3, -2, -1, 0, 1, 2, 3 ...   abs(m) <= n
# theta = [0..2pi]
# phi = [0..pi]
# s = sph_harm(m, n, theta, phi).real
        
class Point:
    x = None
    y = None 
    z = None
    def __init__(self, X, Y, Z):
        self.x = X
        self.y = Y
        self.z = Z

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

def get_transformation_matrix(new_origin_O, H1, H2, external_atom):
# calculate new basis and transformation matrix
    M_translation = np.eye(4)
    M_rotation = np.eye(4)
    M_translation[0,3] = new_origin_O.x
    M_translation[1,3] = new_origin_O.y
    M_translation[2,3] = new_origin_O.z
    plane = plane_from3points(new_origin_O, H1, H2)
    e1 = get_bisection(H1, new_origin_O, H2) # new X axis. bisection unit vector
    e2 = Vector(plane.a, plane.b, plane.c) # new Y axis. normal to plane towards external atom
    e2 = UnitVector(e2)
    y_direction = vector_from_points(new_origin_O, external_atom) 
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
    
def get_angles(new_origin, H1, H2, external_atom):
    p1 = Vector(external_atom.x + new_origin.x, external_atom.y + new_origin.y, external_atom.z + new_origin.z)
    e1, e2, e3, M_translation, M_rotation = get_transformation_matrix(new_origin, H1, H2, external_atom)
    cos_phi = dot_product3(p1, e3) / p1.length
    phi = acos(cos_phi)
    projection_on_z = projection(p1, e3)
    projection_on_xy = vector_substract(p1, projection_on_z)
    cos_theta = dot_product3(projection_on_xy, e1) / projection_on_xy.length
    theta = acos(cos_theta)
    return theta, phi
    
    
"""
def get_bisection(p1, origin, p3):
    from scipy.optimize import fsolve
# origin - origin of bisection
# bisection between p1 and p3 on the same plane
# slow and not good
    def equations(p):
        d1, d2, d3 = p
        return ((b1-a1)*(d1-a1) + (b2-a2)*(d2-a1) + (b3-a3)*(d3-a3) - (c1-a1)*(d1-a1) - (c2-a2)*(d2-a2) - (c3-a3)*(d3-a3),\
            A*d1 + B*d2 + C*d3 - D,\
            (d1-a1)**2 + (d2-a2)**2 + (d3-a3)**2 - 1)

    plane = plane_from3points(p1, origin, p3)
    A, B, C, D = plane.a, plane.b, plane.c, plane.d
    a1, a2, a3 = origin.x, origin.y, origin.z
    b1, b2, b3 = p1.x, p1.y, p1.z
    c1, c2, c3 = p3.x, p3.y, p3.z
    d1, d2, d3 =  fsolve(equations, (1, 1, 1))
    arrow = Point(d1, d2, d3)
    bisection = vector_from_points(origin, arrow)
    bisection = UnitVector(bisection)
    return bisection
"""    






