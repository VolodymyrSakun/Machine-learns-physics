# class Atom
# class AtomCoordinates
# class Molecule
# class Record
# class Distance
# class Distance_to_Power
# class Harmonic - inactive
# class Feature
# class System
# function print_feature
# function print_atom
# function print_distance

# max number of different kind of atoms = 9
import numpy as np
from project1 import spherical
from project1 import library

class Atom:
    Symbol = None # atom symbol. Example: O, H, C, Si
    Index = None # order in the system 0 .. N-1 where N is number of atoms in the system
    AtType = None # atom type identification number. Example atom symbol 'O' corresponds to number 0
    AtTypeDigits = 1 # can be increased
    MolecularIndex = None # which molecule atom belongs to. In other words number of molecule in the system where this atom exists
    Mass = None
    Radius = None
    Bonds = [] # integers (indices of atoms)
    def __init__(self, symbol, index, tYpe, molecular_index, Mass=None, Radius=None, Bonds=None):
        self.Symbol = symbol
        self.Index = index
        self.AtType = tYpe
        self.MolecularIndex = molecular_index
        self.AtTypeDigids = 1
        self.Mass = Mass
        self.Radius = Radius
        self.Bonds = Bonds
        
class AtomCoordinates:
    Atom = None
    x = None
    y = None
    z = None
    def __init__(self, atom, X, Y, Z):
        self.Atom = atom
        self.x = X
        self.y = Y
        self.z = Z        
        
class Molecule:
    Atoms = [] # type AtomCoordinates
    AtomIndex = []
    nAtoms = None
    Bonds = [] # list of tuples, integers (indices of atoms)
    Name = None
    Mass = None
    CenterOfMass = None # type of spherical.Point
    def __init__(self, atoms, Name=None):
        if type(atoms[0]) is Atom:
            Atoms = []
            for i in atoms:
                atom = AtomCoordinates(i, None, None, None)
                Atoms.append(atom)
            self.Atoms = Atoms
        else:
            self.Atoms = atoms # class AtomCoordinates
        self.nAtoms = len(self.Atoms)
        self.Name = Name
        self.Bonds = []
        self._refresh()
    
    def _refresh(self):
        mass = 0
        MissedMass = False
        l1 = []
        for i in range(0, self.nAtoms, 1):
            l1.append(self.Atoms[i].Atom.Index)
            if self.Atoms[i].Atom.Mass is not None:
                mass += self.Atoms[i].Atom.Mass
            else:
                MissedMass = True
        self.AtomIndex = l1
        for i in range(0, len(self.Atoms), 1):
            if (self.Atoms[i].Atom.Bonds is None):                
                break
            if (len(self.Atoms[i].Atom.Bonds) == 0):
                break
            atom1_index = self.Atoms[i].Atom.Index
            atom2_indices = self.Atoms[i].Atom.Bonds # list 1..4 usually
            if atom2_indices is None: 
                continue
            if len(atom2_indices) == 0:
                continue
            else:
                for j in range(0, len(atom2_indices), 1):
                    atom2_index = atom2_indices[j]
                    if atom1_index > atom2_index:
                        a1, a2 = library.Swap(atom1_index, atom2_index)
                    else:
                        a1, a2 = atom1_index, atom2_index
                    new_bond = (a1, a2)
                    if new_bond not in self.Bonds:
                        self.Bonds.append(new_bond)
        if not MissedMass:
            self.Mass = mass
        else:
            self.Mass = None       
        p = spherical.center_of_mass(self.Atoms) # returns False if missing coordinates
        if type(p) is not bool: 
            self.CenterOfMass = p
        else:
            self.CenterOfMass = None
                
class System:
    Atoms=None
    Molecules=None
    Prototypes=None
    nAtoms=None
    nAtTypes=None
    nMolecules=None
    Distances=None
    nDistances=None
    nDiTypes=None
    def __init__(self, Atoms=None, Molecules=None, Prototypes=None,\
        nAtoms=None, nAtTypes=None, nMolecules=None, Distances=None,\
        nDistances=None, nDiTypes=None):
        self.Atoms = Atoms
        self.Molecules = Molecules
        self.Prototypes = Prototypes
        self.nAtoms = nAtoms
        self.nAtTypes = nAtTypes
        self.nMolecules = nMolecules
        self.Distances = Distances
        self.nDistances = nDistances
        self.nDiTypes = nDiTypes
            
class Record:
    atoms = None
    e = None
    def __init__(self, energy, atoms):
        self.e = energy
        self.atoms = atoms # class AtomCoordinates

class Distance:
    Atom1 = None
    Atom2 = None
    isIntermolecular = None
    DiType = None
    def __init__(self, atom1, atom2):
        if (atom1.MolecularIndex != atom2.MolecularIndex):
            self.isIntermolecular = True # atoms belong to different molecules
            i = 1
        else:
            self.isIntermolecular = False # atoms belong to the same molecule
            i = 0
        if atom1.AtType == atom2.AtType: # atoms of same type
            if atom1.MolecularIndex != atom2.MolecularIndex: # atoms of same type belong to different molecules
                if atom1.MolecularIndex > atom2.MolecularIndex:
                    self.Atom1 = atom2 # lowest molecular index first
                    self.Atom2 = atom1  
                else:
                    self.Atom1 = atom1 
                    self.Atom2 = atom2  
            else: # atoms of same type belong to same molecule
                if atom1.Index > atom2.Index:
                    self.Atom1 = atom2 # lowest atom index first
                    self.Atom2 = atom1  
                else:
                    self.Atom1 = atom1 
                    self.Atom2 = atom2  
        else: # atoms of different types
            if atom1.AtType > atom2.AtType:
                self.Atom1 = atom2 # lowest type first
                self.Atom2 = atom1
            else:
                self.Atom1 = atom1
                self.Atom2 = atom2
        self.DiType = 10**self.Atom1.AtTypeDigits*10**self.Atom1.AtTypeDigits*i + 10**self.Atom1.AtTypeDigits*self.Atom1.AtType + self.Atom2.AtType
            
class Distance_to_Power:
# power cannot be zero
    Distance = None
    Power = None
    PowerDigits = 2 # can be increased if necessary
    DtpType = 'None'
    def __init__(self, distance, power):
        self.Distance = distance
        self.Power = power   
        self.PowerDigits = 2
        sign = np.sign(power)
        if sign == -1:
            sign = 0 # negative sign becomes 0 in type record, positive becomes 1
        p = str(abs(power))
        p = p.zfill(self.PowerDigits)
        s = str(sign)
        d = str(distance.DiType)
        self.DtpType = s + p + d
        
# inactive
class Harmonic:
    Order = None # Can be ... -3, -2, -1, 0, 1, 2, 3 ... use only positive. Less or = to Degree 
    Degree = None # 0, 1, 2, 3 ....
    Center = None # which atom is taken as zero reference
    Atom = None # atom from another molecule that harmonic is calculated for
    HaType = None # type of harmonic. each harmonic is unique I hope
    def __init__(self, order, degree, center, atom):
        self.Order = order
        self.Degree = degree
        self.Center = center
        self.Atom = atom
        if order != 0:
            sign = np.sign(order)
        else:
            sign = 1
        if sign == -1:
            sign = 0
        self.HaType = 10000*sign + 1000*abs(self.Order) + 100*self.Degree + 10*self.Center.AtType +self.Atom.AtType
        
# only for single and double distances
class Feature:
    nDistances = 0
    nHarmonics = 0
    DtP1 = None
    DtP2 = None
    FeType = 'None'
    Harmonic1 = None
    Harmonic2 = None
    def __init__(self, DtP1, DtP2=None, Harmonic1=None, Harmonic2=None):
        
        def count_unique(d1, d2):
            l = []
            l.append(d1[0])
            l.append(d1[1])
            l.append(d2[0])
            l.append(d2[1])
            l = list(set(l))
            return len(l)        
        
        def count_bonds(d1, d2):
            if (d1[0] != d1[1]) and (d2[0] != d2[1]):
                return 0
            if (d1[0] == d1[1]) and (d2[0] == d2[1]):
                return 2
            return 1                
        
        if (DtP1 is not None) and (DtP2 is None): # one distance in feature
            self.nDistances = 1
            # get category based on molecular index
            CategoryAtomic = 5 # can only be 5 for single distance
            # get category based on atomic index
            if DtP1.Distance.isIntermolecular:
                CategoryMolecular = 5 # intermolecular
            else:
                CategoryMolecular = 2 # intramolecular
            self.DtP1 = DtP1
            p1 = str(abs(self.DtP1.Power))
            p1 = p1.zfill(DtP1.PowerDigits)
            t1 = str(self.DtP1.Distance.Atom1.AtType)
            t1 = t1.zfill(self.DtP1.Distance.Atom1.AtTypeDigits)
            t2 = str(self.DtP1.Distance.Atom2.AtType)
            t2 = t2.zfill(self.DtP1.Distance.Atom2.AtTypeDigits)
            c1 = str(CategoryMolecular)
            c2 = str(CategoryAtomic)
            self.FeType = p1 + t1 + t2 + c1 + c2
            return
        if (DtP1 is not None) and (DtP2 is not None): # two distances in feature
            self.nDistances = 2
            # arrange distances
#            Swapped = False
            if DtP1.Distance.isIntermolecular and (not DtP2.Distance.isIntermolecular): 
                # first inter second intra
                self.DtP1 = DtP2 # intra first
                self.DtP2 = DtP1
#                Swapped = True   
            else:
                if DtP2.Distance.isIntermolecular and (not DtP1.Distance.isIntermolecular):
                # first intra second inter
                    self.DtP1 = DtP1 # intra first
                    self.DtP2 = DtP2
                else: # both inter or both intra
                    if DtP1.Distance.Atom1.AtType != DtP2.Distance.Atom1.AtType: 
                # first atoms of different types, sort according to first atom types
                        if DtP1.Distance.Atom1.AtType > DtP2.Distance.Atom1.AtType: 
                            self.DtP1 = DtP2 # lowest type first
                            self.DtP2 = DtP1
#                            Swapped = True
                        else:
                            self.DtP1 = DtP1
                            self.DtP2 = DtP2                    
                    else: # first atoms of same types
                        if DtP1.Distance.Atom2.AtType != DtP2.Distance.Atom2.AtType: 
                    # first atoms are of same types, second atoms different types
                    # sort according to second atom types
                            if DtP1.Distance.Atom2.AtType > DtP2.Distance.Atom2.AtType:
                                self.DtP1 = DtP2 # lowest index first
                                self.DtP2 = DtP1
#                                Swapped = True
                            else:
                                self.DtP1 = DtP1 
                                self.DtP2 = DtP2  
                        else: # first and second types have equal types
                            self.DtP1 = DtP1 # as it is
                            self.DtP2 = DtP2
            # get category based on molecular index
            m = count_unique((DtP1.Distance.Atom1.MolecularIndex, DtP1.Distance.Atom2.MolecularIndex), \
                (DtP2.Distance.Atom1.MolecularIndex, DtP2.Distance.Atom2.MolecularIndex))
            if m == 1:
                CategoryMolecular = 2
            else:
                if m == 4:
                    CategoryMolecular = 7
                else:
                    n = count_bonds((DtP1.Distance.Atom1.MolecularIndex, DtP1.Distance.Atom2.MolecularIndex), \
                        (DtP2.Distance.Atom1.MolecularIndex, DtP2.Distance.Atom2.MolecularIndex))
                    if (m == 2) and (n == 1):
                        CategoryMolecular = 3
                    if (m == 2) and (n == 2):
                        CategoryMolecular = 1
                    if (m == 2) and (n == 0):
                        CategoryMolecular = 5
                    if (m == 3) and (n == 1):
                        CategoryMolecular = 4
                    if (m == 3) and (n == 0):
                        CategoryMolecular = 6
            # get category based on atomic index
            m = count_unique((DtP1.Distance.Atom1.Index, DtP1.Distance.Atom2.Index), \
                (DtP2.Distance.Atom1.Index, DtP2.Distance.Atom2.Index))
            if m == 1:
                CategoryAtomic = 2
            else:
                if m == 4:
                    CategoryAtomic = 7
                else:
                    n = count_bonds((DtP1.Distance.Atom1.Index, DtP1.Distance.Atom2.Index), \
                        (DtP2.Distance.Atom1.Index, DtP2.Distance.Atom2.Index))
                    if m == 2 and n == 1:
                        CategoryAtomic = 3
                    if m == 2 and n == 2:
                        CategoryAtomic = 1
                    if m == 2 and n == 0:
                        CategoryAtomic = 5
                    if m == 3 and n == 1:
                        CategoryAtomic = 4
                    if m == 3 and n == 0:
                        CategoryAtomic = 6
        p1 = str(abs(self.DtP1.Power))
        p1 = p1.zfill(DtP1.PowerDigits)
        p2 = str(abs(self.DtP2.Power))
        p2 = p2.zfill(DtP2.PowerDigits)
        t11 = str(self.DtP1.Distance.Atom1.AtType)
        t11 = t11.zfill(self.DtP1.Distance.Atom1.AtTypeDigits)
        t12 = str(self.DtP1.Distance.Atom2.AtType)
        t12 = t12.zfill(self.DtP1.Distance.Atom2.AtTypeDigits)
        t21 = str(self.DtP2.Distance.Atom1.AtType)
        t21 = t21.zfill(self.DtP2.Distance.Atom1.AtTypeDigits)
        t22 = str(self.DtP2.Distance.Atom2.AtType)
        t22 = t22.zfill(self.DtP2.Distance.Atom2.AtTypeDigits)
        c1 = str(CategoryMolecular)
        c2 = str(CategoryAtomic)
        self.FeType = p1 + p2 + t11 + t12 + t21 + t22 + c1 + c2
        return
    
class FeatureNonlinear:
    idx = None
    Distance = None
    nDistances = None
    FeType = 'exp'
    nConstants = None 
    
    def __init__(self, idx, Distance, FeType='exp', nDistances=1, nConstants=2):
        self.idx = idx
        self.Distance = Distance
        self.FeType=FeType
        self.nDistances=nDistances
        self.nConstants=nConstants
        return
        
# Auxilary functions that print objects

def print_feature(feature):
    print('Number of distances = ', feature.nDistances)
    print('Feature type = ', feature.FeType)
    if feature.Harmonic is None:
        print('Not Harmonic')
    print('Distance 1:')
    print('Distance to power type = ', feature.DtP1.DtpType)
    print('Power = ', feature.DtP1.Power)
    print('Distance type = ', feature.DtP1.Distance.DiType)
    print('Is intermolecular? ', feature.DtP1.Distance.isIntermolecular)
    print('Atom1 = ', feature.DtP1.Distance.Atom1.Symbol)
    print('Atom2 = ', feature.DtP1.Distance.Atom2.Symbol)
    print('Atom1 Index = ', feature.DtP1.Distance.Atom1.Index)
    print('Atom2 Index = ', feature.DtP1.Distance.Atom2.Index)
    if feature.nDistances == 2:
        print('Distance 2:')
        print('Distance to power type = ', feature.DtP2.DtpType)
        print('Power = ', feature.DtP2.Power)
        print('Distance type = ', feature.DtP2.Distance.DiType)
        print('Is intermolecular? ', feature.DtP2.Distance.isIntermolecular)
        print('Atom1 = ', feature.DtP2.Distance.Atom1.Symbol)
        print('Atom2 = ', feature.DtP2.Distance.Atom2.Symbol)
        print('Atom1 Index = ', feature.DtP2.Distance.Atom1.Index)
        print('Atom2 Index = ', feature.DtP2.Distance.Atom2.Index)
        
def print_atom(atom):
    print('Symbol: ', atom.Symbol)
    print('Index: ', atom.Index)
    print('Atom type: ', atom.AtType)
    print('Molecular index', atom.MolecularIndex)
    
def print_distance(distance):
    print('Is intermolecular: ', distance.isIntermolecular)  
    print('Distance type: ', distance.DiType)
    print('Atom1:')
    print_atom(distance.Atom1)
    print('Atom2:')
    print_atom(distance.Atom2)
    