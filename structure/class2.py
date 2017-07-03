# Atom
# AtomCoordinates
# record
# Distance
# Distance_to_Power
# Harmonic
# Feature

# max number of different kind of atoms = 9
import numpy as np

class Atom:
    Symbol = None # atom symbol. Example: O, H, C, Si
    Index = None # order in the system 0 .. N-1 where N is number of atoms in the system
    AtType = None # atom type identification number. Example atom symbol 'O' corresponds to number 0
    MolecularIndex = None # which molecule atom belongs to. In other words number of molecule in the system where this atom exists
    def __init__(self, symbol, index, tYpe, molecular_index):
        self.Symbol = symbol
        self.Index = index
        self.AtType = tYpe
        self.MolecularIndex = molecular_index
        
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
        
class record:
    atoms = None
    e = None
    def __init__(self, energy, atoms):
        self.e = energy
        self.atoms = atoms

def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False
        
class Distance:
    Atom1 = None
    Atom2 = None
    isIntermolecular = None
    DiType = None
    def __init__(self, atom1, atom2):
        if atom1.AtType > atom2.AtType:
            self.Atom1 = atom1 # Oxygene first
            self.Atom2 = atom2
        else:
            self.Atom1 = atom2
            self.Atom2 = atom1
        if (atom1.MolecularIndex != atom2.MolecularIndex):
            self.isIntermolecular = True # atoms belong to different molecules
            i = 1
        else:
            self.isIntermolecular = False # atoms belong to the same molecule
            i = 0
        self.DiType = 100*i + 10*max(atom1.AtType, atom2.AtType) + \
            min(atom1.AtType, atom2.AtType)
            
class Distance_to_Power:
# power cannot be zero
    Distance = None
    Power = None
    DtpType = None
    def __init__(self, distance, power):
        self.Distance = distance
        self.Power = power   
        sign = np.sign(power)
        if sign == -1:
            sign = 0
        self.DtpType = distance.DiType + 1000*abs(power) + 100000*sign
        
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
        
class Feature:
    nDistances = 0
    nHarmonics = 0
    DtP1 = None
    DtP2 = None
    FeType = 'None'
    Harmonic1 = None
    Harmonic2 = None
    def __init__(self, DtP1, DtP2=None, Harmonic1=None, Harmonic2=None):
        if DtP1 is not None:
            if DtP2 is not None:
                self.nDistances = 2
            else:
                self.nDistances = 1
        if Harmonic1 is not None:
            if Harmonic2 is not None:
                self.nHarmonics = 2
            else:
                self.nHarmonics = 1
        self.DtP1 = DtP1
        if Harmonic1 is not None:
            self.Harmonic1 = Harmonic1
            h1_type = Harmonic1.HaType
        else:
            h1_type = 0
        if Harmonic2 is not None:
            self.Harmonic2 = Harmonic2
            h2_type = Harmonic2.HaType
        else:
            h2_type = 0
        if DtP1 is not None:
            self.DtP1 = DtP1
            d1_type = DtP1.DtpType
        else:
            d1_type = 0
        if DtP2 is not None:
            self.DtP2 = DtP2
            d2_type = DtP2.DtpType
        else:
            d2_type = 0
        D1 = str(d1_type)   
        D1 = D1.zfill(6)
        D2 = str(d2_type)   
        D2 = D2.zfill(6)
        H1 = str(h1_type)
        H1 = H1.zfill(5)
        H2 = str(h2_type)
        H2 = H2.zfill(5)
        if D1 > D2:
            self.FeType = D1 + D2 + H1 + H2
        else:
            self.FeType = D2 + D1 + H2 + H1
        return
        
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
        
        

        