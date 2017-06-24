# max number of different kind of atoms = 9

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
        self.Atom1 = atom1
        self.Atom2 = atom2
        if (atom1.MolecularIndex != atom2.MolecularIndex):
            self.isIntermolecular = True # atoms belong to different molecules
            i = 1
        else:
            self.isIntermolecular = False # atoms belong to the same molecule
            i = 0
        self.DiType = 100*i + 10*max(atom1.AtType, atom2.AtType) + \
            min(atom1.AtType, atom2.AtType)
            
class Distance_to_Power:
    Distance = None
    Power = None
    DtpType = None
    def __init__(self, distance, power):
        self.Distance = distance
        self.Power = power    
        self.DtpType = distance.DiType + 1000*power
        
class Harmonic:
    Order = None # Can be ... -3, -2, -1, 0, 1, 2, 3 ... use only positive. Less or = to Degree 
    Degree = None # 0, 1, 2, 3 ....
    Center = None # which atom is taken as zero reference
    Basis = None # Method to establish basis
    Atom = None # atom from another molecule that harmonic is calculated for
    HaType = None # type of harmonic. each harmonic is unique I hope
    def __init__(self, order, degree, center, atom, basis='bisection'):
        self.Order = order
        self.Degree = degree
        self.Center = center
        self.Atom = atom
        self.Basis = basis
        
class Feature:
    nDistances = None
    DtP1 = None
    DtP2 = None
    FeType = None
    Harmonic = None
    def __init__(self, nDistances, DtP1, DtP2=None, Harmonic=None):
        self.nDistances = nDistances
        self.DtP1 = DtP1
        if Harmonic is not None:
            self.Harmonic = Harmonic
            return
        if DtP2 is not None:
            self.DtP2 = DtP2
        if DtP2 is None:
            self.FeType = DtP1.DtpType
        if DtP2 is not None:
            T1 = abs(DtP1.DtpType)
            T2 = abs(DtP2.DtpType)
            self.FeType = 100000*min(T1, T2) + max(T1, T2)
            if DtP1.DtpType * DtP2.DtpType < 0:
                self.FeType = self.FeType * (-1)
        
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
        
        
        
        