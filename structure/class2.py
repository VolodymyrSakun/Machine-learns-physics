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
        self.DiType = 100*i + 10*self.Atom1.AtType + self.Atom2.AtType
            
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
            sign = 0 # negative sign becomes 0 in type record, positive becomes 1
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
        swapped = False
        Common = 0
        h1_type = 0
        h2_type = 0
        d1_type = 0
        d2_type = 0
        if DtP1 is not None:
            if DtP2 is not None:
                self.nDistances = 2 # two distances in feature
                if (DtP1.DtpType == DtP2.DtpType): # distances have same type
                    if (not DtP1.Distance.isIntermolecular) and (not DtP2.Distance.isIntermolecular):
                        # both intra 
                        Common = 1 # have common atom
                        sum1 = DtP1.Distance.Atom1.Index + DtP1.Distance.Atom2.Index
                        sum2 = DtP2.Distance.Atom1.Index + DtP2.Distance.Atom2.Index
                        if sum1 > sum2: # sort according to indices if both intramolecular (O H)
                            self.DtP1 = DtP2
                            self.DtP2 = DtP1
                            swapped = True
                        else:
                            self.DtP1 = DtP1
                            self.DtP2 = DtP2       
                    else: # same type but not both intra
                        if not bool(DtP1.Distance.isIntermolecular * DtP2.Distance.isIntermolecular):
                        # one inter, one intra
                            if DtP1.Distance.isIntermolecular: # intermolecular first
                                self.DtP1 = DtP1
                                self.DtP2 = DtP2    
                            else:
                                self.DtP1 = DtP2
                                self.DtP2 = DtP1 
                                swapped = True                                   
                        else: # both inter
                            if DtP1.Distance.Atom1.Index != DtP2.Distance.Atom1.Index:
                                # no common first atom
                                if DtP1.Distance.Atom1.Index > DtP2.Distance.Atom1.Index:
                                    self.DtP1 = DtP2 # atom with smaller index first
                                    self.DtP2 = DtP1
                                    swapped = True
                                else:
                                    self.DtP1 = DtP1
                                    self.DtP2 = DtP2                                     
                            else:
                                Common = 1
                                # common first atom using indices of second atom to sort
                                if DtP1.Distance.Atom2.Index > DtP2.Distance.Atom2.Index:                                
                                    self.DtP1 = DtP2 # atom with smaller index first
                                    self.DtP2 = DtP1
                                    swapped = True
                                else:
                                    self.DtP1 = DtP1
                                    self.DtP2 = DtP2  
                                if (DtP1.Distance.Atom1.Symbol == 'O') and \
                                    (DtP1.Distance.Atom2.Symbol == 'O') and \
                                    (DtP2.Distance.Atom1.Symbol == 'O') and \
                                    (DtP2.Distance.Atom2.Symbol == 'O'):
                                    Common = 0
                else: # distances with different types
                    if (DtP1.Distance.Atom1.AtType == DtP2.Distance.Atom1.AtType) and\
                        (DtP1.Distance.Atom2.AtType == DtP2.Distance.Atom2.AtType):
                        # atoms 1 and 2 of same types, 1 inter, 1 intra
                        if DtP1.Distance.isIntermolecular: # intermolecular first
                            self.DtP1 = DtP1
                            self.DtP2 = DtP2    
                        else:
                            self.DtP1 = DtP2
                            self.DtP2 = DtP1 
                            swapped = True 
                    else: #at least one atom type is different; order: O-O, O-H, O-H
                        if DtP1.Distance.Atom1.AtType != DtP2.Distance.Atom1.AtType:
                            # first atoms of different types
                            if DtP1.Distance.Atom1.AtType > DtP2.Distance.Atom1.AtType:
                                self.DtP1 = DtP2 # atom with smaller index first
                                self.DtP2 = DtP1
                                swapped = True
                            else:
                                self.DtP1 = DtP1
                                self.DtP2 = DtP2 
                        else: # first atomf of same types, sort using second atom
                            if DtP1.Distance.Atom2.AtType > DtP2.Distance.Atom2.AtType:
                                self.DtP1 = DtP2 # atom with smaller index first
                                self.DtP2 = DtP1
                                swapped = True
                            else:
                                self.DtP1 = DtP1
                                self.DtP2 = DtP2                     
            else: # only one distance
                self.nDistances = 1
                self.DtP1 = DtP1
        if Harmonic1 is not None:
            self.Harmonic1 = Harmonic1
            if Harmonic2 is not None:
                self.nHarmonics = 2
                if swapped:
                    self.Harmonic1 = Harmonic2
                    self.Harmonic2 = Harmonic1
                else:
                    self.Harmonic1 = Harmonic1
                    self.Harmonic2 = Harmonic2                    
                h2_type = self.Harmonic2.HaType
            else:
                h2_type = 0
                self.nHarmonics = 1
            h1_type = self.Harmonic1.HaType   
        if DtP1 is not None:
            d1_type = self.DtP1.DtpType
        if DtP2 is not None:
            d2_type = self.DtP2.DtpType
        D1 = str(d1_type)   
        D1 = D1.zfill(6)
        D2 = str(d2_type)   
        D2 = D2.zfill(6)
        H1 = str(h1_type)
        H1 = H1.zfill(5)
        H2 = str(h2_type)
        H2 = H2.zfill(5)
        C = str(Common)
        self.FeType = C + D1 + D2 + H1 + H2
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
    
    
    
    
    
    
    
    
    
    
    