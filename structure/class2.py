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
    AtTypeDigits = 1 # can be increased
    MolecularIndex = None # which molecule atom belongs to. In other words number of molecule in the system where this atom exists
    Mass = None
    def __init__(self, symbol, index, tYpe, molecular_index):
        self.Symbol = symbol
        self.Index = index
        self.AtType = tYpe
        self.MolecularIndex = molecular_index
        self.AtTypeDigids = 1
        
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
        
class Feature2:
# ALSO WORKS for single and double distances only
    nDistances = 0
    nHarmonics = 0
    DtP1 = None
    DtP2 = None
    FeType = 'None'
    Harmonic1 = None
    Harmonic2 = None
    def __init__(self, DtP1, DtP2=None, Harmonic1=None, Harmonic2=None):
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
            Swapped = False

            if DtP1.Distance.isIntermolecular and (not DtP2.Distance.isIntermolecular): 
                # first inter second intra
                self.DtP1 = DtP2 # intra first
                self.DtP2 = DtP1
                Swapped = True   
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
                            Swapped = True
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
                                Swapped = True
                            else:
                                self.DtP1 = DtP1 
                                self.DtP2 = DtP2  
                        else: # first and second types have equal types
                            self.DtP1 = DtP1 # as it is
                            self.DtP2 = DtP2


            # get category based on molecular index
            Found = False
            if (self.DtP1.Distance.Atom1.MolecularIndex == self.DtP1.Distance.Atom2.MolecularIndex) and \
                (self.DtP2.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex):
                if (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom1.MolecularIndex):
                    CategoryMolecular = 1 # category 1
                    Found = True                    
                if (self.DtP1.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom1.MolecularIndex):
                    CategoryMolecular = 2 # category 2
                    Found = True                        
            if (self.DtP1.Distance.Atom1.MolecularIndex == self.DtP1.Distance.Atom2.MolecularIndex): # first distance intra
                if ((self.DtP2.Distance.Atom1.MolecularIndex == self.DtP1.Distance.Atom1.MolecularIndex) and \
                    (self.DtP2.Distance.Atom2.MolecularIndex != self.DtP1.Distance.Atom1.MolecularIndex)) or \
                    ((self.DtP2.Distance.Atom2.MolecularIndex == self.DtP1.Distance.Atom1.MolecularIndex) and \
                    (self.DtP2.Distance.Atom1.MolecularIndex != self.DtP1.Distance.Atom1.MolecularIndex)):
                    CategoryMolecular = 3 # category 3
                    Found = True                        
            if (self.DtP2.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex): # second distance intra
                if ((self.DtP1.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom1.MolecularIndex) and \
                    (self.DtP1.Distance.Atom2.MolecularIndex != self.DtP2.Distance.Atom1.MolecularIndex)) or \
                    ((self.DtP1.Distance.Atom2.MolecularIndex == self.DtP2.Distance.Atom1.MolecularIndex) and \
                    (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom1.MolecularIndex)):
                    CategoryMolecular = 3 # category 3
                    Found = True                        
            if (self.DtP1.Distance.Atom1.MolecularIndex == self.DtP1.Distance.Atom2.MolecularIndex): # first distance intra
                if (self.DtP2.Distance.Atom1.MolecularIndex != self.DtP1.Distance.Atom1.MolecularIndex) and \
                    (self.DtP2.Distance.Atom2.MolecularIndex != self.DtP1.Distance.Atom1.MolecularIndex) and \
                    (self.DtP2.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom2.MolecularIndex):
                    CategoryMolecular = 4 # category 4
                    Found = True         
            if (self.DtP2.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex): # first distance intra
                if (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom1.MolecularIndex) and \
                    (self.DtP1.Distance.Atom2.MolecularIndex != self.DtP2.Distance.Atom1.MolecularIndex) and \
                    (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP1.Distance.Atom2.MolecularIndex):
                    CategoryMolecular = 4 # category 4
                    Found = True         
            if (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP1.Distance.Atom2.MolecularIndex) and \
                (self.DtP2.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom2.MolecularIndex):
                if ((self.DtP1.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom1.MolecularIndex) and \
                    (self.DtP1.Distance.Atom2.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex)) or \
                    ((self.DtP1.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex) and \
                    (self.DtP1.Distance.Atom2.MolecularIndex == self.DtP2.Distance.Atom1.MolecularIndex)):
                    CategoryMolecular = 5 # category 5
                    Found = True                 
            if (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP1.Distance.Atom2.MolecularIndex) and \
                (self.DtP2.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom2.MolecularIndex):
                if ((self.DtP1.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom1.MolecularIndex) and \
                    (self.DtP1.Distance.Atom2.MolecularIndex != self.DtP2.Distance.Atom2.MolecularIndex)) or \
                    ((self.DtP1.Distance.Atom1.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex) and \
                    (self.DtP1.Distance.Atom2.MolecularIndex != self.DtP2.Distance.Atom1.MolecularIndex)) or \
                    ((self.DtP1.Distance.Atom2.MolecularIndex == self.DtP2.Distance.Atom1.MolecularIndex) and \
                    (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom2.MolecularIndex)) or \
                    ((self.DtP1.Distance.Atom2.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex) and \
                    (self.DtP1.Distance.Atom1.MolecularIndex != self.DtP2.Distance.Atom1.MolecularIndex)):
                    CategoryMolecular = 6 # category 6
                    Found = True    
            if not Found:
                idx_list = [] # make a list with atom's indices
                idx_list.append(self.DtP1.Distance.Atom1.MolecularIndex)
                idx_list.append(self.DtP1.Distance.Atom2.MolecularIndex)
                idx_list.append(self.DtP2.Distance.Atom1.MolecularIndex)
                idx_list.append(self.DtP2.Distance.Atom2.MolecularIndex)
                Dup = False
                for i in range(0, len(idx_list), 1):
                    c = idx_list.count(idx_list[i])
                    if c > 1:
                        Dup = True # if there are duplicates
                if not Dup:
                    CategoryMolecular = 7 # one common atom
                    Found = True
            if not Found:
                CategoryMolecular = 'ERROR'
            # get category based on atomic index
            Found = False
            if (self.DtP1.Distance.Atom1.Index == self.DtP1.Distance.Atom2.Index) and \
                (self.DtP2.Distance.Atom1.Index == self.DtP2.Distance.Atom2.Index):
                if (self.DtP1.Distance.Atom1.Index != self.DtP2.Distance.Atom1.Index):
                    CategoryAtomic = 1 # category 1
                    Found = True                    
                if (self.DtP1.Distance.Atom1.Index == self.DtP2.Distance.Atom1.Index):
                    CategoryAtomic = 2 # category 2
                    Found = True 
            if (self.DtP1.Distance.Atom1.Index == self.DtP1.Distance.Atom2.Index): # first distance intra
                if ((self.DtP2.Distance.Atom1.Index == self.DtP1.Distance.Atom1.Index) and \
                    (self.DtP2.Distance.Atom2.Index != self.DtP1.Distance.Atom1.Index)) or \
                    ((self.DtP2.Distance.Atom2.Index == self.DtP1.Distance.Atom1.Index) and \
                    (self.DtP2.Distance.Atom1.Index != self.DtP1.Distance.Atom1.Index)):
                    CategoryAtomic = 3 # category 3
                    Found = True                        
            if (self.DtP2.Distance.Atom1.Index == self.DtP2.Distance.Atom2.Index): # second distance intra
                if ((self.DtP1.Distance.Atom1.Index == self.DtP2.Distance.Atom1.Index) and \
                    (self.DtP1.Distance.Atom2.Index != self.DtP2.Distance.Atom1.Index)) or \
                    ((self.DtP1.Distance.Atom2.Index == self.DtP2.Distance.Atom1.Index) and \
                    (self.DtP1.Distance.Atom1.Index != self.DtP2.Distance.Atom1.Index)):
                    CategoryAtomic = 3 # category 3
                    Found = True                        
            if (self.DtP1.Distance.Atom1.Index == self.DtP1.Distance.Atom2.Index): # first distance intra
                if (self.DtP2.Distance.Atom1.Index != self.DtP1.Distance.Atom1.Index) and \
                    (self.DtP2.Distance.Atom2.Index != self.DtP1.Distance.Atom1.Index) and \
                    (self.DtP2.Distance.Atom1.Index != self.DtP2.Distance.Atom2.Index):
                    CategoryAtomic = 4 # category 4
                    Found = True         
            if (self.DtP2.Distance.Atom1.Index == self.DtP2.Distance.Atom2.Index): # first distance intra
                if (self.DtP1.Distance.Atom1.Index != self.DtP2.Distance.Atom1.Index) and \
                    (self.DtP1.Distance.Atom2.Index != self.DtP2.Distance.Atom1.Index) and \
                    (self.DtP1.Distance.Atom1.Index != self.DtP1.Distance.Atom2.Index):
                    CategoryAtomic = 4 # category 4
                    Found = True         
            if ((self.DtP1.Distance.Atom1.Index == self.DtP2.Distance.Atom1.Index) and \
                (self.DtP1.Distance.Atom2.Index == self.DtP2.Distance.Atom2.Index)) or \
                ((self.DtP1.Distance.Atom1.Index == self.DtP2.Distance.Atom2.Index) and \
                (self.DtP1.Distance.Atom2.Index == self.DtP2.Distance.Atom1.Index)): # category 5
                 CategoryAtomic = 5 # category 5
                 Found = True                 
            if (self.DtP1.Distance.Atom1.Index != self.DtP1.Distance.Atom2.Index) and \
                (self.DtP2.Distance.Atom1.Index != self.DtP2.Distance.Atom2.Index):
                if ((self.DtP1.Distance.Atom1.Index == self.DtP2.Distance.Atom1.Index) and \
                    (self.DtP1.Distance.Atom2.Index != self.DtP2.Distance.Atom2.Index)) or \
                    ((self.DtP1.Distance.Atom1.Index == self.DtP2.Distance.Atom2.Index) and \
                    (self.DtP1.Distance.Atom2.Index != self.DtP2.Distance.Atom1.Index)) or \
                    ((self.DtP1.Distance.Atom2.Index == self.DtP2.Distance.Atom1.Index) and \
                    (self.DtP1.Distance.Atom1.Index != self.DtP2.Distance.Atom2.Index)) or \
                    ((self.DtP1.Distance.Atom2.Index == self.DtP2.Distance.Atom2.Index) and \
                    (self.DtP1.Distance.Atom1.Index != self.DtP2.Distance.Atom1.Index)):
                    CategoryAtomic = 6 # category 6
                    Found = True    
            if not Found:
                idx_list = [] # make a list with atom's indices
                idx_list.append(self.DtP1.Distance.Atom1.Index)
                idx_list.append(self.DtP1.Distance.Atom2.Index)
                idx_list.append(self.DtP2.Distance.Atom1.Index)
                idx_list.append(self.DtP2.Distance.Atom2.Index)
                Dup = False
                for i in range(0, len(idx_list), 1):
                    c = idx_list.count(idx_list[i])
                    if c > 1:
                        Dup = True # if there are duplicates
                if not Dup:
                    CategoryAtomic = 7 # one common atom
                    Found = True
            if not Found:
                CategoryAtomic = 'ERROR'
                    
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
            Swapped = False
            if DtP1.Distance.isIntermolecular and (not DtP2.Distance.isIntermolecular): 
                # first inter second intra
                self.DtP1 = DtP2 # intra first
                self.DtP2 = DtP1
                Swapped = True   
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
                            Swapped = True
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
                                Swapped = True
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
                    
# not active
class Feature1:
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
                                mol_list = []
                                count_list = []
                                mol_list.append(self.DtP1.Distance.Atom1.MolecularIndex)
                                mol_list.append(self.DtP1.Distance.Atom2.MolecularIndex)
                                mol_list.append(self.DtP2.Distance.Atom1.MolecularIndex)
                                mol_list.append(self.DtP2.Distance.Atom2.MolecularIndex)
                                for i in range(0, len(mol_list), 1):
                                    count_list.append(mol_list.count(mol_list[i]))
                                if count_list.count(2) == 2:
                                    Common = 4 # one common molecule
                                if count_list.count(2) == 4:
                                    Common = 5 # two common molecules                                
                            else: # both inter with common first atom
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
                                if self.DtP1.Distance.Atom1.Symbol == 'O': # first common atom is Oxygen
                                    if self.DtP1.Distance.Atom2.Symbol == 'H' and self.DtP2.Distance.Atom2.Symbol == 'H':
                                        # both second atoms hydrogens
                                        if self.DtP1.Distance.Atom2.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex:
                                            Common = 2 # second atoms belong to same molecule
                                        else:
                                            Common = 3 # second atoms belong to different molecules
                                if self.DtP1.Distance.Atom1.Symbol == 'H': # first common atom is Hydorogen
                                    if self.DtP1.Distance.Atom2.Symbol == 'H' and self.DtP2.Distance.Atom2.Symbol == 'H':
                                        # both second atoms hydrogens
                                        if self.DtP1.Distance.Atom2.MolecularIndex == self.DtP2.Distance.Atom2.MolecularIndex:
                                            Common = 2 # second atoms belong to same molecule
                                        else:
                                            Common = 3 # second atoms belong to different molecules
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
        
class System:
    Atoms=None
    nAtoms=None
    nAtTypes=None
    Distances=None
    nDistances=None
    nDiTypes=None
    def __init__(self, Atoms=None, nAtoms=None, nAtTypes=None, Distances=None, nDistances=None, nDiTypes=None):
        self.Atoms = Atoms
        self.nAtoms = nAtoms
        self.nAtTypes = nAtTypes
        self.Distances = Distances
        self.nDistances = nDistances
        self.nDiTypes = nDiTypes
    
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
    
    
    
    
    
    
    
    
    
    
    