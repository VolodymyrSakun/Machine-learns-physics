# max number of different kinds of atoms = 9
import numpy as np
from project1 import spherical
from project1 import library

class Atom(dict):
    """
    Symbol = None # atom symbol. Example: O, H, C, Si
    Index = None # order in the system 0 .. N-1 where N is number of atoms in the system
    AtType = None # atom type identification number. Example atom symbol 'O' corresponds to number 0
    AtTypeDigits = 1 # can be increased
    MolecularIndex = None # which molecule atom belongs to. In other words number of molecule in the system where this atom exists
    Mass = None
    Radius = None
    Bonds = [] # integers (indices of atoms)
    x, y, z - atom coordinates
    """
    def __init__(self, Symbol, Index, AtType, MolecularIndex, AtTypeDigits=1,\
            Mass=None, Radius=None, Bonds=None, x=None, y=None, z=None):
        self.Symbol = Symbol
        self.Index = Index
        self.AtType = AtType
        self.MolecularIndex = MolecularIndex
        self.AtTypeDigits = AtTypeDigits
        self.Mass = Mass
        self.Radius = Radius
        self.Bonds = Bonds
        self.x = x
        self.y = y
        self.z = z
        return
    
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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
            
class Molecule(dict):
    """
    Atoms
    Atoms
    Name
    Bonds
    AtomIndex
    CenterOfMass
    """
    def __init__(self, Atoms=None, Name=None, Mass=None):
        if type(Atoms) is Atom:
            Atoms = [Atoms]
        self.Atoms = Atoms # class Atom
        self.nAtoms = len(self.Atoms)
        self.Name = Name
        self.Bonds = []
        self.AtomIndex = None
        self.CenterOfMass = None
        self._refresh()
        
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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
    
    def _refresh(self):
        mass = 0
        MissedMass = False
        l1 = []
        for i in range(0, self.nAtoms, 1):
            l1.append(self.Atoms[i].Index)
            if self.Atoms[i].Mass is not None:
                mass += self.Atoms[i].Mass
            else:
                MissedMass = True
        self.AtomIndex = l1
        for i in range(0, len(self.Atoms), 1):
            if (self.Atoms[i].Bonds is None):                
                break
            if (len(self.Atoms[i].Bonds) == 0):
                break
            atom1_index = self.Atoms[i].Index
            atom2_indices = self.Atoms[i].Bonds # list 1..4 usually
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
                
class System(dict):

    def __init__(self, Atoms=None, Molecules=None, Prototypes=None,\
        nAtoms=None, nAtomTypes=None, nMolecules=None, Distances=None,\
        nDistances=None, nDiTypes=None):
        self.Atoms = Atoms
        self.Molecules = Molecules
        self.Prototypes = Prototypes
        self.nAtoms = nAtoms
        self.nAtomTypes = nAtomTypes
        self.nMolecules = nMolecules
        self.Distances = Distances
        self.nDistances = nDistances
        self.nDiTypes = nDiTypes

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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
            
class RecordAtoms(dict):

    def __init__(self, Atoms=None, Energy=None):
        self.atoms = Atoms # class Atom
        self.e = Energy

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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
    
class Distance(dict):

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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
    
class Distance_to_Power(dict):

    def __init__(self, Distance, Power, PowerDigits=2):
        self.Distance = Distance
        self.Power = Power
        self.PowerDigits = 2
        sign = np.sign(Power)
        if sign == -1:
            sign = 0 # negative sign becomes 0 in type record, positive becomes 1
        p = str(abs(Power))
        p = p.zfill(self.PowerDigits)
        s = str(sign)
        d = str(Distance.DiType)
        self.DtpType = s + p + d

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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
                
def GenFeType(DtP1, DtP2=None, DtP3=None):
    
    def count_unique(d1, d2, d3=None):
        l = []
        l.append(d1[0])
        l.append(d1[1])
        l.append(d2[0])
        l.append(d2[1])
        if d3 is not None:
            l.append(d3[0])
            l.append(d3[1])            
        l = list(set(l))
        return len(l)        
        
    def count_intra(d1, d2):
        if (d1[0] != d1[1]) and (d2[0] != d2[1]):
            return 0
        if (d1[0] == d1[1]) and (d2[0] == d2[1]):
            return 2
        return 1                

    def count_intra3(d1, d2, d3):
        r = 0
        l1 = list(set(d1)) 
        c1 = len(l1) # 1 or 2
        if c1 == 1:
            r += 1
        l2 = list(set(d2))
        c2 = len(l2)
        if c2 == 1:
            r += 1
        l3 = list(set(d3))
        c3 = len(l3)
        if c3 == 1:
            r += 1
        return r

    def Swap(a, b):
        tmp = a
        a = b
        b = tmp
        return a, b

    def count_v_nodes(d1, d2, d3):
        D1 = (min(d1[0], d1[1]), max(d1[0], d1[1]))
        D2 = (min(d2[0], d2[1]), max(d2[0], d2[1]))
        D3 = (min(d3[0], d3[1]), max(d3[0], d3[1]))
        if D3[0] < D2[0]:
            Swap(D3, D2)
        if D2[0] < D1[0]:
            Swap(D2, D1)
        if D3[0] < D2[0]:
            Swap(D3, D2)
        k = 0
        if D1[1] == D2[0]:
            k += 1
        if D2[1] == D3[0]:
            k += 1    
        return k
                
    if (DtP1 is not None) and (DtP2 is None): # one distance in feature
        # get category based on molecular index
        CategoryAtomic = 5 # can only be 5 for single distance
        # get category based on atomic index
        if DtP1.Distance.isIntermolecular:
            CategoryMolecular = 5 # intermolecular
        else:
            CategoryMolecular = 2 # intramolecular
        self_DtP1 = DtP1
        p1 = str(abs(self_DtP1.Power))
        p1 = p1.zfill(DtP1.PowerDigits)
        t1 = str(self_DtP1.Distance.Atom1.AtType)
        t1 = t1.zfill(self_DtP1.Distance.Atom1.AtTypeDigits)
        t2 = str(self_DtP1.Distance.Atom2.AtType)
        t2 = t2.zfill(self_DtP1.Distance.Atom2.AtTypeDigits)
        c1 = str(CategoryMolecular)
        c2 = str(CategoryAtomic)
        FeType = p1 + t1 + t2 + c1 + c2
        return FeType
    
    if (DtP1 is not None) and (DtP2 is not None) and (DtP3 is not None): # three distances in feature
        m = count_unique((DtP1.Distance.Atom1.MolecularIndex, DtP1.Distance.Atom2.MolecularIndex), \
            (DtP2.Distance.Atom1.MolecularIndex, DtP2.Distance.Atom2.MolecularIndex),\
            (DtP3.Distance.Atom1.MolecularIndex, DtP3.Distance.Atom2.MolecularIndex))
        n = count_intra((DtP1.Distance.Atom1.MolecularIndex, DtP1.Distance.Atom2.MolecularIndex), \
            (DtP2.Distance.Atom1.MolecularIndex, DtP2.Distance.Atom2.MolecularIndex),\
            (DtP3.Distance.Atom1.MolecularIndex, DtP3.Distance.Atom2.MolecularIndex))
        o = count_v_nodes((DtP1.Distance.Atom1.MolecularIndex, DtP1.Distance.Atom2.MolecularIndex), \
            (DtP2.Distance.Atom1.MolecularIndex, DtP2.Distance.Atom2.MolecularIndex),\
            (DtP3.Distance.Atom1.MolecularIndex, DtP3.Distance.Atom2.MolecularIndex))
        CategoryMolecular = '{}{}{}'.format(m, n, o)
        m = count_unique((DtP1.Distance.Atom1.Index, DtP1.Distance.Atom2.Index), \
            (DtP2.Distance.Atom1.Index, DtP2.Distance.Atom2.Index),\
            (DtP3.Distance.Atom1.Index, DtP3.Distance.Atom2.Index))
        n = count_intra((DtP1.Distance.Atom1.Index, DtP1.Distance.Atom2.Index), \
                    (DtP2.Distance.Atom1.Index, DtP2.Distance.Atom2.Index),\
                    (DtP3.Distance.Atom1.Index, DtP3.Distance.Atom2.Index))
        o = count_v_nodes((DtP1.Distance.Atom1.Index, DtP1.Distance.Atom2.Index), \
                    (DtP2.Distance.Atom1.Index, DtP2.Distance.Atom2.Index),\
                    (DtP3.Distance.Atom1.Index, DtP3.Distance.Atom2.Index))   
        CategoryAtomic = '{}{}{}'.format(m, n, o)
        p1 = str(abs(DtP1.Power))
        p1 = p1.zfill(DtP1.PowerDigits)
        p2 = str(abs(DtP2.Power))
        p2 = p2.zfill(DtP2.PowerDigits)
        p3 = str(abs(DtP3.Power))
        p3 = p3.zfill(DtP3.PowerDigits)
        t11 = str(DtP1.Distance.Atom1.AtType)
        t11 = t11.zfill(DtP1.Distance.Atom1.AtTypeDigits)
        t12 = str(DtP1.Distance.Atom2.AtType)
        t12 = t12.zfill(DtP1.Distance.Atom2.AtTypeDigits)
        t21 = str(DtP2.Distance.Atom1.AtType)
        t21 = t21.zfill(DtP2.Distance.Atom1.AtTypeDigits)
        t22 = str(DtP2.Distance.Atom2.AtType)
        t22 = t22.zfill(DtP2.Distance.Atom2.AtTypeDigits)
        t31 = str(DtP3.Distance.Atom1.AtType)
        t31 = t31.zfill(DtP3.Distance.Atom1.AtTypeDigits)
        t32 = str(DtP3.Distance.Atom2.AtType)
        t32 = t32.zfill(DtP3.Distance.Atom2.AtTypeDigits)
        FeType = p1 + p2 + p3 + t11 + t12 + t21 + t22 + t31 + t32 + CategoryMolecular + CategoryAtomic
        return FeType
    
    if (DtP1 is not None) and (DtP2 is not None): # two distances in feature
        # arrange distances
        if DtP1.Distance.isIntermolecular and (not DtP2.Distance.isIntermolecular): 
            # first inter second intra
            self_DtP1 = DtP2 # intra first
            self_DtP2 = DtP1  
        else:
            if DtP2.Distance.isIntermolecular and (not DtP1.Distance.isIntermolecular):
            # first intra second inter
                self_DtP1 = DtP1 # intra first
                self_DtP2 = DtP2
            else: # both inter or both intra
                if DtP1.Distance.Atom1.AtType != DtP2.Distance.Atom1.AtType: 
            # first atoms of different types, sort according to first atom types
                    if DtP1.Distance.Atom1.AtType > DtP2.Distance.Atom1.AtType: 
                        self_DtP1 = DtP2 # lowest type first
                        self_DtP2 = DtP1
                    else:
                        self_DtP1 = DtP1
                        self_DtP2 = DtP2                    
                else: # first atoms of same types
                    if DtP1.Distance.Atom2.AtType != DtP2.Distance.Atom2.AtType: 
                # first atoms are of same types, second atoms different types
                # sort according to second atom types
                        if DtP1.Distance.Atom2.AtType > DtP2.Distance.Atom2.AtType:
                            self_DtP1 = DtP2 # lowest index first
                            self_DtP2 = DtP1
                        else:
                            self_DtP1 = DtP1 
                            self_DtP2 = DtP2  
                    else: # first and second types have equal types
                        self_DtP1 = DtP1 # as it is
                        self_DtP2 = DtP2
        # get category based on molecular index
        m = count_unique((DtP1.Distance.Atom1.MolecularIndex, DtP1.Distance.Atom2.MolecularIndex), \
            (DtP2.Distance.Atom1.MolecularIndex, DtP2.Distance.Atom2.MolecularIndex))
        if m == 1:
            CategoryMolecular = 2
        else:
            if m == 4:
                CategoryMolecular = 7
            else:
                n = count_intra((DtP1.Distance.Atom1.MolecularIndex, DtP1.Distance.Atom2.MolecularIndex), \
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
                n = count_intra((DtP1.Distance.Atom1.Index, DtP1.Distance.Atom2.Index), \
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
        p1 = str(abs(self_DtP1.Power))
        p1 = p1.zfill(DtP1.PowerDigits)
        p2 = str(abs(self_DtP2.Power))
        p2 = p2.zfill(DtP2.PowerDigits)
        t11 = str(self_DtP1.Distance.Atom1.AtType)
        t11 = t11.zfill(self_DtP1.Distance.Atom1.AtTypeDigits)
        t12 = str(self_DtP1.Distance.Atom2.AtType)
        t12 = t12.zfill(self_DtP1.Distance.Atom2.AtTypeDigits)
        t21 = str(self_DtP2.Distance.Atom1.AtType)
        t21 = t21.zfill(self_DtP2.Distance.Atom1.AtTypeDigits)
        t22 = str(self_DtP2.Distance.Atom2.AtType)
        t22 = t22.zfill(self_DtP2.Distance.Atom2.AtTypeDigits)
        c1 = str(CategoryMolecular)
        c2 = str(CategoryAtomic)
        FeType = p1 + p2 + t11 + t12 + t21 + t22 + c1 + c2
        return FeType
                
class Feature(dict):
    
    def __init__(self, FeType='Linear', DtP1=None, DtP2=None, DtP3=None, nDistances=1, nConstants=1):
# FeType = Linear, Exp
# a0*DtP1 - Linear single
# a0*DtP1*DtP2 - Linear Double
# a0*exp(a1*r1) - Exponential simple, nDistances=0, nConstants=2
# a0*exp(a1*r1)*r1**n = nDistances=1, nConstants=2
# a0*exp(a1*r1+a2*r1) (r1**n * r2**m) = nDistances=2, nConstants=3
        self.DtP1=DtP1
        self.DtP2=DtP2
        self.DtP3=DtP3
        self.nDistances=nDistances
        self.nConstants=nConstants
        self.idx = []
        self.FeType = '{}{}{}{}'.format(nDistances, nConstants, FeType, GenFeType(DtP1, DtP2))

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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 

class RecordMolecules(dict):
    
    def __init__(self, Molecules, E_True=None, E_Predicted=None, MSE=None):
        self.Molecules = Molecules
        self.E_True = E_True
        self.E_Predicted = E_Predicted
        self.MSE = MSE
        self.nMolecules = len(Molecules)
        self.CenterOfMass = spherical.center_of_mass(Molecules)
        self.R_Average = None # average distance between COM of two molecules (used for 2 molecules)
        self.R_CenterOfMass_Average = None # average distance between COM of molecules to COM of system(for 3+ molecules)
        R = 0
        for molecule in Molecules:
            v = spherical.vector_from_points(self.CenterOfMass, molecule.CenterOfMass)
            R += v.length        
        self.R_CenterOfMass_Average = R / self.nMolecules
        R = 0
        n = 0
        if self.nMolecules > 1:
            for i1 in range(0, self.nMolecules, 1):
                for i2 in range(i1+1, self.nMolecules, 1):
                    v = spherical.vector_from_points(Molecules[i1].CenterOfMass, Molecules[i2].CenterOfMass)
                    R += v.length   
                    n += 1 
            self.R_Average = R / n
        return

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
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys()) 
        
# only for linear now    
    def predict(self, chromosome, FeaturesAll, FeaturesReduced):
        idx_lin = chromosome.get_genes_list(Type=0)
        coef = chromosome.get_coeff_list(Type=0)
        E = 0
        for i in range(0, chromosome.Size, 1): # for each nonzero coefficient
            current_feature_idx = idx_lin[i] # idx of FeaturesReduced
            variable = 0
            for j in FeaturesReduced[current_feature_idx].idx:
                if FeaturesAll[j].nDistances == 1:
                    atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                    atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                    for molecule in self.Molecules:
                        for atom in molecule.Atoms:
                            if atom.Index == atom1_index:
                                x1 = atom.x
                                y1 = atom.y
                                z1 = atom.z
                            if atom.Index == atom2_index:
                                x2 = atom.x
                                y2 = atom.y
                                z2 = atom.z                               
                    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)            
                    r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
                if FeaturesAll[j].nDistances == 2:
                    atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                    atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                    atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                    atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                    for molecule in self.Molecules:
                        for atom in molecule.Atoms:
                            if atom.Index == atom11_index:
                                x11 = atom.x
                                y11 = atom.y
                                z11 = atom.z
                            if atom.Index == atom12_index:
                                x12 = atom.x
                                y12 = atom.y
                                z12 = atom.z       
                            if atom.Index == atom21_index:
                                x21 = atom.x
                                y21 = atom.y
                                z21 = atom.z
                            if atom.Index == atom22_index:
                                x22 = atom.x
                                y22 = atom.y
                                z22 = atom.z                                     
                    d1 = np.sqrt((x11 - x12)**2 + (y11 - y12)**2 + (z11 - z12)**2)            
                    r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                    d2 = np.sqrt((x21 - x22)**2 + (y21 - y22)**2 + (z21 - z22)**2)            
                    r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power
                    r = r1 * r2      
                variable += r
            E += variable * coef[i] # combined features by coefficient
        self.E_Predicted = E
        self.MSE = (self.E_True - self.E_Predicted)**2
        return 
   

