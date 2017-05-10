from collections import namedtuple

# Symbol - atom symbol. Example: O, H, C, Si
# Index - order in the system 0 .. N-1 where N is number of atoms in the system
# Type - atom type identification number. Example atom symbol 'O' corresponds to number 0
# atom symbol 'H' corresponds to number 1
# MolecularIndex - which molecule atom belongs to. In other words number of molecule in the system where this atom exists
class Atom(namedtuple('Atom', ['Symbol', 'Index', 'Type', 'MolecularIndex'], verbose=False)):
    __slots__ = ()

class AtomCoordinates(namedtuple('AtomCoordinates', ['Atom', 'x', 'y', 'z'], verbose=False)):
    __slots__ = ()

class Distance(namedtuple('Distance', ['Atom1', 'Atom2', 'isIntermolecular', 'DiType'], verbose=False)):
    __slots__ = ()
    def __new__(_cls, Atom1, Atom2, isIntermolecular, DiType):
        'Create new instance of Distance(Atom1, Atom2, isIntermolecular, DiType)'
        if (Atom1.MolecularIndex == Atom2.MolecularIndex):
            i = 0
        else:
            i = 1
# compute a UNIQUE identification number for a distance type        
        d = DiType*max(Atom1.Type, Atom2.Type) + min(Atom1.Type, Atom2.Type)+\
            DiType**2 * i
        return tuple.__new__(_cls, (Atom1, Atom2, i, d))

class Angle(namedtuple('Angle', ['Atom1', 'Atom2', 'Atom3', 'isIntermolecular12', 'isIntermolecular13', 'isIntermolecular23', 'AnType'], verbose=False)):
    __slots__ = ()
    
class System(namedtuple('System', ['atoms', 'nAtoms', 'nAtType', 'distances', 'nDistances', 'nDiTypes'], verbose=False)):
    __slots__ = ()

class InvPowDistancesFeature(namedtuple('InvPowDistancesFeature', ['nDistances', 'distances', 'powers', 'FeType'], verbose=False)):
    __slots__ = ()    
    def __new__(_cls, nDistances, distances, powers, FeType):
        'Create new instance of InvPowDistancesFeature'
        return tuple.__new__(_cls, (nDistances, distances, powers, FeType))

def AreTwoFeaturesEquivalent(Feature1, Feature2):
# check number of distances in features (equal or not)
    if (Feature1.nDistances != Feature2.nDistances):
        return False
    else:
        ipow = 0
        while ipow < Feature1.nDistances:
            if (Feature1.distances[ipow].DiType != Feature2.distances[ipow].DiType):
                return False
            ipow += 1
        for j in range(0, Feature1.nDistances, 1):
            if (Feature1.powers[j] != Feature2.powers[j]):
                return False   
    return True

class record:
    def __init__(self, e, atoms):
        self.e = e
        self.atoms = atoms
       
def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False
