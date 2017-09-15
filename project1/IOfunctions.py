import re
import os
from project1 import structure
from project1 import library
from project1 import spherical
import pandas as pd
import pickle 

Separators = '=|,| |:|;|: |'   

def ReadMoleculeDescription(F):    
# read descriptor from file
    with open(F) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string
    if ('&MoleculeDescription' in lines):
        i = 0 # index of line in file
        Molecules = [] # create list of atom structures from file
        while ((lines[i].find('&MoleculeDescription') == -1) and (i < len(lines))):
            i += 1
        i += 1 # go to next line
        j = 0
        nMolecule = 0
        atoms = []
        types_list = []
        idx_list = []
        k = 0
        while ((lines[i].find('&endMoleculeDescription') == -1) and (i < len(lines))):
            if (lines[i][0] == '#'):
                i += 1
                continue
            x = lines[i]      
            if len(x) == 0: # empty line
                i += 1
                continue
            if x.find('&Molecule') != -1:
                s = re.split(Separators, x)
                del(s[0])
                s0 = s[0]
                for l in range(1, len(s), 1):
                    s0 = s0 + ' ' + s[l]
                MoleculeName = s0
                i += 1    
                continue
            if x.find('&endMolecule') != -1:
                Molecules.append(structure.Molecule(atoms, Name=MoleculeName))
                nMolecule += 1
                atoms = []      
                i += 1
                continue
            s = re.split(Separators, x)
            s = list(filter(bool, s))
            try:
                index = int(s[0])
            except:
                print('Error reading file')
                break                
            symbol_idx = s.index("Symbol")
            symbol_idx += 1
            symbol = s[symbol_idx]
            bond_idx = s.index("Bond")
            mass_idx = s.index("Mass")   
            radius_idx = s.index("Radius")  
            coord_idx = s.index("Coordinates")
            bond_idx += 1
            Bonds = []
            while bond_idx < mass_idx:
                Bonds.append(int(s[bond_idx]))
                bond_idx += 1
            mass_idx += 1
            Mass = float(s[mass_idx])
            radius_idx += 1
            Radius = float(s[radius_idx])
            X = float(s[coord_idx+1])
            Y = float(s[coord_idx+2])
            Z = float(s[coord_idx+3])
            if symbol not in types_list:
                types_list.append(symbol)
                idx_list.append(k)
                k += 1
            idx = types_list.index(symbol)
            atom = structure.Atom(symbol, index, idx_list[idx], nMolecule, Mass, Radius, Bonds)
            atoms.append(structure.AtomCoordinates(atom, X, Y, Z))
            j += 1
            i += 1
    return Molecules

# reads feature set and objects that describes it
def ReadFeatures(F_Nonlinear=None, F_linear=None, F_Response=None, F_NonlinearFeatures=None,\
        F_FeaturesAll=None, F_FeaturesReduced=None, F_System=None, F_Records=None, verbose=False):
# F_features - .csv file that contains data stored by Generape combined features.py
# F_FeaturesAll - .dat file that contains object with all informations about features
# F_FeaturesReduced - .dat file that contains object with all informations about reduced features
# F_System - .dat file that contains object Class.System
# F_Records - .dat file that contains list of records 
# returns:
# X - [n x m] numpy array of features
# rows correspond to observations
# columns correspond to features
# Y - [n x 1] numpy array, recponse variable
# FeaturesAll - class1.InvPowDistancesFeature object. Contains all features
# FeaturesReduced - class1.InvPowDistancesFeature object. Contains combined features
    if verbose:
        print('Reading data from file')
# load features for non-linear fit
    if (F_Nonlinear is not None) and os.path.isfile(F_Nonlinear):
        dataset = pd.read_csv(F_Nonlinear)
        X_Nonlinear = dataset.iloc[:, :].values
    else:
        X_Nonlinear = None
# load features for linear fit        
    if (F_linear is not None) and os.path.isfile(F_linear):
        dataset = pd.read_csv(F_linear)
        X_linear = dataset.iloc[:, :].values
    else:
        X_linear = None
# load response variable (y)        
    if (F_Response is not None) and os.path.isfile(F_Response):
        dataset = pd.read_csv(F_Response)
        Y = dataset.iloc[:, :].values
    else:
        Y = None
# load list FeaturesNonlinear into file        
    if (F_NonlinearFeatures is not None) and os.path.isfile(F_NonlinearFeatures):
        f = open(F_NonlinearFeatures, "rb")
        FeaturesNonlinear = pickle.load(f)
        f.close()        
# load reduced features and energy from file
    if (F_FeaturesReduced is not None) and os.path.isfile(F_FeaturesReduced):
        f = open(F_FeaturesReduced, "rb")
        FeaturesReduced = pickle.load(f)
        f.close()
    else:
        FeaturesReduced = None
# load list FeaturesAll from file
    if (F_FeaturesAll is not None) and os.path.isfile(F_FeaturesAll):
        f = open(F_FeaturesAll, "rb")
        FeaturesAll = pickle.load(f)
        f.close()
    else:
        FeaturesAll = None
# load system object from file
    if (F_System is not None) and os.path.isfile(F_System):
        f = open(F_System, "rb")
        system = pickle.load(f)
        f.close()    
    else:
        system = None
# load records list from file
    if F_Records is not None and os.path.isfile(F_Records):
        f = open(F_Records, "rb")
        records = pickle.load(f)
        f.close() 
    else:
        records = None
    return X_Nonlinear, X_linear, Y, FeaturesNonlinear, FeaturesAll, FeaturesReduced, system, records
