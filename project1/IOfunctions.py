import re
import os
from project1 import structure
from project1 import library
import pandas as pd
import pickle 
import copy
import numpy as np

Separators = '=|,| |:|;|: '

# &FEATURES part of SystemDescriptor
def ReadFeatureDescription(F, keyword='LinearSingle'):
    
    def is_exists(keyword, x):
        result = None
        if (x.find(keyword) != -1):
            s = re.split(Separators, x)
            if 'True' in s:
                result = 1
            elif 'False' in s:
                result = 0  
        else:
            result = -1
        return result

# keyword = 'LinearSingle', 'LinearDouble', 'ExpSimple', 'ExpSingle', 'ExpDouble'
    first_outer_line = '{}{}{}'.format('&', keyword, 'Distances')
    last_outer_line = '{}{}{}'.format('&end', keyword, 'Distances')        
    first_inner_line = '{}{}{}'.format('&', keyword, 'DistancesDescription')
    last_inner_line = '{}{}{}'.format('&end', keyword, 'DistancesDescription')        
    include_line = '{}{}'.format(keyword, 'DistancesInclude')
    include_same_type_line = '{}{}'.format(keyword, 'IncludeSameType')
    include_only_same_power_line = '{}{}'.format(keyword, 'IncludeOnlySamePower')
    Include = None
    IncludeSameType = None
    IncludeOnlySamePower = None
    
    with open(F) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string 
    i = 0 # remove empty lines and remarks
    while i < len(lines):
        if (len(lines[i]) == 0) or (lines[i][0] == '#'):
            del(lines[i])
        else:
            i += 1
    i = 0
    if (first_outer_line in lines):   
        while (lines[i] != last_outer_line) and (i < len(lines)):  
            include = is_exists(include_line, lines[i])
            if include != -1:
                Include = bool(include)
                i += 1
                continue
            includeSameType = is_exists(include_same_type_line, lines[i])
            if includeSameType != -1:
                IncludeSameType = bool(includeSameType)
                i += 1
                continue
            includeOnlySamePower = is_exists(include_only_same_power_line, lines[i])
            if includeOnlySamePower != -1:
                IncludeOnlySamePower = bool(includeOnlySamePower)
                i += 1
                continue
            if (lines[i] == first_inner_line):
                i += 1
                Description = []
                while (lines[i] != last_inner_line):
                    s = re.split(Separators, lines[i]) # string to list
                    s = list(filter(bool, s)) # remove spaces
                    if s[2] == 'intermolecular':
                        inter = True
                    else:
                        inter = False
                    j = 3 # powers start
                    powers = []
                    while j < len(s):
                        powers.append(int(s[j]))
                        j += 1
                    Description.append([s[0], s[1], inter, powers])
                    i += 1
            else:
                i += 1
    else:
        results = {'Include': None, 'IncludeSameType': None,\
            'IncludeOnlySamePower': None, 'Description': None}
        return results
    
    results = {'Include': Include, 'IncludeSameType': IncludeSameType,\
        'IncludeOnlySamePower': IncludeOnlySamePower, 'Description': Description}
    return results

# &SYSTEM part of SystemDesctiptor
def ReadSystemDescription(F, keyword='SYSTEM'):
# returns atoms and molecules    
    first_line = '{}{}'.format('&', keyword)
    last_line = '{}{}'.format('&end', keyword)
    F = 'SystemDescriptor.'
    with open(F) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string    
    if (first_line in lines):
        i = 0 # index of line in file
        Atoms = [] # create list of atom structures from file
        while ((lines[i].find(first_line) == -1) and (i < len(lines))):
            i += 1
        i += 1 # next line in file after &keyword
        j = 0 # atoms order in the system 
        types_list = []
        idx_list = []
        molecules_idx_list = []
        Molecules = []
        k = 0 # types of atom index
        while ((lines[i].find(last_line) == -1) and (i < len(lines))):
            x = lines[i]
            if (x[0] == '#'): # skip remark
                i += 1
                continue            
            if (x.find('&Molecule') != -1):
                s = re.split(Separators, x)
                del(s[0]) # 'Molecule'
                s0 = s[0]
                for l in range(1, len(s), 1):
                    s0 = s0 + ' ' + s[l]
                MoleculeName = s0
                i += 1
                one_molecule_atoms = []
                x = lines[i]
                while not (x.find('&endMolecule') == 0):                    
                    if (x[0] == '#'): # skip remark
                        i += 1
                        continue                             
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    symbol = s[0]
                    if symbol not in types_list:
                        types_list.append(symbol)
                        idx_list.append(k)
                        k += 1
                    idx = types_list.index(symbol)    
                    molecule_index = int(s[1])
                    if molecule_index not in molecules_idx_list: # next molecule
                        molecules_idx_list.append(molecule_index)
                    atom = structure.Atom(Symbol=symbol, Index=j, AtType=idx_list[idx],\
                        MolecularIndex=molecule_index, AtTypeDigits=1, Mass=None,\
                        Radius=None, Bonds=None)
                    one_molecule_atoms.append(atom)
                    j += 1 # to next index
                    i += 1 # to next line   
                    x = lines[i]
                i += 1    
                for l in one_molecule_atoms: # store atoms from finished molecule
                    Atoms.append(l)                    
                Molecules.append(structure.Molecule(one_molecule_atoms, Name=MoleculeName))
            else:
                i += 1
    
    return Atoms, Molecules

# &MoleculeDescription part of SystemDescripror or MoleculeDescriptor
def ReadMoleculeDescription(F, keyword='MoleculeDescription'):    
# read molecule descriptor from file
    first_line = '{}{}'.format('&', keyword)
    last_line = '{}{}'.format('&end', keyword)
    with open(F) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string
    if (first_line in lines):
        i = 0 # index of line in file
        Molecules = [] # create list of atom structures from file
        while ((lines[i].find(first_line) == -1) and (i < len(lines))):
            i += 1
        i += 1 # go to next line
        j = 0
        nMolecule = 0
        atoms = []
        types_list = []
        idx_list = []
        k = 0
        while ((lines[i].find(last_line) == -1) and (i < len(lines))):
            x = lines[i]
            if (x[0] == '#'):
                i += 1
                continue     
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
            
            atom = structure.Atom(Symbol=symbol, Index=index, AtType=idx_list[idx],\
                MolecularIndex=nMolecule, AtTypeDigits=1, Mass=Mass, Radius=Radius,\
                Bonds=Bonds, x=X, y=Y, z=Z)
            atoms.append(atom)
            j += 1
            i += 1
    return Molecules
    
# reads feature set and objects that describes it
def ReadFeatures(F_Nonlinear_Train=None, F_Nonlinear_Test=None, F_linear_Train=None, F_Response_Train=None,\
        F_linear_Test=None, F_Response_Test=None, F_NonlinearFeatures=None,\
        F_FeaturesAll=None, F_FeaturesReduced=None, F_System=None, F_Records=None,\
        verbose=False):
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
    if (F_Nonlinear_Train is not None) and os.path.isfile(F_Nonlinear_Train):
        dataset = pd.read_csv(F_Nonlinear_Train)
        X_Nonlinear_Train = dataset.iloc[:, :].values
    else:
        X_Nonlinear_Train = None
    if (F_Nonlinear_Test is not None) and os.path.isfile(F_Nonlinear_Test):
        dataset = pd.read_csv(F_Nonlinear_Test)
        X_Nonlinear_Test = dataset.iloc[:, :].values
    else:
        X_Nonlinear_Test = None
# load features for linear fit        
    if (F_linear_Train is not None) and os.path.isfile(F_linear_Train):
        dataset = pd.read_csv(F_linear_Train)
        X_linear_Train = dataset.iloc[:, :].values
    else:
        X_linear_Train = None
# load response variable (y)        
    if (F_Response_Train is not None) and os.path.isfile(F_Response_Train):
        dataset = pd.read_csv(F_Response_Train)
        Y_Train = dataset.iloc[:, :].values
    else:
        Y_Train = None
# load features for linear fit        
    if (F_linear_Test is not None) and os.path.isfile(F_linear_Test):
        dataset = pd.read_csv(F_linear_Test)
        X_linear_Test = dataset.iloc[:, :].values
    else:
        X_linear_Test = None
# load response variable (y)        
    if (F_Response_Test is not None) and os.path.isfile(F_Response_Test):
        dataset = pd.read_csv(F_Response_Test)
        Y_Test = dataset.iloc[:, :].values
    else:
        Y_Test = None
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
    return {'X Nonlinear Train': X_Nonlinear_Train, 'X Nonlinear Test': X_Nonlinear_Test, 'X Linear Train': X_linear_Train,\
            'Response Train': Y_Train, 'X Linear Test': X_linear_Test,\
            'Response Test': Y_Test, 'Nonlinear Features': FeaturesNonlinear,\
            'Linear Features All': FeaturesAll, 'Linear Features Reduced': FeaturesReduced,\
            'System': system, 'Records': records}

# save as a text file F records of class RecordMolecules
def store_records(F, record_list):
    records = []
    for i in range(0, len(record_list), 1):
        for molecule in record_list[i].Molecules:
            for atom in molecule.Atoms:
                S = atom.Symbol
                x = str(atom.x)
                y = str(atom.y)
                z = str(atom.z)
                line = S + ': ' + x + '\t' + y + '\t' + z + '\n'
                records.append(line)
        line = str(record_list[i].E_True) + '\n'
        records.append(line)
        records.append('\n')
    
    f = open(F, "w")
    f.writelines(records)
    f.close()
    return

# save as .csv file average distance between centers of masses of two molecules or
# average distance from center of mass of system to center of mass of molecule (for 3+ molecules)
def store_average_distances(F, record_list):
    Size = len(record_list)
    distances = np.zeros(shape=(Size, 1), dtype=float)
    for i in range(0, Size, 1):
        if record_list[i].nMolecules == 2: # if 2 molecules in the system
            distances[i] = record_list[i].R_Average
        else:
            distances[i] = record_list[i].R_CenterOfMass_Average # 1 or more than 2 molecules            
    Table = pd.DataFrame(distances, columns=['COM average'], dtype=float)
    f = open(F, 'w')
    Table.to_csv(f, index=False)
    f.close()
    return

# Read records and return as list of RecordMolecules objects
def ReadRecordMolecules(F, MoleculePrototypes):
    # Read coordinates from file
    f = open(F, "r")
    data0 = f.readlines()
    f.close()
    data1 = []
    for i in range(0, len(data0), 1):
        data1.append(data0[i].rstrip())
    del(data0)
    # Rearrange data in structure
    i = 0 # counts lines in textdata
    j = 0 # molecule number
    k = 0 # atom number
    molecules = copy.deepcopy(MoleculePrototypes)
    Records = []
    while i < len(data1):
        s = data1[i].split() # line of text separated in list
        if len(s) == 0: # empty line
            i += 1 # next line
            j = 0 # molecule number
            k = 0 # atom number
            molecules = copy.deepcopy(MoleculePrototypes)
            continue
        elif (len(s) == 1) and library.isfloat(s[0]): # energy line
            j = 0
            k = 0
            e = float(s[0])    
            for molecule in molecules:
                molecule._refresh()
            rec = structure.RecordMolecules(molecules, E_True=e)
            Records.append(rec)
        elif (len(s) == 4): 
            molecules[j].Atoms[k].x = float(s[1])
            molecules[j].Atoms[k].y = float(s[2])
            molecules[j].Atoms[k].z = float(s[3])
            k += 1
            if k >= molecules[j].nAtoms: # go to next molecule
                k = 0
                j += 1
        i += 1
    return Records

# Read records and return as list of RecordAtoms objects
def ReadRecordAtoms(F_data, Atoms):
        # Read coordinates from file
    atoms = copy.deepcopy(Atoms)
    f = open(F_data, "r")
    data0 = f.readlines()
    f.close()
    data = []
    for i in range(0, len(data0), 1):
        data.append(data0[i].rstrip())
    del(data0)
    # Rearrange data in structure
    i = 0 # counts lines in textdata
    j = 0 # counts atom records for each energy value
    atoms_list = [] # temporary list
    records_list = []
    while i < len(data):
        s = data[i].split() # line of text separated in list
        if len(s) == 0: # empty line
            i += 1
            continue
    # record for energy value
        elif (len(s) == 1) and library.isfloat(s[0]): 
            records_list.append(structure.RecordAtoms(Atoms=atoms_list, Energy=float(s[0])))
            j = 0
            atoms_list = []
            atoms = copy.deepcopy(Atoms)
        elif (len(s) == 4): 
            atoms[j].x = float(s[1])
            atoms[j].y = float(s[2])
            atoms[j].z = float(s[3])
            atoms_list.append(atoms[j])
            j += 1
        i += 1
    return records_list

def StoreEnergy(F_Response, record_list):
    Size = len(record_list)
    if Size > 0:
        energy = np.zeros(shape=(Size, 1), dtype=float)
        for i in range(0, Size, 1):
            energy[i, 0] = record_list[i].e # energy    
        Table = pd.DataFrame(energy, columns=['response'], dtype=float)
        f = open(F_Response, 'w')
        Table.to_csv(f, index=False)
        f.close()
    return

# can store single and double reduced linear features
def StoreLinearFeatures(F, FeaturesAll, FeaturesReduced, records_list, Atoms):
    nReduced = len(FeaturesReduced)
    nDistances = FeaturesAll[0].nDistances
# determine type of feature (exponential or linear)    
#    feature_class = ''.join(re.findall("[a-zA-Z]+", FeaturesAll[0].FeType))
    Size = len(records_list)
    features_reduced_array = np.zeros(shape=(Size, nReduced), dtype=float)
# calculating and storing distances  
    for i in range(0, nReduced, 1): # feature reduced index
        for j in range(0, Size, 1): # record index
            for idx in FeaturesReduced[i].idx: # indices of all features to combine in reduced set
                if (nDistances == 1):
# linear features with only one distance 
                    atom1_index = FeaturesAll[idx].DtP1.Distance.Atom1.Index # first atom number
                    atom2_index = FeaturesAll[idx].DtP1.Distance.Atom2.Index # second atom number
                    d = np.sqrt((records_list[j].atoms[atom1_index].x - records_list[j].atoms[atom2_index].x)**2 +\
                        (records_list[j].atoms[atom1_index].y - records_list[j].atoms[atom2_index].y)**2 +\
                        (records_list[j].atoms[atom1_index].z - records_list[j].atoms[atom2_index].z)**2)            
                    r = d**FeaturesAll[idx].DtP1.Power # distance to correcponding power
                elif (nDistances == 2):
# linear features with two distances
                    atom11_index = FeaturesAll[idx].DtP1.Distance.Atom1.Index
                    atom12_index = FeaturesAll[idx].DtP1.Distance.Atom2.Index
                    atom21_index = FeaturesAll[idx].DtP2.Distance.Atom1.Index
                    atom22_index = FeaturesAll[idx].DtP2.Distance.Atom2.Index
                    d1 = np.sqrt((records_list[j].atoms[atom11_index].x - records_list[j].atoms[atom12_index].x)**2 +\
                        (records_list[j].atoms[atom11_index].y - records_list[j].atoms[atom12_index].y)**2 +\
                        (records_list[j].atoms[atom11_index].z - records_list[j].atoms[atom12_index].z)**2)            
                    r1 = d1**FeaturesAll[idx].DtP1.Power # distance to correcponding power
                    d2 = np.sqrt((records_list[j].atoms[atom21_index].x - records_list[j].atoms[atom22_index].x)**2 +\
                        (records_list[j].atoms[atom21_index].y - records_list[j].atoms[atom22_index].y)**2 +\
                        (records_list[j].atoms[atom21_index].z - records_list[j].atoms[atom22_index].z)**2)            
                    r2 = d2**FeaturesAll[idx].DtP2.Power # distance to correcponding power
                    r = r1 * r2       
                elif (nDistances == 3):
# linear features with two distances
                    atom11_index = FeaturesAll[idx].DtP1.Distance.Atom1.Index
                    atom12_index = FeaturesAll[idx].DtP1.Distance.Atom2.Index
                    atom21_index = FeaturesAll[idx].DtP2.Distance.Atom1.Index
                    atom22_index = FeaturesAll[idx].DtP2.Distance.Atom2.Index
                    atom31_index = FeaturesAll[idx].DtP3.Distance.Atom1.Index
                    atom32_index = FeaturesAll[idx].DtP3.Distance.Atom2.Index                    
                    d1 = np.sqrt((records_list[j].atoms[atom11_index].x - records_list[j].atoms[atom12_index].x)**2 +\
                        (records_list[j].atoms[atom11_index].y - records_list[j].atoms[atom12_index].y)**2 +\
                        (records_list[j].atoms[atom11_index].z - records_list[j].atoms[atom12_index].z)**2)            
                    r1 = d1**FeaturesAll[idx].DtP1.Power # distance to correcponding power
                    d2 = np.sqrt((records_list[j].atoms[atom21_index].x - records_list[j].atoms[atom22_index].x)**2 +\
                        (records_list[j].atoms[atom21_index].y - records_list[j].atoms[atom22_index].y)**2 +\
                        (records_list[j].atoms[atom21_index].z - records_list[j].atoms[atom22_index].z)**2)            
                    r2 = d2**FeaturesAll[idx].DtP2.Power # distance to correcponding power
                    d3 = np.sqrt((records_list[j].atoms[atom31_index].x - records_list[j].atoms[atom32_index].x)**2 +\
                        (records_list[j].atoms[atom31_index].y - records_list[j].atoms[atom32_index].y)**2 +\
                        (records_list[j].atoms[atom31_index].z - records_list[j].atoms[atom32_index].z)**2)            
                    r3 = d3**FeaturesAll[idx].DtP3.Power # distance to correcponding power                    
                    r = r1 * r2 * r3           
                features_reduced_array[j, i] += r # add to same feature type
# save reduced features into file
    Table = pd.DataFrame(features_reduced_array, dtype=float)
    f = open(F, 'w')
    Table.to_csv(f, index=False)
    f.close()
    return

# store features for exponential single fit
def StoreExpSingleFeatures(F_D, F_Dn, FeaturesAll, records_list, Atoms):
    nFeatures = len(FeaturesAll)
    Size = len(records_list)
    distances_array = np.zeros(shape=(Size, nFeatures), dtype=float)
    distances_powers_array = np.zeros(shape=(Size, nFeatures), dtype=float)
    D = pd.DataFrame(distances_array, dtype=float) 
    Dn = pd.DataFrame(distances_powers_array, dtype=float) 
    # calculating and storing distances  
    for i in range(0, nFeatures, 1): # feature index
        for j in range(0, Size, 1): # record index
    # features for exponential single
            atom1_index = FeaturesAll[i].DtP1.Distance.Atom1.Index # first atom number
            atom2_index = FeaturesAll[i].DtP1.Distance.Atom2.Index # second atom number
            d = np.sqrt((records_list[j].atoms[atom1_index].x - records_list[j].atoms[atom2_index].x)**2 +\
                (records_list[j].atoms[atom1_index].y - records_list[j].atoms[atom2_index].y)**2 +\
                (records_list[j].atoms[atom1_index].z - records_list[j].atoms[atom2_index].z)**2)            
            dn = d**FeaturesAll[i].DtP1.Power
            D.iloc[j, i] = d # distances
            Dn.iloc[j, i] = dn # distances raised to power
    # save exp features into file   
    f = open(F_D, 'w')
    D.to_csv(f, index=False)
    f.close()
    f = open(F_Dn, 'w')
    Dn.to_csv(f, index=False)
    f.close()
    return

# store double exponential features
def StoreExpDoubleFeatures(F_D1, F_D2, F_D1mD2n, FeaturesAll, records_list, Atoms):
    nFeatures = len(FeaturesAll)
    Size = len(records_list)
    d1_array = np.zeros(shape=(Size, nFeatures), dtype=float)
    d2_array = np.zeros(shape=(Size, nFeatures), dtype=float)
    d1d2_array = np.zeros(shape=(Size, nFeatures), dtype=float)
    D1 = pd.DataFrame(d1_array, dtype=float) 
    D2 = pd.DataFrame(d2_array, dtype=float) 
    D1mD2n = pd.DataFrame(d1d2_array, dtype=float) 
# calculating and storing distances  
    for i in range(0, nFeatures, 1): # feature index
        for j in range(0, Size, 1): # record index
# linear features with two distances
            atom11_index = FeaturesAll[i].DtP1.Distance.Atom1.Index
            atom12_index = FeaturesAll[i].DtP1.Distance.Atom2.Index
            atom21_index = FeaturesAll[i].DtP2.Distance.Atom1.Index
            atom22_index = FeaturesAll[i].DtP2.Distance.Atom2.Index
            d1 = np.sqrt((records_list[j].atoms[atom11_index].x - records_list[j].atoms[atom12_index].x)**2 +\
                (records_list[j].atoms[atom11_index].y - records_list[j].atoms[atom12_index].y)**2 +\
                (records_list[j].atoms[atom11_index].z - records_list[j].atoms[atom12_index].z)**2)            
            r1 = d1**FeaturesAll[i].DtP1.Power # distance to correcponding power
            d2 = np.sqrt((records_list[j].atoms[atom21_index].x - records_list[j].atoms[atom22_index].x)**2 +\
                (records_list[j].atoms[atom21_index].y - records_list[j].atoms[atom22_index].y)**2 +\
                (records_list[j].atoms[atom21_index].z - records_list[j].atoms[atom22_index].z)**2)            
            r2 = d2**FeaturesAll[i].DtP2.Power # distance to correcponding power               
            D1.iloc[j, i] = d1
            D2.iloc[j, i] = d2
            D1mD2n.iloc[j, i] = r1*r2
# product of distances raised to some negative powers to be multiplied by exponent)
# save reduced features into file
    f = open(F_D1, 'w')
    D1.to_csv(f, index=False)
    f.close()
    f = open(F_D2, 'w')
    D2.to_csv(f, index=False)
    f.close()    
    f = open(F_D1mD2n, 'w')
    D1mD2n.to_csv(f, index=False)
    f.close()
    return

# save object into binary file
def SaveObject(F, Object):    
    f = open(F, "wb")
    pickle.dump(Object, f)
    f.close()
    return

def LoadObject(F):
    if os.path.isfile(F):
        f = open(F, "rb")
        Object = pickle.load(f)
        f.close()
        return Object
    else:
        return None
    
def StoreFeatures(writer, sheet_name, Features, Reduced=False):
    if Features is None:
        return
    workbook = writer.book
    format_normal = workbook.add_format({'align':'center','valign':'vcenter'})
    format_red = workbook.add_format({'bold': True,'align':'center','valign':'vcenter','font_color': 'red'})    
    if Features[0].nDistances <= 1: # 0 or 1
        Features_table = pd.DataFrame(np.zeros(shape = (len(Features), 18)).astype(int),\
            columns=['FeType','Category Molecular','Category Atomic','Power','DtpType','IsIntermolecular',\
            'DiType','Atom1 Symbol','Atom1 Index','Atom1 AtType','Atom1 Molecular Index',\
            'Atom2 Symbol','Atom2 Index','Atom2 AtType','Atom2 Molecular Index',\
            'nDistances','nConstants','Idx'], dtype=str)
        for i in range(0, len(Features), 1):
            Features_table.loc[i]['FeType'] = Features[i].FeType
            Features_table.loc[i]['Category Molecular'] = int(Features[i].FeType[-2])
            Features_table.loc[i]['Category Atomic'] = int(Features[i].FeType[-1])
            Features_table.loc[i]['Power'] = Features[i].DtP1.Power
            Features_table.loc[i]['DtpType'] = int(Features[i].DtP1.DtpType)
            Features_table.loc[i]['IsIntermolecular'] = Features[i].DtP1.Distance.isIntermolecular
            Features_table.loc[i]['DiType'] = Features[i].DtP1.Distance.DiType
            Features_table.loc[i]['Atom1 Symbol'] = Features[i].DtP1.Distance.Atom1.Symbol
            Features_table.loc[i]['Atom1 Index'] = Features[i].DtP1.Distance.Atom1.Index
            Features_table.loc[i]['Atom1 AtType'] = Features[i].DtP1.Distance.Atom1.AtType
            Features_table.loc[i]['Atom1 Molecular Index'] = Features[i].DtP1.Distance.Atom1.MolecularIndex
            Features_table.loc[i]['Atom2 Symbol'] = Features[i].DtP1.Distance.Atom2.Symbol
            Features_table.loc[i]['Atom2 Index'] = Features[i].DtP1.Distance.Atom2.Index
            Features_table.loc[i]['Atom2 AtType'] = Features[i].DtP1.Distance.Atom2.AtType
            Features_table.loc[i]['Atom2 Molecular Index'] = Features[i].DtP1.Distance.Atom2.MolecularIndex
            Features_table.loc[i]['nDistances'] = Features[i].nDistances
            Features_table.loc[i]['nConstants'] = Features[i].nConstants
            if Reduced:
                Features_table.loc[i]['Idx'] = Features[i].idx
        if not Reduced:
            del(Features_table['Idx'])
        Features_table.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        worksheet.set_column('A:R', 16, format_normal)
        worksheet.set_column('I:I', 16, format_red)
        worksheet.set_column('M:M', 16, format_red)
    elif Features[0].nDistances == 2:           
        Features_table = pd.DataFrame(np.zeros(shape = (len(Features), 30)).astype(int),\
            columns=['FeType','Category Molecular','Category Atomic','Power1','DtpType1','IsIntermolecular1',\
            'DiType1','Atom11 Symbol','Atom11 Index','Atom11 AtType','Atom11 Molecular Index',\
            'Atom12 Symbol','Atom12 Index','Atom12 AtType','Atom12 Molecular Index',\
            'Power2','DtpType2','IsIntermolecular2','DiType2','Atom21 Symbol','Atom21 Index',\
            'Atom21 AtType','Atom21 Molecular Index','Atom22 Symbol','Atom22 Index',\
            'Atom22 AtType','Atom22 Molecular Index','nDistances','nConstants','Idx'], dtype=str)
        for i in range(0, len(Features), 1):
            Features_table.loc[i]['FeType'] = Features[i].FeType
            Features_table.loc[i]['Category Molecular'] = int(Features[i].FeType[-2])
            Features_table.loc[i]['Category Atomic'] = int(Features[i].FeType[-1])
            Features_table.loc[i]['Power1'] = Features[i].DtP1.Power
            Features_table.loc[i]['DtpType1'] = int(Features[i].DtP1.DtpType)
            Features_table.loc[i]['IsIntermolecular1'] = Features[i].DtP1.Distance.isIntermolecular
            Features_table.loc[i]['DiType1'] = Features[i].DtP1.Distance.DiType
            Features_table.loc[i]['Atom11 Symbol'] = Features[i].DtP1.Distance.Atom1.Symbol
            Features_table.loc[i]['Atom11 Index'] = Features[i].DtP1.Distance.Atom1.Index
            Features_table.loc[i]['Atom11 AtType'] = Features[i].DtP1.Distance.Atom1.AtType
            Features_table.loc[i]['Atom11 Molecular Index'] = Features[i].DtP1.Distance.Atom1.MolecularIndex
            Features_table.loc[i]['Atom12 Symbol'] = Features[i].DtP1.Distance.Atom2.Symbol
            Features_table.loc[i]['Atom12 Index'] = Features[i].DtP1.Distance.Atom2.Index
            Features_table.loc[i]['Atom12 AtType'] = Features[i].DtP1.Distance.Atom2.AtType
            Features_table.loc[i]['Atom12 Molecular Index'] = Features[i].DtP1.Distance.Atom2.MolecularIndex
            Features_table.loc[i]['Power2'] = Features[i].DtP2.Power
            Features_table.loc[i]['DtpType2'] = int(Features[i].DtP2.DtpType)
            Features_table.loc[i]['IsIntermolecular2'] = Features[i].DtP2.Distance.isIntermolecular
            Features_table.loc[i]['DiType2'] = Features[i].DtP2.Distance.DiType
            Features_table.loc[i]['Atom21 Symbol'] = Features[i].DtP2.Distance.Atom1.Symbol
            Features_table.loc[i]['Atom21 Index'] = Features[i].DtP2.Distance.Atom1.Index
            Features_table.loc[i]['Atom21 AtType'] = Features[i].DtP2.Distance.Atom1.AtType
            Features_table.loc[i]['Atom21 Molecular Index'] = Features[i].DtP2.Distance.Atom1.MolecularIndex
            Features_table.loc[i]['Atom22 Symbol'] = Features[i].DtP2.Distance.Atom2.Symbol
            Features_table.loc[i]['Atom22 Index'] = Features[i].DtP2.Distance.Atom2.Index
            Features_table.loc[i]['Atom22 AtType'] = Features[i].DtP2.Distance.Atom2.AtType
            Features_table.loc[i]['Atom22 Molecular Index'] = Features[i].DtP2.Distance.Atom2.MolecularIndex
            Features_table.loc[i]['nDistances'] = Features[i].nDistances
            Features_table.loc[i]['nConstants'] = Features[i].nConstants            
            if Reduced:
                Features_table.loc[i]['Idx'] = Features[i].idx
        if not Reduced:
            del(Features_table['Idx'])
        Features_table.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        worksheet.set_column('A:AF', 16, format_normal)
        worksheet.set_column('I:I', 16, format_red)
        worksheet.set_column('M:M', 16, format_red)    
        worksheet.set_column('U:U', 16, format_red)
        worksheet.set_column('Y:Y', 16, format_red)
    elif Features[0].nDistances == 3:           
        Features_table = pd.DataFrame(np.zeros(shape = (len(Features), 42)).astype(int),\
            columns=['FeType','Category Molecular','Category Atomic','Power1','DtpType1','IsIntermolecular1',\
            'DiType1','Atom11 Symbol','Atom11 Index','Atom11 AtType','Atom11 Molecular Index',\
            'Atom12 Symbol','Atom12 Index','Atom12 AtType','Atom12 Molecular Index',\
            'Power2','DtpType2','IsIntermolecular2','DiType2','Atom21 Symbol','Atom21 Index',\
            'Atom21 AtType','Atom21 Molecular Index','Atom22 Symbol','Atom22 Index',\
            'Atom22 AtType','Atom22 Molecular Index','Power3','DtpType3','IsIntermolecular3','DiType3','Atom31 Symbol','Atom31 Index',\
            'Atom31 AtType','Atom31 Molecular Index','Atom32 Symbol','Atom32 Index',\
            'Atom32 AtType','Atom32 Molecular Index','nDistances','nConstants','Idx'], dtype=str)
        for i in range(0, len(Features), 1):
            Features_table.loc[i]['FeType'] = Features[i].FeType
            Features_table.loc[i]['Category Molecular'] = int(Features[i].FeType[-2])
            Features_table.loc[i]['Category Atomic'] = int(Features[i].FeType[-1])
            Features_table.loc[i]['Power1'] = Features[i].DtP1.Power
            Features_table.loc[i]['DtpType1'] = int(Features[i].DtP1.DtpType)
            Features_table.loc[i]['IsIntermolecular1'] = Features[i].DtP1.Distance.isIntermolecular
            Features_table.loc[i]['DiType1'] = Features[i].DtP1.Distance.DiType
            Features_table.loc[i]['Atom11 Symbol'] = Features[i].DtP1.Distance.Atom1.Symbol
            Features_table.loc[i]['Atom11 Index'] = Features[i].DtP1.Distance.Atom1.Index
            Features_table.loc[i]['Atom11 AtType'] = Features[i].DtP1.Distance.Atom1.AtType
            Features_table.loc[i]['Atom11 Molecular Index'] = Features[i].DtP1.Distance.Atom1.MolecularIndex
            Features_table.loc[i]['Atom12 Symbol'] = Features[i].DtP1.Distance.Atom2.Symbol
            Features_table.loc[i]['Atom12 Index'] = Features[i].DtP1.Distance.Atom2.Index
            Features_table.loc[i]['Atom12 AtType'] = Features[i].DtP1.Distance.Atom2.AtType
            Features_table.loc[i]['Atom12 Molecular Index'] = Features[i].DtP1.Distance.Atom2.MolecularIndex
            Features_table.loc[i]['Power2'] = Features[i].DtP2.Power
            Features_table.loc[i]['DtpType2'] = int(Features[i].DtP2.DtpType)
            Features_table.loc[i]['IsIntermolecular2'] = Features[i].DtP2.Distance.isIntermolecular
            Features_table.loc[i]['DiType2'] = Features[i].DtP2.Distance.DiType
            Features_table.loc[i]['Atom21 Symbol'] = Features[i].DtP2.Distance.Atom1.Symbol
            Features_table.loc[i]['Atom21 Index'] = Features[i].DtP2.Distance.Atom1.Index
            Features_table.loc[i]['Atom21 AtType'] = Features[i].DtP2.Distance.Atom1.AtType
            Features_table.loc[i]['Atom21 Molecular Index'] = Features[i].DtP2.Distance.Atom1.MolecularIndex
            Features_table.loc[i]['Atom22 Symbol'] = Features[i].DtP2.Distance.Atom2.Symbol
            Features_table.loc[i]['Atom22 Index'] = Features[i].DtP2.Distance.Atom2.Index
            Features_table.loc[i]['Atom22 AtType'] = Features[i].DtP2.Distance.Atom2.AtType
            Features_table.loc[i]['Atom22 Molecular Index'] = Features[i].DtP2.Distance.Atom2.MolecularIndex
            Features_table.loc[i]['Power3'] = Features[i].DtP3.Power
            Features_table.loc[i]['DtpType3'] = int(Features[i].DtP3.DtpType)
            Features_table.loc[i]['IsIntermolecular3'] = Features[i].DtP3.Distance.isIntermolecular
            Features_table.loc[i]['DiType3'] = Features[i].DtP3.Distance.DiType
            Features_table.loc[i]['Atom31 Symbol'] = Features[i].DtP3.Distance.Atom1.Symbol
            Features_table.loc[i]['Atom31 Index'] = Features[i].DtP3.Distance.Atom1.Index
            Features_table.loc[i]['Atom31 AtType'] = Features[i].DtP3.Distance.Atom1.AtType
            Features_table.loc[i]['Atom31 Molecular Index'] = Features[i].DtP3.Distance.Atom1.MolecularIndex
            Features_table.loc[i]['Atom32 Symbol'] = Features[i].DtP3.Distance.Atom2.Symbol
            Features_table.loc[i]['Atom32 Index'] = Features[i].DtP3.Distance.Atom2.Index
            Features_table.loc[i]['Atom32 AtType'] = Features[i].DtP3.Distance.Atom2.AtType
            Features_table.loc[i]['Atom32 Molecular Index'] = Features[i].DtP3.Distance.Atom2.MolecularIndex
            Features_table.loc[i]['nDistances'] = Features[i].nDistances
            Features_table.loc[i]['nConstants'] = Features[i].nConstants            
            if Reduced:
                Features_table.loc[i]['Idx'] = Features[i].idx
        if not Reduced:
            del(Features_table['Idx'])
        Features_table.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        worksheet.set_column('A:AQ', 16, format_normal)
        worksheet.set_column('I:I', 16, format_red)
        worksheet.set_column('M:M', 16, format_red)    
        worksheet.set_column('U:U', 16, format_red)
        worksheet.set_column('Y:Y', 16, format_red)
        worksheet.set_column('AG:AG', 16, format_red)
        worksheet.set_column('AK:AK', 16, format_red)        
    return
    
def StoreStructure(F_xls, Structure):
    writer = pd.ExcelWriter(F_xls, engine='xlsxwriter')  
    workbook = writer.book
    format_normal = workbook.add_format({'align':'center','valign':'vcenter'})
    format_red = workbook.add_format({'bold': True,'align':'center','valign':'vcenter','font_color': 'red'})
    Atoms = Structure['System'].Atoms
    Distances = Structure['System'].Distances    
    Atoms_table = pd.DataFrame(np.zeros(shape = (Structure['System'].nAtoms, 4)).astype(int),\
        columns=['Symbol','Index','Type','Molecular Index'], dtype=str)
    for i in range(0, len(Atoms), 1):
        Atoms_table.loc[i]['Symbol'] = Atoms[i].Symbol
        Atoms_table.loc[i]['Index'] = Atoms[i].Index
        Atoms_table.loc[i]['Type'] = Atoms[i].AtType
        Atoms_table.loc[i]['Molecular Index'] = Atoms[i].MolecularIndex
    Atoms_table.to_excel(writer, sheet_name='Atoms')        
    worksheet_Atoms = writer.sheets['Atoms']
    worksheet_Atoms.set_column('A:E', 16, format_normal)
    worksheet_Atoms.set_column('B:B', 16, format_red)    
    Distances_table = pd.DataFrame(np.zeros(shape = (Structure['System'].nDistances, 10)).astype(int),\
        columns=['DiType','IsIntermolecular','Atom1 Symbol','Atom1 Index',\
        'Atom1 AtType','Atom1 Molecular Index','Atom2 Symbol','Atom2 Index',\
        'Atom2 AtType','Atom2 Molecular Index'], dtype=str)
    for i in range(0, len(Distances), 1):
        Distances_table.loc[i]['IsIntermolecular'] = Distances[i].isIntermolecular
        Distances_table.loc[i]['DiType'] = Distances[i].DiType
        Distances_table.loc[i]['Atom1 Symbol'] = Distances[i].Atom1.Symbol
        Distances_table.loc[i]['Atom1 Index'] = Distances[i].Atom1.Index
        Distances_table.loc[i]['Atom1 AtType'] = Distances[i].Atom1.AtType
        Distances_table.loc[i]['Atom1 Molecular Index'] = Distances[i].Atom1.MolecularIndex
        Distances_table.loc[i]['Atom2 Symbol'] = Distances[i].Atom2.Symbol
        Distances_table.loc[i]['Atom2 Index'] = Distances[i].Atom2.Index
        Distances_table.loc[i]['Atom2 AtType'] = Distances[i].Atom2.AtType
        Distances_table.loc[i]['Atom2 Molecular Index'] = Distances[i].Atom2.MolecularIndex
    Distances_table.to_excel(writer, sheet_name='Distances')
    worksheet_Distances = writer.sheets['Distances']
    worksheet_Distances.set_column('A:K', 16, format_normal)
    worksheet_Distances.set_column('D:D', 16, format_red)
    worksheet_Distances.set_column('H:H', 16, format_red)
    StoreFeatures(writer, 'LinearSingleAll', Structure['FeaturesLinearSingleAll'], Reduced=False)
    StoreFeatures(writer, 'LinearSingleReduced', Structure['FeaturesLinearSingleReduced'], Reduced=True)
    StoreFeatures(writer, 'LinearDoubleAll', Structure['FeaturesLinearDoubleAll'], Reduced=False)
    StoreFeatures(writer, 'LinearDoubleReduced', Structure['FeaturesLinearDoubleReduced'], Reduced=True)
    StoreFeatures(writer, 'LinearTripleAll', Structure['FeaturesLinearTripleAll'], Reduced=False)
    StoreFeatures(writer, 'LinearTripleReduced', Structure['FeaturesLinearTripleReduced'], Reduced=True)
    StoreFeatures(writer, 'ExpSingle', Structure['FeaturesExpSingleAll'], Reduced=False)
    StoreFeatures(writer, 'ExpDouble', Structure['FeaturesExpDoubleAll'], Reduced=False)
    StoreFeatures(writer, 'Gaussian', Structure['FeaturesGaussianSingleAll'], Reduced=False)
    writer.save()     
    return

def ReadCSV(F):
    if os.path.isfile(F):
        dataset = pd.read_csv(F)
        Data = dataset.iloc[:, :].values # numpy array
        if type(Data.shape) is tuple:
            if Data.shape[1] == 1 or Data.shape[0] == 1: 
                Data = Data.reshape(-1) # reshape to 1-D if possible
        return Data
    else:
        return None
    

    
    
    