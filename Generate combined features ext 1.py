# Makes combined features from coordinate file
# Requires SystemDescriptor which must accord with coordinates
# one and two distances in one feature for any number of any kind of molecules

import sys
import os
Current_dir = os.getcwd()
sys.path.append(Current_dir) # requires to import structure.class1
import numpy as np
import pandas as pd
import pickle 
from structure import class1

def StoreFeatures(F_out_features, first, last, FeaturesAll, record_list):
# Storing energy
    energy = np.zeros(shape=(last - first, 1), dtype=float)
    for i in range(first, last, 1):
        energy[i-first, 0] = record_list[i].e # energy
# calculating and storing distances    
    features_array = np.zeros(shape=(last-first, len(FeaturesAll)), dtype=float) 
    for j in range(0, len(FeaturesAll), 1):
        for i in range(first, last, 1):
            if FeaturesAll[j].nDistances == 1:
                atom1_index = FeaturesAll[j].distances[0].Atom1.Index # first atom number
                atom2_index = FeaturesAll[j].distances[0].Atom2.Index # second atom number
                d = np.sqrt((record_list[i].atoms[atom1_index].x - record_list[i].atoms[atom2_index].x)**2 +\
                            (record_list[i].atoms[atom1_index].y - record_list[i].atoms[atom2_index].y)**2 +\
                            (record_list[i].atoms[atom1_index].z - record_list[i].atoms[atom2_index].z)**2)            
                r = d**FeaturesAll[j].powers[0] # distance to correcponding power
            if FeaturesAll[j].nDistances == 2:
                atom11_index = FeaturesAll[j].distances[0].Atom1.Index
                atom12_index = FeaturesAll[j].distances[0].Atom2.Index
                atom21_index = FeaturesAll[j].distances[1].Atom1.Index
                atom22_index = FeaturesAll[j].distances[1].Atom2.Index
                d1 = np.sqrt((record_list[i].atoms[atom11_index].x - record_list[i].atoms[atom12_index].x)**2 +\
                             (record_list[i].atoms[atom11_index].y - record_list[i].atoms[atom12_index].y)**2 +\
                             (record_list[i].atoms[atom11_index].z - record_list[i].atoms[atom12_index].z)**2)            
                r1 = d1**FeaturesAll[j].powers[0] # distance to correcponding power
                d2 = np.sqrt((record_list[i].atoms[atom21_index].x - record_list[i].atoms[atom22_index].x)**2 +\
                             (record_list[i].atoms[atom21_index].y - record_list[i].atoms[atom22_index].y)**2 +\
                             (record_list[i].atoms[atom21_index].z - record_list[i].atoms[atom22_index].z)**2)            
                r2 = d2**FeaturesAll[j].powers[1] # distance to correcponding power
                r = r1 * r2        
            if FeaturesAll[j].nDistances == 3:
                atom11_index = FeaturesAll[j].distances[0].Atom1.Index
                atom12_index = FeaturesAll[j].distances[0].Atom2.Index
                atom21_index = FeaturesAll[j].distances[1].Atom1.Index
                atom22_index = FeaturesAll[j].distances[1].Atom2.Index
                atom31_index = FeaturesAll[j].distances[2].Atom1.Index
                atom32_index = FeaturesAll[j].distances[2].Atom2.Index
                d1 = np.sqrt((record_list[i].atoms[atom11_index].x - record_list[i].atoms[atom12_index].x)**2 +\
                             (record_list[i].atoms[atom11_index].y - record_list[i].atoms[atom12_index].y)**2 +\
                             (record_list[i].atoms[atom11_index].z - record_list[i].atoms[atom12_index].z)**2)            
                r1 = d1**FeaturesAll[j].powers[0] # distance to correcponding power
                d2 = np.sqrt((record_list[i].atoms[atom21_index].x - record_list[i].atoms[atom22_index].x)**2 +\
                             (record_list[i].atoms[atom21_index].y - record_list[i].atoms[atom22_index].y)**2 +\
                             (record_list[i].atoms[atom21_index].z - record_list[i].atoms[atom22_index].z)**2)            
                r2 = d2**FeaturesAll[j].powers[1] # distance to correcponding power
                d3 = np.sqrt((record_list[i].atoms[atom31_index].x - record_list[i].atoms[atom32_index].x)**2 +\
                             (record_list[i].atoms[atom31_index].y - record_list[i].atoms[atom32_index].y)**2 +\
                             (record_list[i].atoms[atom31_index].z - record_list[i].atoms[atom32_index].z)**2)            
                r3 = d3**FeaturesAll[j].powers[2] # distance to correcponding power
                r = r1 * r2 * r3       
            features_array[i-first, j] = r # store to array
# making array of reduced features
    features_array_reduced = np.zeros(shape=(last-first, NofFeaturesReduced), dtype=float)
    for k in range(0, NofFeaturesReduced, 1):
        for j in range(0, NofFeatures, 1):
            if FeaturesAll[j].FeType == k:
                features_array_reduced[:, k] += features_array[:, j]
# save reduced features and energy into file
    Table = pd.DataFrame(features_array_reduced, dtype=float)
    Table['energy'] = energy
    f = open(F_out_features, 'a')
    if first == 0:
        Table.to_csv(f, index=False)
    else:
        Table.to_csv(f, index=False, header=False)
    f.close()

max_number_of_distances_in_feature = 1 # if not defined in SystemDescriptor
F = 'SystemDescriptor.' # file with info about system structure
#F_data = 'datafile short.x'
#F_data = 'datafile1 from github gaussian process.x' # file with coordinates
F_data = 'datafile3 2 water molecules.x'
#F_data = 'datafile4 3 water molecules small.x'
#F_data = 'datafile5 3 water molecules big.x'
#F_data = 'datafile2.x'
F_out_features = 'Features and energy two distances reduced.csv' # output csv file with combined features and energy
F_out_structure_FeaturesReduced = 'FeaturesReduced.dat' # output data structure which contains combined features
F_out_structure_FeaturesAll = 'FeaturesAll.dat' # output data structure which contains all features
try:
    os.remove(F_out_features) # erase old file if exists
except:
    pass

# read descriptor from file

with open(F) as f:
    lines = f.readlines()
f.close()
lines = [x.strip() for x in lines] # x is string
# find '&Features' part
PowersDefault = []
DistanceDescriptionFirst = 0 #if they have the same value use only default
DistanceDescriptionLast = 0
if ('&FEATURES' in lines):
    for i in range(0, len(lines), 1):
        x = lines[i]
        if (x.find('Distances') != -1):
            max_number_of_distances_str = x.split(',', -1)
            max_number_of_distances_in_feature = int(max_number_of_distances_str[1])
        if (x.find('&DistanceDescription') != -1):
            DistanceDescriptionFirst = i + 1 # index of first record of distance description
        if (x.find('&endDistanceDescription') != -1):
            DistanceDescriptionLast = i # index +1 of last record of distance description
    for i in range(0, len(lines), 1):
        x = lines[i]
        if (x.find('Powers') != -1):
            PowersStr = x.split(',', -1)
            for j in range(1, len(PowersStr), 1):
                PowersDefault.append(int(PowersStr[j]))
if len(PowersDefault) == 0:
    PowersDefault.append(-1) # Defalt value
    
# read specific powers for distances if they are
DList = []
if DistanceDescriptionFirst != DistanceDescriptionLast:
    for i in range(DistanceDescriptionFirst, DistanceDescriptionLast, 1):
        Str = lines[i].split(',', -1)
        if Str[2] == 'intermolecular':
            Str[2] = 1
        if Str[2] == 'intramolecular':
            Str[2] = 0
        for k in range(3, len(Str), 1):
            Str[k] = int(Str[k])
        DList.append(Str)
        
Atoms = [] # create list of atom structures from file
j = 0 # order in the system 
if ('&SYSTEM' in lines):
    i = 0
    while ((lines[i].find('&SYSTEM') == -1) & (i < len(lines))):
        i += 1
    i += 1
    while ((lines[i].find('&END') == -1) & (i < len(lines))):
        if (lines[i][0] == '#'):
            i += 1
            continue
        else:
            Str = lines[i].split(',', -1)
            Atoms.append(class1.Atom(Str[0], j, int(Str[1]), int(Str[2])))
            j += 1
            i += 1
# determine number of atom types from list
nAtTypes = 0        
for i in range(0, len(Atoms), 1):
    if (Atoms[i].Type > nAtTypes):
        nAtTypes = Atoms[i].Type
nAtTypes += 1        

# replace atom symbols with numbers
for i in range(0, len(DList), 1):
    for j in range(0, len(Atoms), 1):
        if DList[i][0] == Atoms[j].Symbol:
            DList[i][0] = Atoms[j].Type 
        if DList[i][1] == Atoms[j].Symbol:
            DList[i][1] = Atoms[j].Type    
     
# constructor for distances works in two steps:
# 1. 4th argument to pass diring initialization of the new record is number of atom types
# 2. Constructor will return as 4th argument calculated DiType

nAtoms = len(Atoms)
Distances = [] # create list of distances from list of atoms
for i in range(0, nAtoms, 1):
    for j in range(i+1, nAtoms, 1):
        Distances.append(class1.Distance(Atoms[i], Atoms[j], None, nAtTypes, []))

nDiTypes = 1     
DiTypesList = [] # store unique distance types in the list
DiTypesList.append(Distances[0].DiType)
for i in range(1, len(Distances), 1):
    if (Distances[i].DiType in DiTypesList):
        continue
    else:
        DiTypesList.append(Distances[i].DiType)
        nDiTypes += 1
    
Sys = class1.System(Atoms, nAtoms, nAtTypes, Distances, len(Distances), nDiTypes)
empty_atom = class1.Atom('', -1, -1, -1)
empty_distance = class1.Distance(empty_atom, empty_atom, 0, 0, [])

for i in range(0, len(DList), 1):
    d = nAtTypes*max(DList[i][0], DList[i][1]) + min(DList[i][0], DList[i][1])+\
        nAtTypes**2 * DList[i][2]
    DList[i].insert(3, d)   
    
def CheckDType(value, column, List):
    for i in range(0, len(List), 1):
        if List[i][column] == value:
            return i # return number of row if found
    return -1

# Replace empry spaces with corresponding list of powers
for i in range(0, len(Distances), 1):
    Row = CheckDType(Distances[i].DiType, 3, DList)
    if Row != -1:
        PowersToCopy = DList[Row][4:len(DList[Row])]
        Distances[i] = Distances[i]._replace(Powers=PowersToCopy)
    else:
        Distances[i] = Distances[i]._replace(Powers=PowersDefault)


# make list of features with one distance
# total number = MaxPower * nDistances
FeaturesAll = [] # list of all features

for i in range(0, Sys.nDistances, 1):
    for j in Distances[i].Powers:
        FeaturesAll.append(class1.InvPowDistancesFeature(1, (Distances[i], Distances[i]), (j, j), -1))

# continue adding features with two distances if defined
# total number = MaxPower**2 * nDistances**2
if max_number_of_distances_in_feature >= 2:
    for i1 in range(0, Sys.nDistances, 1):
        for j1 in Distances[i1].Powers:
            for i2 in range(0, Sys.nDistances, 1):
                for j2 in Distances[i2].Powers:
                    if i1 == i2:
                        continue
                    FeaturesAll.append(class1.InvPowDistancesFeature(2, (Distances[i1], Distances[i2]), (j1, j2), -1))

# continue adding features with three distances if defined
if max_number_of_distances_in_feature == 3:
    for i1 in range(0, Sys.nDistances, 1):
        for j1 in Distances[i1].Powers:
            for i2 in range(0, Sys.nDistances, 1):
                for j2 in Distances[i2].Powers:
                    for i3 in range(0, Sys.nDistances, 1):
                        for j3 in Distances[i3].Powers:
                            if (i1 == i2) or (i2 == i3) or (i3 == i1):
                                continue
                            FeaturesAll.append(class1.InvPowDistancesFeature(3, (Distances[i1], Distances[i2], Distances[i3]), (j1, j2, j3), -1))

NofFeatures = len(FeaturesAll) # Total number of features
currentFeatureType = 0   

# Make list of reduced features
FeaturesReduced = []
# number of combined or reduced features = nDiTypes * MaxPower * nDiTypes * MaxPower + nDiTypes * MaxPower
for i in range(0, NofFeatures, 1):
    if FeaturesAll[i].FeType == -1:
        FeaturesAll[i] = FeaturesAll[i]._replace(FeType=currentFeatureType)
        FeaturesReduced.append(FeaturesAll[i])
        for j in range(i+1, NofFeatures, 1):
            if class1.AreTwoFeaturesEquivalent(FeaturesAll[i], FeaturesAll[j]):
                FeaturesAll[j] = FeaturesAll[j]._replace(FeType=currentFeatureType)
        currentFeatureType += 1

# save list FeaturesAll into file
f = open(F_out_structure_FeaturesAll, "wb")
pickle.dump(FeaturesAll, f)
f.close()

# save list FeaturesReduced into file
f = open(F_out_structure_FeaturesReduced, "wb")
pickle.dump(FeaturesReduced, f)
f.close()

NofFeaturesReduced = len(FeaturesReduced)
# Read coordinates from file
f = open(F_data, "r")
data0 = f.readlines()
f.close()
data1 = []
for i in range(0, len(data0), 1):
    data1.append(data0[i].rstrip())
    
# Rearrange data in structure
i = 0 # counts lines in textdata
j = 0 # counts atom records for each energy value
atoms_list = [] # temporary list
record_list = []
while i < len(data1):
    s = data1[i].split() # line of text separated in list
    if len(s) == 0: # empty line
        j = 0
        atoms_list = []
# record for energy value
    elif (len(s) == 1) and class1.isfloat(s[0]): 
        e = float(s[0])
        rec = class1.record(e, atoms_list)
        record_list.append(rec)
    elif (len(s) == 4): 
        atom_symbol = s[0]
        x = float(s[1])
        y = float(s[2])
        z = float(s[3])
        atoms_list.append(class1.AtomCoordinates(Atoms[j], x, y, z))
        j += 1
    i += 1

Size = len(record_list) # N of observations

# How to access to structured data:   
    
# record_list[Number of observation].atoms[Number of atom].Atom.Symbol
# record_list[Number of observation].atoms[Number of atom].Atom.Index
# record_list[Number of observation].atoms[Number of atom].Atom.Type
# record_list[Number of observation].atoms[Number of atom].Atom.MolecularIndex
# record_list[Number of observation].atoms[Number of atom].x
# record_list[Number of observation].atoms[Number of atom].y
# record_list[Number of observation].atoms[Number of atom].z
# record_list[Number of observation].e
             
# FeaturesAll[Number of feature].distances[0].Atom1.Symbol
# FeaturesAll[Number of feature].distances[0].Atom1.Index
# FeaturesAll[Number of feature].distances[0].Atom1.Type
# FeaturesAll[Number of feature].distances[0].Atom1.MolecularIndex
# FeaturesAll[Number of feature].distances[0].Atom2.Symbol
# FeaturesAll[Number of feature].distances[0].Atom2.Index
# FeaturesAll[Number of feature].distances[0].Atom2.Type
# FeaturesAll[Number of feature].distances[0].Atom2.MolecularIndex
# FeaturesAll[100].powers
# FeaturesAll[100].FeType
# FeaturesAll[100].distances[0].isIntermolecular

# split array if too big
# create endpoints for array size_list[1 - inf][0 - 1]
i = 0 # Number of observation
j = 0 # Number of feature
size_list = []
first = 0
last = len(record_list)
k = 0
BufSize = 20000
if len(record_list) > BufSize:
    while k * BufSize < len(record_list):
        first = k * BufSize
        k += 1
        last = k * BufSize
        if last > (len(record_list)):
            last = len(record_list)
        size_list.append((first, last))
else:
    size_list.append((first, last))

# Generate features from distances and store them into file
i = 0
while i < len(size_list):
    print(i)
    StoreFeatures(F_out_features, size_list[i][0], size_list[i][1], FeaturesAll, record_list)
    i += 1
        
print("DONE")