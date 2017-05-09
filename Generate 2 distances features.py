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

max_number_of_distances_in_feature = 3 # how many distances can one feature contain 1 or 2
F = 'SystemDescriptor' # file with info about system structure
#F_data = 'datafile short.x'
F_data = 'datafile1 from github gaussian process.x' # file with coordinates
F_out_features = 'Features and energy two distances reduced.csv' # output csv file with combined features and energy
F_out_structure_FeaturesReduced = 'FeaturesReduced.dat' # output data structure which contains combined features
F_out_structure_FeaturesAll = 'FeaturesAll.dat' # output data structure which contains all features

# read descriptor from file

with open(F) as f:
    lines = f.readlines()
f.close()
lines = [x.strip() for x in lines] # x is string
# find '&Features' part
if ('&FEATURES' in lines):
    for i in range(0, len(lines), 1):
        x = lines[i]
        if (x.find('MaxPower') != -1):
            MaxPowerStr = x.split(',', -1)
MaxPower = int(MaxPowerStr[1])            
    
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
# determine n of atom types from list
nAtTypes = 0        
for i in range(0, len(Atoms), 1):
    if (Atoms[i].Type > nAtTypes):
        nAtTypes = Atoms[i].Type
nAtTypes += 1        

# constructor for distances works in two steps:
# 1. 4th argument to pass diring initialization of the new record is number of atom types
# 2. Constructor will return as 4th argument calculated DiType

nAtoms = len(Atoms)
Distances = [] # create list of distances from list of atoms
for i in range(0, nAtoms, 1):
    for j in range(i+1, nAtoms, 1):
        Distances.append(class1.Distance(Atoms[i], Atoms[j], None, nAtTypes))

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
empty_distance = class1.Distance(empty_atom, empty_atom, 0, 0)

# make list of features with one distance
# total number = MaxPower * nDistances
FeaturesAll = [] # list of all features
for i in range(1, MaxPower+1, 1):
    for j in range(0, Sys.nDistances, 1):
        FeaturesAll.append(class1.InvPowDistancesFeature(1, (Distances[j], Distances[j]), (-i, -i), -1))

# continue adding features with two distances if defined
# total number = MaxPower**2 * nDistances**2
if max_number_of_distances_in_feature >= 2:
    for i in range(1, MaxPower+1, 1):
        for j in range(0, Sys.nDistances, 1):
            for k in range(1, MaxPower+1, 1):
                for l in range(0, Sys.nDistances, 1):
                    FeaturesAll.append(class1.InvPowDistancesFeature(2, (Distances[j], Distances[l]), (-i, -k), -1))

# continue adding features with three distances if defined
if max_number_of_distances_in_feature == 3:
    for i1 in range(1, MaxPower+1, 1):
        for j1 in range(0, Sys.nDistances, 1):
            for i2 in range(1, MaxPower+1, 1):
                for j2 in range(0, Sys.nDistances, 1):
                    for i3 in range(1, MaxPower+1, 1):
                        for j3 in range(0, Sys.nDistances, 1):
                            FeaturesAll.append(class1.InvPowDistancesFeature(3, (Distances[j1], Distances[j2], Distances[j3]), (-i1, -i2, -i3), -1))

NofFeatures = len(FeaturesAll) # Total number of features
currentFeatureType = 0   

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
             
# FeaturesAll[Number of feature].distances.Atom1.Symbol
# FeaturesAll[Number of feature].distances.Atom1.Index
# FeaturesAll[Number of feature].distances.Atom1.Type
# FeaturesAll[Number of feature].distances.Atom1.MolecularIndex
# FeaturesAll[Number of feature].distances.Atom2.Symbol
# FeaturesAll[Number of feature].distances.Atom2.Index
# FeaturesAll[Number of feature].distances.Atom2.Type
# FeaturesAll[Number of feature].distances.Atom2.MolecularIndex
# FeaturesAll[100].powers
# FeaturesAll[100].FeType

print("Records")
# creating the array of all features
i = 0 # N of observation
j = 0 # N of feature
features_array = np.zeros(shape=(Size, NofFeatures), dtype=float)
energy = np.zeros(shape=(Size, 1), dtype=float)
for j in range(0, len(FeaturesAll), 1):
    for i in range(0, len(record_list), 1):
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
            r = r1 * r2 *r3       
        features_array[i, j] = r # store to array
        energy[i, 0] = record_list[i].e # energy

print("Reduced list")
features_array_reduced = np.zeros(shape=(Size, NofFeaturesReduced), dtype=float)
for k in range(0, NofFeaturesReduced, 1):
    for j in range(0, NofFeatures, 1):
        if FeaturesAll[j].FeType == k:
            features_array_reduced[:, k] += features_array[:, j]

print("Saving data")
# save reduced features and energy into file
Table = pd.DataFrame(features_array_reduced, dtype=float)
Table['energy'] = energy
Table.to_csv(F_out_features, index=False)
# save list FeaturesReduced into file
f = open(F_out_structure_FeaturesReduced, "wb")
pickle.dump(FeaturesReduced, f)
f.close()
# save list FeaturesAll into file
f = open(F_out_structure_FeaturesAll, "wb")
pickle.dump(FeaturesAll, f)
f.close()
print("DONE")