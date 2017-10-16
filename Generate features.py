# Generates feature set using coordinates from datafile and according to 
# information in SystemDescriptor
import os
import numpy as np
import pandas as pd
import pickle 
from project1 import structure
from project1 import library
from project1 import spherical
from project1 import IOfunctions
from joblib import Parallel, delayed
import multiprocessing as mp
import re
import time
import shutil
import random

# record_list[0].atoms[0].Atom.Symbol
# record_list[0].atoms[0].Atom.Index
# record_list[0].atoms[0].Atom.Type
# record_list[0].atoms[0].Atom.MolecularIndex
# record_list[0].atoms[0].x
# record_list[0].atoms[0].y
# record_list[0].atoms[0].z
# record_list[0].e
             
# FeaturesAll[0].nDistances
# FeaturesAll[0].FeType
# FeaturesAll[0].Harmonic1
# FeaturesAll[0].Harmonic2
# FeaturesAll[0].DtP1.Power
# FeaturesAll[0].DtP1.DtpType
# FeaturesAll[0].DtP1.Distance.isIntermolecular
# FeaturesAll[0].DtP1.Distance.DiType
# FeaturesAll[0].DtP1.Distance.Atom1.Symbol
# FeaturesAll[0].DtP1.Distance.Atom1.Index
# FeaturesAll[0].DtP1.Distance.Atom1.AtType
# FeaturesAll[0].DtP1.Distance.Atom1.MolecularIndex
# FeaturesAll[0].DtP1.Distance.Atom2.Symbol
# FeaturesAll[0].DtP1.Distance.Atom2.Index
# FeaturesAll[0].DtP1.Distance.Atom2.AtType
# FeaturesAll[0].DtP1.Distance.Atom2.MolecularIndex
# FeaturesAll[0].DtP2.Power
# FeaturesAll[0].DtP2.DtpType
# FeaturesAll[0].DtP2.Distance.isIntermolecular
# FeaturesAll[0].DtP2.Distance.DiType
# FeaturesAll[0].DtP2.Distance.Atom1.Symbol
# FeaturesAll[0].DtP2.Distance.Atom1.Index
# FeaturesAll[0].DtP2.Distance.Atom1.AtType
# FeaturesAll[0].DtP2.Distance.Atom1.MolecularIndex
# FeaturesAll[0].DtP2.Distance.Atom2.Symbol
# FeaturesAll[0].DtP2.Distance.Atom2.Index
# FeaturesAll[0].DtP2.Distance.Atom2.AtType
# FeaturesAll[0].DtP2.Distance.Atom2.MolecularIndex

def ReadData(F_data):
        # Read coordinates from file
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
    record_list = []
    while i < len(data):
        s = data[i].split() # line of text separated in list
        if len(s) == 0: # empty line
            i += 1
            continue
    # record for energy value
        elif (len(s) == 1) and library.isfloat(s[0]): 
            e = float(s[0])
            rec = structure.Record(e, atoms_list)
            record_list.append(rec)
            j = 0
            atoms_list = []
        elif (len(s) == 4): 
            x = float(s[1])
            y = float(s[2])
            z = float(s[3])
            atoms_list.append(structure.AtomCoordinates(Atoms[j], x, y, z))
            j += 1
        i += 1
    return record_list

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

def StoreDistances(F_Distances, record_list, Distances):
    Size = len(record_list)
    nDistances = len(Distances)
    distances_array = np.zeros(shape=(Size, nDistances), dtype=float)
    for i in range(0, Size, 1):
        for j in range(0, nDistances, 1):
            r = np.sqrt((record_list[i].atoms[Distances[j].Atom1.Index].x - record_list[i].atoms[Distances[j].Atom2.Index].x)**2 +\
                        (record_list[i].atoms[Distances[j].Atom1.Index].y - record_list[i].atoms[Distances[j].Atom2.Index].y)**2 +\
                        (record_list[i].atoms[Distances[j].Atom1.Index].z - record_list[i].atoms[Distances[j].Atom2.Index].z)**2)            
            distances_array[i, j] = r
# save distances
    Table = pd.DataFrame(distances_array, dtype=float)
    f = open(F_Distances, 'w')
    Table.to_csv(f, index=False)
    f.close()
    return

def StoreFeatures(F_LinearFeatures, first, last, FeaturesAll, FeaturesReduced, record_list, Atoms):
# Storing energy
    NofFeaturesReduced = len(FeaturesReduced)
    NofFeatures = len(FeaturesAll)
# calculating and storing distances  
    features_array = np.zeros(shape=(last-first, len(FeaturesAll)), dtype=float) 
    for j in range(0, len(FeaturesAll), 1):
        for i in range(first, last, 1):
            if (FeaturesAll[j].nDistances == 1) and (FeaturesAll[j].Harmonic1 is None) and (FeaturesAll[j].Harmonic2 is None):
# features with only one distance. no harmonics
                atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                d = np.sqrt((record_list[i].atoms[atom1_index].x - record_list[i].atoms[atom2_index].x)**2 +\
                            (record_list[i].atoms[atom1_index].y - record_list[i].atoms[atom2_index].y)**2 +\
                            (record_list[i].atoms[atom1_index].z - record_list[i].atoms[atom2_index].z)**2)            
                r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
            if (FeaturesAll[j].nDistances == 2) and (FeaturesAll[j].Harmonic1 is None) and (FeaturesAll[j].Harmonic2 is None):
# features with two distances without harmonics
                atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                d1 = np.sqrt((record_list[i].atoms[atom11_index].x - record_list[i].atoms[atom12_index].x)**2 +\
                             (record_list[i].atoms[atom11_index].y - record_list[i].atoms[atom12_index].y)**2 +\
                             (record_list[i].atoms[atom11_index].z - record_list[i].atoms[atom12_index].z)**2)            
                r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                d2 = np.sqrt((record_list[i].atoms[atom21_index].x - record_list[i].atoms[atom22_index].x)**2 +\
                             (record_list[i].atoms[atom21_index].y - record_list[i].atoms[atom22_index].y)**2 +\
                             (record_list[i].atoms[atom21_index].z - record_list[i].atoms[atom22_index].z)**2)            
                r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power
                r = r1 * r2       
            if (FeaturesAll[j].nDistances == 2) and (FeaturesAll[j].Harmonic1 is not None) and (FeaturesAll[j].Harmonic2 is not None):
# features with two distances and two harmonics            
                atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                d1 = np.sqrt((record_list[i].atoms[atom11_index].x - record_list[i].atoms[atom12_index].x)**2 +\
                             (record_list[i].atoms[atom11_index].y - record_list[i].atoms[atom12_index].y)**2 +\
                             (record_list[i].atoms[atom11_index].z - record_list[i].atoms[atom12_index].z)**2)            
                r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                d2 = np.sqrt((record_list[i].atoms[atom21_index].x - record_list[i].atoms[atom22_index].x)**2 +\
                             (record_list[i].atoms[atom21_index].y - record_list[i].atoms[atom22_index].y)**2 +\
                             (record_list[i].atoms[atom21_index].z - record_list[i].atoms[atom22_index].z)**2)            
                r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power

                center_index = FeaturesAll[j].Harmonic1.Center.Index
                external_index = FeaturesAll[j].Harmonic1.Atom.Index
                new_origin = spherical.Point(record_list[i].atoms[center_index].x, record_list[i].atoms[center_index].y, record_list[i].atoms[center_index].z)
                external_atom = spherical.Point(record_list[i].atoms[external_index].x, record_list[i].atoms[external_index].y, record_list[i].atoms[external_index].z)
                h_list = []
                O_list = []
                for k in range(0, len(Atoms), 1): 
# find two hydrogens that belong to same molecule
                    if (FeaturesAll[j].Harmonic1.Center.MolecularIndex == Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic1.Center.AtType != Atoms[k].AtType):
                        h_list.append(Atoms[k])
# find two oxygens that belong to other molecules                            
                    if (FeaturesAll[j].Harmonic1.Center.MolecularIndex != Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic1.Center.AtType == Atoms[k].AtType):
                        O_list.append(Atoms[k])
                H1_index = h_list[0].Index
                H2_index = h_list[1].Index
                H1 = spherical.Point(record_list[i].atoms[H1_index].x, record_list[i].atoms[H1_index].y, record_list[i].atoms[H1_index].z)
                H2 = spherical.Point(record_list[i].atoms[H2_index].x, record_list[i].atoms[H2_index].y, record_list[i].atoms[H2_index].z)
                if len(O_list) == 1: # two water molecules system
                    O2_index = O_list[0].Index
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    directing_point = O2
                else:
                    O2_index = O_list[0].Index
                    O3_index = O_list[1].Index                    
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    O3 = spherical.Point(record_list[i].atoms[O3_index].x, record_list[i].atoms[O3_index].y, record_list[i].atoms[O3_index].z)
                    directing_point = spherical.get_directing_point_O2_O3(O2, O3)
                theta, phi = spherical.get_angles(new_origin, H1, H2, external_atom, directing_point)
                s1 = spherical.get_real_form2(FeaturesAll[j].Harmonic1.Order, FeaturesAll[j].Harmonic1.Degree, theta, phi)

                center_index = FeaturesAll[j].Harmonic2.Center.Index
                external_index = FeaturesAll[j].Harmonic2.Atom.Index
                new_origin = spherical.Point(record_list[i].atoms[center_index].x, record_list[i].atoms[center_index].y, record_list[i].atoms[center_index].z)
                external_atom = spherical.Point(record_list[i].atoms[external_index].x, record_list[i].atoms[external_index].y, record_list[i].atoms[external_index].z)
                h_list = []
                O_list = []
                for k in range(0, len(Atoms), 1): 
# find two hydrogens that belong to same molecule
                    if (FeaturesAll[j].Harmonic2.Center.MolecularIndex == Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic2.Center.AtType != Atoms[k].AtType):
                        h_list.append(Atoms[k])
# find two oxygens that belong to other molecules                            
                    if (FeaturesAll[j].Harmonic2.Center.MolecularIndex != Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic2.Center.AtType == Atoms[k].AtType):
                        O_list.append(Atoms[k])
                H1_index = h_list[0].Index
                H2_index = h_list[1].Index
                H1 = spherical.Point(record_list[i].atoms[H1_index].x, record_list[i].atoms[H1_index].y, record_list[i].atoms[H1_index].z)
                H2 = spherical.Point(record_list[i].atoms[H2_index].x, record_list[i].atoms[H2_index].y, record_list[i].atoms[H2_index].z)
                if len(O_list) == 1: # two water molecules system
                    O2_index = O_list[0].Index
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    directing_point = O2
                else:
                    O2_index = O_list[0].Index
                    O3_index = O_list[1].Index                    
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    O3 = spherical.Point(record_list[i].atoms[O3_index].x, record_list[i].atoms[O3_index].y, record_list[i].atoms[O3_index].z)
                    directing_point = spherical.get_directing_point_O2_O3(O2, O3)                    
                theta, phi = spherical.get_angles(new_origin, H1, H2, external_atom, directing_point)
                s2 = spherical.get_real_form2(FeaturesAll[j].Harmonic2.Order, FeaturesAll[j].Harmonic2.Degree, theta, phi)

                r = r1*r2*s1*s2
                
            features_array[i-first, j] = r # store to array
# sum features with equal FeType
    features_array_reduced = np.zeros(shape=(last-first, NofFeaturesReduced), dtype=float)
    for k in range(0, NofFeaturesReduced, 1):
        for j in range(0, NofFeatures, 1):
            if (FeaturesAll[j].FeType == FeaturesReduced[k].FeType):
                features_array_reduced[:, k] += features_array[:, j]


# removing NaN from dataset
#    mask = ~np.any(np.isnan(features_array_reduced), axis=1)
#    features_array_reduced = features_array_reduced[mask]
#    energy = energy[mask]
# save reduced features and energy into file
    Table = pd.DataFrame(features_array_reduced, dtype=float)
    f = open(F_LinearFeatures, 'a')
    if first == 0:
        Table.to_csv(f, index=False)
    else:
        Table.to_csv(f, index=False, header=False)
    f.close()
    return
# end of StoreFeatures

if __name__ == '__main__':
    F_SystemDescriptor = 'SystemDescriptor.' # file with info about system structure
    F_Response_Train = 'ResponseTrain.csv' # response variable (y)
    F_Response_Test = 'ResponseTest.csv' # response variable (y)
    F_LinearFeaturesTrain = 'LinearFeaturesTrain.csv' # output csv file with combined features and energy
    F_LinearFeaturesTest = 'LinearFeaturesTest.csv' # output csv file with combined features and energy
    F_Distances_Train = 'Distances Train.csv' # output csv file. distances
    F_Distances_Test = 'Distances Test.csv' # output csv file. distances
    F_NonlinearFeatures = 'NonlinearFeatures.dat'
    F_LinearFeaturesAll = 'LinearFeaturesAll.dat' # output data structure which contains all features
    F_LinearFeaturesReduced = 'LinearFeaturesReduced.dat' # output data structure which contains combined features
    F_System = 'system.dat' # output data system structure
    F_record_list = 'records.dat'
    F_LinearFeaturesList = 'Linear Features Reduced List.xlsx'
    F_NonlinearFeaturesList = 'Nonlinear Features List.xlsx'
    F_Structure = 'Structure.xlsx'
    F_Filter = 'Filter.dat'
    Separators = '=|,| |:|;|: '   
    RandomSeed = 101
    if RandomSeed is not None:
        random.seed(RandomSeed)
    else:
        random.seed()
    try:        
        os.remove(F_Response_Train)
        os.remove(F_Response_Test)
        os.remove(F_LinearFeaturesTrain) # erase old files if exist
        os.remove(F_LinearFeaturesTest)
        os.remove(F_Distances_Train)
        os.remove(F_Distances_Test)
        os.remove(F_NonlinearFeatures)
        os.remove(F_LinearFeaturesAll) 
        os.remove(F_LinearFeaturesReduced) 
        os.remove(F_System)
        os.remove(F_record_list)
        os.remove(F_LinearFeaturesList)
        os.remove(F_Structure)
    except:
        pass    
    f = open(F_Filter, "rb")
    Filter = pickle.load(f)
    f.close()
    F_train_data = Filter['Training Set']
    F_test_data = Filter['Test Set']
    nTrainPoints = Filter['Train records number']
    nTestPoints = Filter['Test records number']
    # read descriptor from file
    with open(F_SystemDescriptor) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string
    ProceedSingle = False
    ProceedDouble = False
    ProceedHarmonics = False    
    F_data = ''
    for i in range(0, len(lines), 1):
        x = lines[i]
        if len(x) == 0:
            continue
        if x[0] == '#':
            continue
        if (x.find('SingleDistancesInclude') != -1):
            s = re.split(Separators, x)
            s = list(filter(bool, s)) # removes empty records
            if 'True' in s: # proceed single distances
                ProceedSingle = True
        if (x.find('DoubleDistancesInclude') != -1):
            s = re.split(Separators, x)
            s = list(filter(bool, s)) # removes empty records
            if 'True' in s: # proceed single distances
                ProceedDouble = True
        if (x.find('HarmonicsInclude') != -1):
            s = re.split(Separators, x)
            s = list(filter(bool, s)) # removes empty records
            if 'True' in s: # proceed single distances
                ProceedHarmonics = True
    if ProceedSingle:
        if ('&SingleDistancesDescription' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&SingleDistancesDescription') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endSingleDistancesDescription') != -1):
                    Last = i # index +1 of last record of single distance description
            SingleDescription = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                SingleDescription.append(s)
            del(First)
            del(Last)
        if ('&DefaultSingleDistances' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('SinglePowers') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    SinglePowersDefault = list(filter(bool, s)) # removes empty records
                    del(SinglePowersDefault[0])
    if ProceedDouble:
        if ('&DoubleDistancesDescription' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&DoubleDistancesDescription') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endDoubleDistancesDescription') != -1):
                    Last = i # index +1 of last record of single distance description
            DoubleDescription = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                DoubleDescription.append(s)
            del(First)
            del(Last)
        if ('&DefaultDoubleDistances' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('DoublePowers') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    DoublePowersDefault = list(filter(bool, s)) # removes empty records
                    del(DoublePowersDefault[0])
                if (x.find('IncludeAllExcept') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        IncludeAllExcept = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        IncludeAllExcept = False
                if (x.find('ExcludeAllExcept') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        ExcludeAllExcept = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        ExcludeAllExcept = False 
                if (x.find('IncludeSameType') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        IncludeSameType = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        IncludeSameType = False                    
        if ('&IncludeExcludeList' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&IncludeExcludeList') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endIncludeExcludeList') != -1):
                    Last = i # index +1 of last record of single distance description
            IncludeExcludeList = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                IncludeExcludeList.append(s)
    else:
        DtP_Double_list = []
    if ProceedHarmonics:
        if ('&HarmonicDescription' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&HarmonicDescription') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endHarmonicDescription') != -1):
                    Last = i # index +1 of last record of single distance description
            HarmonicDescription = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                HarmonicDescription.append(s)
            del(First)
            del(Last)
        if ('&Harmonics' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('Order') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    o = list(filter(bool, s)) # removes empty records
                    del(o[0])    
                    Order = []
                    for j in range(0, len(o), 1):
                        Order.append(int(o[j]))
                if (x.find('Degree') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    Degree = list(filter(bool, s)) # removes empty records
                    del(Degree[0])              
                if (x.find('Degree') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    d = list(filter(bool, s)) # removes empty records
                    del(d[0])   
                    Degree = []
                    for j in range(0, len(d), 1):
                        Degree.append(int(d[j]))
                if (x.find('HarmonicCenter') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s)) # removes empty records
                    HarmonicCenter = s[1]
                if (x.find('HarmonicAtoms') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s)) # removes empty records
                    del(s[0])
                    HarmonicAtoms = s
        if ('&DefaultHarmonics' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('HarmonicPowers') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    HarmonicPowersDefault = list(filter(bool, s)) # removes empty records
                    del(HarmonicPowersDefault[0])
                if (x.find('IncludeHarmonicAllExcept') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        IncludeHarmonicAllExcept = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        IncludeHarmonicAllExcept = False
                if (x.find('ExcludeHarmonicAllExcept') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        ExcludeHarmonicAllExcept = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        ExcludeHarmonicAllExcept = False 
                if (x.find('IncludeHarmonicSameType') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        IncludeHarmonicSameType = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        IncludeHarmonicSameType = False   
        if ('&IncludeExcludeHarmonicList' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&IncludeExcludeHarmonicList') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endIncludeExcludeHarmonicList') != -1):
                    Last = i # index +1 of last record of single distance description
            IncludeExcludeHarmonicList = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                IncludeExcludeHarmonicList.append(s)             
                    
    if ('&SYSTEM' in lines):
        i = 0 # index of line in file
        Atoms = [] # create list of atom structures from file
        while ((lines[i].find('&SYSTEM') == -1) and (i < len(lines))):
            i += 1
        i += 1 # next line in file after &SYSTEM
        j = 0 # atoms order in the system 
        types_list = []
        idx_list = []
        molecules_idx_list = []
        molecules = []
        k = 0 # types of atom index
        while ((lines[i].find('&endSYSTEM') == -1) and (i < len(lines))):
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
                molecules.append(structure.Molecule(one_molecule_atoms, Name=MoleculeName))
            else:
                i += 1
           
    nMolecules = len(molecules_idx_list)                
    nAtTypes = len(types_list)       
    nAtoms = len(Atoms)
    Prototypes = IOfunctions.ReadMoleculeDescription(F_SystemDescriptor) # read molecule prototypes 
    MoleculeNames = []
    for i in Prototypes:
        MoleculeNames.append(i.Name)
    for i in molecules:
        if i.Name in MoleculeNames:
            idx = MoleculeNames.index(i.Name)
            i.Mass = Prototypes[idx].Mass
            for j in range(0, len(i.Atoms), 1):
                i.Atoms[j].Atom.Mass = Prototypes[idx].Atoms[j].Atom.Mass
                i.Atoms[j].Atom.Radius = Prototypes[idx].Atoms[j].Atom.Radius
                i.Atoms[j].Atom.Bonds = Prototypes[idx].Atoms[j].Atom.Bonds
                i.Atoms[j].Atom.Bonds = library.replace_numbers(i.AtomIndex, Prototypes[idx].AtomIndex, i.Atoms[j].Atom.Bonds)
    for i in molecules:
        i._refresh() # update data from prototype
    # determine number of atom types from list
    Distances = [] # create list of distances from list of atoms
    for i in range(0, nAtoms, 1):
        for j in range(i+1, nAtoms, 1):
            Distances.append(structure.Distance(Atoms[i], Atoms[j]))
    DiTypeList = []
    for i in Distances:
        if i.DiType not in DiTypeList:
            DiTypeList.append(i.DiType)
    nDistances = len(Distances)
    nDiTypes = len(DiTypeList)
    system = structure.System(Atoms=Atoms, Molecules=molecules, Prototypes=Prototypes,\
        nAtoms=nAtoms, nAtTypes=nAtTypes, nMolecules=nMolecules, Distances=Distances,\
        nDistances=nDistances, nDiTypes=nDiTypes)

    FeaturesNonlinear = []    
    for i in range(0, nDistances, 1):
        FeaturesNonlinear.append(structure.FeatureNonlinear(i, Distances[i], FeType='exp', nDistances=1, nConstants=2))
        
        
    FeaturesAll = [] # list of all features
    
    if ProceedSingle:
        # add DiType for each record   
        SingleDescriptionDiType = []                 
        for i in range(0, len(SingleDescription), 1):
            a1 = SingleDescription[i][0]
            a2 = SingleDescription[i][1]
            if SingleDescription[i][2] == 'intermolecular':
                inter = True
            else:
                inter = False
            for j in Distances:
                if j.isIntermolecular == inter:
                    if ((j.Atom1.Symbol == a1) and (j.Atom2.Symbol == a2)) or \
                        ((j.Atom1.Symbol == a2) and (j.Atom2.Symbol == a1)):
                        SingleDescriptionDiType.append(j.DiType)
                        break
        for i in DiTypeList:
            if i not in SingleDescriptionDiType:
                SingleDescriptionDiType.append(i)
                for j in Distances:
                    if j.DiType == i:
                        idx = j
                        break
                if idx.isIntermolecular:
                    inter = 'intermolecular'
                else:
                    inter = 'intramolecular'
                SingleDescription.append([idx.Atom1.Symbol, idx.Atom2.Symbol, inter])
                SingleDescription[-1] += SinglePowersDefault

    # make list of features with only one distance
        DtP_Single_list = []
        for i in range(0, len(Distances), 1):
            k = SingleDescriptionDiType.index(Distances[i].DiType)
            Powers = SingleDescription[k][3:]
            for j in Powers:
                DtP_Single_list.append(structure.Distance_to_Power(Distances[i], int(j)))
#                DtP_Single_list.append(class2.Distance_to_Power(Distances[i], float(j)))
    
        for i in DtP_Single_list:
            FeaturesAll.append(structure.Feature(i, DtP2=None))        
            
    if ProceedDouble:
        # add DiType for each record   
        DoubleDescriptionDiType = []                 
        for i in range(0, len(DoubleDescription), 1):
            a1 = DoubleDescription[i][0]
            a2 = DoubleDescription[i][1]
            if DoubleDescription[i][2] == 'intermolecular':
                inter = True
            else:
                inter = False
            for j in Distances:
                if j.isIntermolecular == inter:
                    if ((j.Atom1.Symbol == a1) and (j.Atom2.Symbol == a2)) or \
                        ((j.Atom1.Symbol == a2) and (j.Atom2.Symbol == a1)):
                        DoubleDescriptionDiType.append(j.DiType)
                        break
        for i in DiTypeList:
            if i not in DoubleDescriptionDiType:
                DoubleDescriptionDiType.append(i)
                for j in Distances:
                    if j.DiType == i:
                        idx = j
                        break
                if idx.isIntermolecular:
                    inter = 'intermolecular'
                else:
                    inter = 'intramolecular'
                DoubleDescription.append([idx.Atom1.Symbol, idx.Atom2.Symbol, inter])
                DoubleDescription[-1] += DoublePowersDefault

    # make list of features with only one distance
        DtP_Double_list = []
        for i in range(0, len(Distances), 1):
            k = DoubleDescriptionDiType.index(Distances[i].DiType)
            Powers = DoubleDescription[k][3:]
            for j in Powers:
                DtP_Double_list.append(structure.Distance_to_Power(Distances[i], int(j)))
#                DtP_Double_list.append(structure.Distance_to_Power(Distances[i], float(j)))
        
        IncludeExcludeDiTypes = [] # can be empty
        for i in IncludeExcludeList:
            a11 = i[0]
            a12 = i[1]
            a21 = i[3]
            a22 = i[4]
            if i[2] == 'intermolecular':
                inter1 = True
            else:
                inter1 = False
            if i[5] == 'intermolecular':
                inter2 = True
            else:
                inter2 = False
            for j in Distances:
                if j.isIntermolecular == inter1:
                    if ((j.Atom1.Symbol == a11) and (j.Atom2.Symbol == a12)) or \
                        ((j.Atom1.Symbol == a12) and (j.Atom2.Symbol == a11)):
                        Type1 = j.DiType
                        break
            for j in Distances:
                if j.isIntermolecular == inter2:
                    if ((j.Atom1.Symbol == a21) and (j.Atom2.Symbol == a22)) or \
                        ((j.Atom1.Symbol == a22) and (j.Atom2.Symbol == a21)):
                        Type2 = j.DiType
                        break
            IncludeExcludeDiTypes.append((Type1, Type2))

        for i in range(0, len(DtP_Double_list), 1):
            for j in range(i+1, len(DtP_Double_list), 1):
                if DtP_Double_list[i].Power > DtP_Double_list[j].Power:
                    continue # skip duplicates
                if not IncludeSameType:
                    if DtP_Double_list[i].Distance.DiType == DtP_Double_list[j].Distance.DiType: # skip if distances of the same type
                        continue
                if len(IncludeExcludeDiTypes) == 0:
                    FeaturesAll.append(structure.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j]))
                else:
                    for k in IncludeExcludeDiTypes:
                        if ExcludeAllExcept:
                            if (DtP_Double_list[i].Distance.DiType == k[0] and DtP_Double_list[j].Distance.DiType == k[1]) or (DtP_Double_list[i].Distance.DiType == k[1] and DtP_Double_list[j].Distance.DiType == k[0]):
                               FeaturesAll.append(structure.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j])) # append if match
                        if IncludeAllExcept:
                            if (DtP_Double_list[i].Distance.DiType == k[0] and DtP_Double_list[j].Distance.DiType == k[1]) or (DtP_Double_list[i].Distance.DiType == k[1] and DtP_Double_list[j].Distance.DiType == k[0]):
                                continue #skip if match
                            FeaturesAll.append(structure.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j]))

    if ProceedHarmonics:
        # add DiType for each record   
        HarmonicDescriptionDiType = []                 
        for i in range(0, len(HarmonicDescription), 1):
            a1 = HarmonicDescription[i][0]
            a2 = HarmonicDescription[i][1]
            if HarmonicDescription[i][2] == 'intermolecular':
                inter = True
            else:
                inter = False
            for j in Distances:
                if j.isIntermolecular == inter:
                    if ((j.Atom1.Symbol == a1) and (j.Atom2.Symbol == a2)) or \
                        ((j.Atom1.Symbol == a2) and (j.Atom2.Symbol == a1)):
                        HarmonicDescriptionDiType.append(j.DiType)
                        break
        for i in DiTypeList:
            if i not in HarmonicDescriptionDiType:
                HarmonicDescriptionDiType.append(i)
                for j in Distances:
                    if j.DiType == i:
                        idx = j
                        break
                if idx.isIntermolecular:
                    inter = 'intermolecular'
                else:
                    inter = 'intramolecular'
                HarmonicDescription.append([idx.Atom1.Symbol, idx.Atom2.Symbol, inter])
                HarmonicDescription[-1] += HarmonicPowersDefault

    # make list of features with only one distance
    
        DtP_Harmonic_list = []
        for i in range(0, len(Distances), 1):
            k = HarmonicDescriptionDiType.index(Distances[i].DiType)
            Powers = HarmonicDescription[k][3:]
            for j in Powers:
                DtP_Harmonic_list.append(structure.Distance_to_Power(Distances[i], int(j)))
        
        IncludeExcludeHarmonicDiTypes = [] # can be empty
        for i in IncludeExcludeHarmonicList:
            a11 = i[0]
            a12 = i[1]
            a21 = i[3]
            a22 = i[4]
            if i[2] == 'intermolecular':
                inter1 = True
            else:
                inter1 = False
            if i[5] == 'intermolecular':
                inter2 = True
            else:
                inter2 = False
            for j in Distances:
                if j.isIntermolecular == inter1:
                    if ((j.Atom1.Symbol == a11) and (j.Atom2.Symbol == a12)) or \
                        ((j.Atom1.Symbol == a12) and (j.Atom2.Symbol == a11)):
                        Type1 = j.DiType
                        break
            for j in Distances:
                if j.isIntermolecular == inter2:
                    if ((j.Atom1.Symbol == a21) and (j.Atom2.Symbol == a22)) or \
                        ((j.Atom1.Symbol == a22) and (j.Atom2.Symbol == a21)):
                        Type2 = j.DiType
                        break
            IncludeExcludeHarmonicDiTypes.append((Type1, Type2))

        for i in range(0, len(DtP_Harmonic_list), 1):
            for j in range(i, len(DtP_Harmonic_list), 1):
                for li in Degree:
                    for mi in Order:
                        if abs(mi) > li:
                            continue
                        for lj in Degree:
                            for mj in Order:
                                if abs(mj) > lj:
                                    continue
                                center_i = DtP_Harmonic_list[i].Distance.Atom1 # Oxygen1
                                atom_i = DtP_Harmonic_list[i].Distance.Atom2 # Hydrogen1
                                center_j = DtP_Harmonic_list[j].Distance.Atom1 # Oxygen2
                                atom_j = DtP_Harmonic_list[j].Distance.Atom2 # Hydrogen2
                                if center_i.MolecularIndex != center_j.MolecularIndex:
                                    continue # skip if centers are different
                                if atom_i.MolecularIndex == atom_j.MolecularIndex:
                                    continue # skip if atoms of same molecule
                                if atom_i.Index == atom_j.Index:
                                    continue # skip if external atoms are same
                                if DtP_Harmonic_list[i].Power > DtP_Harmonic_list[j].Power:
                                    continue # skip duplicates
                                FeaturesAll.append(structure.Feature(DtP_Harmonic_list[i], \
                                    DtP2=DtP_Harmonic_list[j], Harmonic1=structure.Harmonic(mi, li, center_i, atom_i),\
                                    Harmonic2=structure.Harmonic(mj, lj, center_j, atom_j)))
                        
    # Make list of reduced features
    FeaturesReduced = []
    FeType_list = []
    for i in range(0, len(FeaturesAll), 1):
        if (FeaturesAll[i].FeType not in FeType_list):
            FeType_list.append(FeaturesAll[i].FeType)
            FeaturesReduced.append(FeaturesAll[i])
# store global indices for each reduced feature
    for k in range(0, len(FeaturesReduced), 1):
        for j in range(0, len(FeaturesAll), 1):
            if (FeaturesAll[j].FeType == FeaturesReduced[k].FeType):
                if j not in FeaturesReduced[k].idx:
                    FeaturesReduced[k].idx.append(j)
                
    NofFeatures = len(FeaturesAll) # Total number of features
    NofFeaturesReduced = len(FeaturesReduced)
    
    # save list FeaturesNonlinear into file
    f = open(F_NonlinearFeatures, "wb")
    pickle.dump(FeaturesNonlinear, f)
    f.close()
    
    # save list FeaturesAll into file
    f = open(F_LinearFeaturesAll, "wb")
    pickle.dump(FeaturesAll, f)
    f.close()
    
    # save list FeaturesReduced into file
    f = open(F_LinearFeaturesReduced, "wb")
    pickle.dump(FeaturesReduced, f)
    f.close()

    # save system object into file
    f = open(F_System, "wb")
    pickle.dump(system, f)
    f.close()
    
    library.StoreNonlinearFeaturesDescriprion(F_NonlinearFeaturesList, FeaturesNonlinear) # xlsx
    library.StoreLinearFeaturesDescriprion(F_LinearFeaturesList, FeaturesAll, FeaturesReduced) # xlsx
    library.store_structure(F_Structure, Atoms, Distances, DtP_Double_list, FeaturesAll) # xlsx
    record_list_train = ReadData(F_train_data)
    record_list_test = ReadData(F_test_data)
    record_list = record_list_train + record_list_test
    StoreDistances(F_Distances_Train, record_list_train, Distances)
    StoreDistances(F_Distances_Test, record_list_test, Distances)  
    StoreEnergy(F_Response_Train, record_list_train)
    StoreEnergy(F_Response_Test, record_list_test)
        
    print('Train = ', len(record_list_train))
    print('# train points', nTrainPoints)
    print('Test = ', len(record_list_test))
    print('# test points', nTestPoints)
    print('All = ', len(record_list))

# split array if too big
    NpArrayCapacity = 1e+8
    Size = len(record_list) # N of observations
    Length = len(FeaturesAll)
    if (Size * Length) > NpArrayCapacity:
        BufSize = int(NpArrayCapacity / Length)
    else:
        BufSize = Size
    # create endpoints for array size_list[1 - inf][0 - 1]
    i = 0 # Number of observation
    j = 0 # Number of feature
    size_list = []
    size_list_str = []
    nCPU = mp.cpu_count()
#    nCPU = 1
    print('Start Multiprocessing with ', nCPU, ' cores')
    size = int(Size / nCPU)
    first = 0
    if size < BufSize:
        for i in range(0, nCPU, 1):
            first = i * size
            last = (i+1) * size
            if i == (nCPU-1):
                last = Size
            size_list.append((first, last))
            size_list_str.append(str(first) + '-' + str(last-1) + '.csv')
    else:# if number of records is huge
        i = 0
        last = 0
        while last != Size:
            first = i * BufSize
            last = (i+1) * BufSize
            if last > Size:
                last = Size
            size_list.append((first, last))
            size_list_str.append(str(first) + '-' + str(last-1) + '.csv')
            i += 1


    ran = list(range(0, len(size_list_str), 1))
    jobs = (delayed(StoreFeatures)(size_list_str[i], size_list[i][0], size_list[i][1], FeaturesAll, FeaturesReduced, record_list, Atoms) for i in ran)
    N = Parallel(n_jobs=nCPU)(jobs)
    print('Storing results in one file')
    f = open('Tmp.csv', "w")
    for i in range(0, len(size_list_str), 1):
        fin = open(size_list_str[i], "r")
        S = fin.readlines()
        f.writelines(S)
        fin.close()
    f.close()
    f = open('Tmp.csv', "r")
    data = f.readlines()
    f.close()
    os.remove('Tmp.csv')
    f = open(F_LinearFeaturesTrain, "w")
    i = 0
    while i <= nTrainPoints:
        f.write(data[i])
        i += 1
    f.close()
    if i < len(data):
        f = open(F_LinearFeaturesTest, "w")
        f.write(data[0])
        while i < len(data):
            f.write(data[i])
            i += 1
        f.close()       
    for i in range(0, len(size_list_str), 1):
        try:
            os.remove(size_list_str[i]) # erase old files if exist
        except:
            pass
            
    directory = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    if not os.path.exists(directory):
        os.makedirs(directory)    
    try:
        shutil.copyfile(F_SystemDescriptor, directory + '\\' + F_SystemDescriptor)
        shutil.copyfile(F_Response_Train, directory + '\\' + F_Response_Train)
        shutil.copyfile(F_Response_Test, directory + '\\' + F_Response_Test)
        shutil.copyfile(F_LinearFeaturesTrain, directory + '\\' + F_LinearFeaturesTrain)
        shutil.copyfile(F_LinearFeaturesTest, directory + '\\' + F_LinearFeaturesTest)
        shutil.copyfile(F_Distances_Train, directory + '\\' + F_Distances_Train)
        shutil.copyfile(F_Distances_Test, directory + '\\' + F_Distances_Test)
        shutil.copyfile(F_LinearFeaturesAll, directory + '\\' + F_LinearFeaturesAll)
        shutil.copyfile(F_NonlinearFeatures, directory + '\\' + F_NonlinearFeatures)
        shutil.copyfile(F_LinearFeaturesReduced, directory + '\\' + F_LinearFeaturesReduced)
        shutil.copyfile(F_System, directory + '\\' + F_System)
        shutil.copyfile(F_record_list, directory + '\\' + F_record_list)
        shutil.copyfile(F_LinearFeaturesList, directory + '\\' + F_LinearFeaturesList)
        shutil.copyfile(F_Structure, directory + '\\' + F_Structure)
        shutil.copyfile(F_NonlinearFeaturesList, directory + '\\' + F_NonlinearFeaturesList)
    except:
        pass

    print("DONE")

