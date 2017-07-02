import os
import numpy as np
import pandas as pd
import pickle 
from structure import class2
from structure import library2
from structure import spherical
from joblib import Parallel, delayed
import multiprocessing as mp # Pool, freez_support
import re

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

def StoreFeatures(F_out_features, first, last, FeaturesAll, FeaturesReduced, record_list, Atoms):
# Storing energy
    NofFeaturesReduced = len(FeaturesReduced)
    NofFeatures = len(FeaturesAll)
    energy = np.zeros(shape=(last - first, 1), dtype=float)
    for i in range(first, last, 1):
        energy[i-first, 0] = record_list[i].e # energy
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
    mask = ~np.any(np.isnan(features_array_reduced), axis=1)
    features_array_reduced = features_array_reduced[mask]
    energy = energy[mask]
# save reduced features and energy into file
    Table = pd.DataFrame(features_array_reduced, dtype=float)
    Table['energy'] = energy
    f = open(F_out_features, 'a')
    if first == 0:
        Table.to_csv(f, index=False)
    else:
        Table.to_csv(f, index=False, header=False)
    f.close()
    return
# end of StoreFeatures

if __name__ == '__main__':
    F = 'SystemDescriptor.' # file with info about system structure
    #F_data = 'datafile short.x'
    #F_data = 'short.x'
    #F_data = 'short three water molecules.x'
    #F_data = 'datafile1 from github gaussian process.x' # file with coordinates
    #F_data = 'datafile2.x'
    #F_data = 'datafile3 2 water molecules.x'
    F_data = 'datafile4 3 water molecules small.x'
    #F_data = 'datafile5 3 water molecules big.x'
    F_HarmonicFeatures = 'Harmonic Features.csv' # output csv file with combined features and energy
    F_HarmonicFeaturesAll = 'HarmonicFeaturesAll.dat' # output data structure which contains all features
    F_HarmonicFeaturesReduced = 'HarmonicFeaturesReduced.dat' # output data structure which contains combined features
    FileName = 'Harmonic Features Reduced List.xlsx'
    # temporary variables. to be replaced
    Degree = [0, 1]
    Order = [-1, 0, 1]
    Separators = '=|,| |:|;|: |'
    
    try:
        os.remove(F_HarmonicFeatures) # erase old files if exist
        os.remove(F_HarmonicFeaturesAll) 
        os.remove(F_HarmonicFeaturesReduced) 
    except:
        pass    
    # read descriptor from file
    with open(F) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string
    ProceedSingle = False
    ProceedDouble = False
    ProceedHarmonics = False    
    for i in range(0, len(lines), 1):
        x = lines[i]
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
        if (x.find('F_data') != -1):        
            s = re.split(Separators, x)
            s = list(filter(bool, s))  
            F_data = s[1]
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
    if ProceedHarmonics:
        pass
    if ('&SYSTEM' in lines):
        i = 0 # index of line in file
        Atoms = [] # create list of atom structures from file
        while ((lines[i].find('&SYSTEM') == -1) & (i < len(lines))):
            i += 1
        i += 1 # next line after &SYSTEM
        j = 0 # order in the system 
        while ((lines[i].find('&END') == -1) & (i < len(lines))):
            if (lines[i][0] == '#'):
                i += 1
                continue
            else:
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s))
                Atoms.append(class2.Atom(s[0], j, int(s[1]), int(s[2])))
                j += 1
                i += 1
    else:
        quit()
    
    # determine number of atom types from list
    nAtTypes = 0        
    for i in range(0, len(Atoms), 1):
        if (Atoms[i].AtType > nAtTypes):
            nAtTypes = Atoms[i].AtType
    nAtTypes += 1        
    
    nAtoms = len(Atoms)
    Distances = [] # create list of distances from list of atoms
    for i in range(0, nAtoms, 1):
        for j in range(i+1, nAtoms, 1):
            Distances.append(class2.Distance(Atoms[i], Atoms[j]))
    DiTypeList = []
    for i in Distances:
        if i.DiType not in DiTypeList:
            DiTypeList.append(i.DiType)
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
                DtP_Single_list.append(class2.Distance_to_Power(Distances[i], int(j)))
    
        for i in DtP_Single_list:
            FeaturesAll.append(class2.Feature(i, DtP2=None))        
            
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
                DtP_Double_list.append(class2.Distance_to_Power(Distances[i], int(j)))
        
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
                if not IncludeSameType:
                    if DtP_Double_list[i].Distance.DiType == DtP_Double_list[j].Distance.DiType: # skip if distances of the same type
                        continue
                if len(IncludeExcludeDiTypes) == 0:
                    FeaturesAll.append(class2.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j]))
                else:
                    for k in IncludeExcludeDiTypes:
                        if ExcludeAllExcept:
                            if (DtP_Double_list[i].Distance.DiType == k[0] and DtP_Double_list[j].Distance.DiType == k[1]) or (DtP_Double_list[i].Distance.DiType == k[1] and DtP_Double_list[j].Distance.DiType == k[0]):
                               FeaturesAll.append(class2.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j])) # append if match
                        if IncludeAllExcept:
                            if (DtP_Double_list[i].Distance.DiType == k[0] and DtP_Double_list[j].Distance.DiType == k[1]) or (DtP_Double_list[i].Distance.DiType == k[1] and DtP_Double_list[j].Distance.DiType == k[0]):
                                continue #skip if match
                            FeaturesAll.append(class2.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j]))
                        
    #    HarmonicDescriptionFirst = 0 #if they have the same value use only default
    #    HarmonicDescriptionLast = 0
    #    Hamonics_include = False
    #    for i in range(0, len(lines), 1):
    #        x = lines[i]
    #        if (x.find('&HarmonicsDescription') != -1):
    #            HarmonicDescriptionFirst = i + 1
    #        if (x.find('&endHarmonicsDescription') != -1):   
    #            HarmonicDescriptionLast = i
    #    if HarmonicDescriptionFirst != HarmonicDescriptionLast:
    #        for i in range(HarmonicDescriptionFirst, HarmonicDescriptionLast, 1):
    #            x = lines[i]
    #            if (x.find('Order') != -1):
    #                Str_order = x.split(',', -1)
    #                for k in range(1, len(Str_order), 1):
    #                    Str_order[k] = int(Str_order[k])
    #            if (x.find('Degree') != -1):
    #                Str_degree = x.split(',', -1)   
    #                for k in range(1, len(Str_degree), 1):
    #                    Str_degree[k] = int(Str_degree[k])   
    #            if (x.find('Center') != -1):
    #                Str_center = x.split(',', -1)   
    #        Order_list = Str_order[1:]
    #        Degree_list = Str_degree[1:]
    #        Center = Str_center[1]
    #        Hamonics_include = True
            
        # list of harmonics. only between O and H from different molecules
    #    H_list = []
    #    for i in Degree:
    #        for j in Order:
    #            if abs(j) > i:
    #                continue # abs value of Order must be <= Degree
    #            for k in range(0, len(Atoms), 1): # centers
    #                if Atoms[k].Symbol != 'O': # can also use Atoms[k].Symbol != 'O'
    #                    continue # if not oxygen
    #                for l in range(0, len(Atoms), 1): # hydrogens
    #                    if Atoms[l].Symbol != 'H': # can also use Atoms[l].Symbol != 'H'
    #                        continue # if not hydrogen
    #                    if Atoms[k].MolecularIndex == Atoms[l].MolecularIndex:
    #                        continue # O and H belong to the same molecule
    #                    H_list.append(class2.Harmonic(j, i, Atoms[k], Atoms[l]))
                        
        # include r*r*H*H only O-H intermolecular distances and harmonics
    #    for i in range(0, len(DtP_list), 1): # first O-H distance
    #        if not DtP_list[i].Distance.isIntermolecular: 
    #            continue # if distance is not intermolecular skip it
    #        if (DtP_list[i].Distance.Atom1.AtType == DtP_list[i].Distance.Atom2.AtType):
    #            continue # if distance is between same types of atoms - skip it (O-O or H-H)
    #        for j in range(0, len(DtP_list), 1): # second O-H distance
    #            if not DtP_list[j].Distance.isIntermolecular: 
    #                continue # if distance is not intermolecular skip it
    #            if (DtP_list[j].Distance.Atom1.AtType == DtP_list[j].Distance.Atom2.AtType):
    #                continue
    #            for k in range(0, len(H_list), 1): # first harmonic corresponds to DtP1
    #                if not(((H_list[k].Center.Index == DtP_list[i].Distance.Atom1.Index) and (H_list[k].Atom.Index == DtP_list[i].Distance.Atom2.Index)) or ((H_list[k].Center.Index == DtP_list[i].Distance.Atom2.Index) and (H_list[k].Atom.Index == DtP_list[i].Distance.Atom1.Index))):
    #                    continue    
    #                for l in range(0, len(H_list), 1): # second harmonic corresponds to DtP2
    #                    if not(((H_list[l].Center.Index == DtP_list[j].Distance.Atom1.Index) and (H_list[l].Atom.Index == DtP_list[j].Distance.Atom2.Index)) or ((H_list[l].Center.Index == DtP_list[j].Distance.Atom2.Index) and (H_list[l].Atom.Index == DtP_list[j].Distance.Atom1.Index))):
    #                        continue    
    #                    FeaturesAll.append(class2.Feature(DtP_list[i], DtP2=DtP_list[j], Harmonic1=H_list[k], Harmonic2=H_list[l]))
                            
    # Make list of reduced features
    FeaturesReduced = []
    FeType_list = []
    for i in range(0, len(FeaturesAll), 1):
        if (FeaturesAll[i].FeType not in FeType_list):
            FeType_list.append(FeaturesAll[i].FeType)
            FeaturesReduced.append(FeaturesAll[i])
    
    NofFeatures = len(FeaturesAll) # Total number of features
    NofFeaturesReduced = len(FeaturesReduced)
    
    # save list FeaturesAll into file
    f = open(F_HarmonicFeaturesAll, "wb")
    pickle.dump(FeaturesAll, f)
    f.close()
    
    # save list FeaturesReduced into file
    f = open(F_HarmonicFeaturesReduced, "wb")
    pickle.dump(FeaturesReduced, f)
    f.close()
    
    library2.StoreFeaturesDescriprion(FileName, FeaturesAll, FeaturesReduced)

    # Read coordinates from file
    f = open(F_data, "r")
    data0 = f.readlines()
    f.close()
    data1 = []
    for i in range(0, len(data0), 1):
        data1.append(data0[i].rstrip())
    del(data0)
    
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
        elif (len(s) == 1) and class2.isfloat(s[0]): 
            e = float(s[0])
            rec = class2.record(e, atoms_list)
            record_list.append(rec)
        elif (len(s) == 4): 
            atom_symbol = s[0]
            x = float(s[1])
            y = float(s[2])
            z = float(s[3])
            atoms_list.append(class2.AtomCoordinates(Atoms[j], x, y, z))
            j += 1
        i += 1
    
    # split array if too big
    NpArrayCapacity = 2*1e+8
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
    f = open(F_HarmonicFeatures, "w")
    for i in range(0, len(size_list_str), 1):
        fin = open(size_list_str[i], "r")
        S = fin.readlines()
        f.writelines(S)
        fin.close()
    f.close()
    for i in range(0, len(size_list_str), 1):
        try:
            os.remove(size_list_str[i]) # erase old files if exist
        except:
            pass
            
    
    print("DONE")
    

    
    
