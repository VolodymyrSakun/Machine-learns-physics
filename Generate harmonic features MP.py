import os
import multiprocessing as mp # Pool, freez_support
import numpy as np
import pandas as pd
import pickle 
from structure import class2
from structure import spherical
from structure import library2
from scipy.special import sph_harm
from joblib import Parallel, delayed

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
# FeaturesAll[0].Harmonic
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
    energy = np.zeros(shape=(last - first, 1), dtype=float)
    for i in range(first, last, 1):
        energy[i-first, 0] = record_list[i].e # energy
# calculating and storing distances    
    features_array = np.zeros(shape=(last-first, len(FeaturesAll)), dtype=float) 
    for j in range(0, len(FeaturesAll), 1):
        for i in range(first, last, 1):
            if FeaturesAll[j].nDistances == 1:
                atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                d = np.sqrt((record_list[i].atoms[atom1_index].x - record_list[i].atoms[atom2_index].x)**2 +\
                            (record_list[i].atoms[atom1_index].y - record_list[i].atoms[atom2_index].y)**2 +\
                            (record_list[i].atoms[atom1_index].z - record_list[i].atoms[atom2_index].z)**2)            
                r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
                if FeaturesAll[j].Harmonic is not None:
                    center_index = FeaturesAll[j].Harmonic.Center.Index
                    external_index = FeaturesAll[j].Harmonic.Atom.Index
                    new_origin = spherical.Point(record_list[i].atoms[center_index].x, record_list[i].atoms[center_index].y, record_list[i].atoms[center_index].z)
                    external_atom = spherical.Point(record_list[i].atoms[external_index].x, record_list[i].atoms[external_index].y, record_list[i].atoms[external_index].z)
                    H_list = []
                    for k in range(0, len(Atoms), 1):
                        if FeaturesAll[j].Harmonic.Center.MolecularIndex == Atoms[k].MolecularIndex:
                            H_list.append(Atoms[k])
                    H1_index = H_list[0].Index
                    H2_index = H_list[1].Index
                    H1 = spherical.Point(record_list[i].atoms[H1_index].x, record_list[i].atoms[H1_index].y, record_list[i].atoms[H1_index].z)
                    H2 = spherical.Point(record_list[i].atoms[H2_index].x, record_list[i].atoms[H2_index].y, record_list[i].atoms[H2_index].z)
                    theta, phi = spherical.get_angles(new_origin, H1, H2, external_atom)
                    s = sph_harm(FeaturesAll[j].Harmonic.Order, FeaturesAll[j].Harmonic.Degree, theta, phi).real
                    r = r * s
            if FeaturesAll[j].nDistances == 2:
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
            features_array[i-first, j] = r # store to array
    NofFeaturesReduced = len(FeaturesReduced)
    NofFeatures = len(FeaturesAll)
# sum features with equal FeType
    last_rec = 0
    features_array_reduced = np.zeros(shape=(last-first, NofFeaturesReduced), dtype=float)
    for k in range(0, NofFeaturesReduced, 1):
        for j in range(0, NofFeatures, 1):
            if (FeaturesAll[j].FeType == FeaturesReduced[k].FeType) and (FeaturesAll[j].Harmonic is None):
                features_array_reduced[:, k] += features_array[:, j]
                last_rec = k
    last_rec += 1
    for j in range(0, NofFeatures, 1):
        if (FeaturesAll[j].Harmonic is not None):
            features_array_reduced[:, last_rec] = features_array[:, j]
            last_rec += 1
                
# removing NaN from dataset
    mask = ~np.any(np.isnan(features_array_reduced), axis=1)
    features_array_reduced = features_array_reduced[mask]
    energy = energy[mask]
# save reduced features and energy into file
    Table = pd.DataFrame(features_array_reduced, dtype=float)
    Table['energy'] = energy
    f = open(F_out_features, 'w')
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
    #F_data = 'short 3 mol.x'
    F_data = 'datafile1 from github gaussian process.x' # file with coordinates
    #F_data = 'datafile2.x'
    #F_data = 'datafile3 2 water molecules.x'
    #F_data = 'datafile4 3 water molecules small.x'
    #F_data = 'datafile5 3 water molecules big.x'
    FileName = 'FeaturesReduced List.xlsx'
    F_HarmonicFeatures = 'Harmonic Features.csv' # output csv file with combined features and energy
    F_HarmonicFeaturesAll = 'HarmonicFeaturesAll.dat' # output data structure which contains all features
    F_HarmonicFeaturesReduced = 'HarmonicFeaturesReduced.dat' # output data structure which contains combined features
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
                if len(PowersStr) > 1:
                    for j in range(1, len(PowersStr), 1):
                        PowersDefault.append(int(PowersStr[j]))
                        
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
    
    HarmonicDescriptionFirst = 0 #if they have the same value use only default
    HarmonicDescriptionLast = 0
    Hamonics_include = False
    for i in range(0, len(lines), 1):
        x = lines[i]
        if (x.find('&HarmonicsDescription') != -1):
            HarmonicDescriptionFirst = i + 1
        if (x.find('&endHarmonicsDescription') != -1):   
            HarmonicDescriptionLast = i
    if HarmonicDescriptionFirst != HarmonicDescriptionLast:
        for i in range(HarmonicDescriptionFirst, HarmonicDescriptionLast, 1):
            x = lines[i]
            if (x.find('Order') != -1):
                Str_order = x.split(',', -1)
                for k in range(1, len(Str_order), 1):
                    Str_order[k] = int(Str_order[k])
            if (x.find('Degree') != -1):
                Str_degree = x.split(',', -1)   
                for k in range(1, len(Str_degree), 1):
                    Str_degree[k] = int(Str_degree[k])   
            if (x.find('Center') != -1):
                Str_center = x.split(',', -1)   
        Order_list = Str_order[1:]
        Degree_list = Str_degree[1:]
        Center = Str_center[1]
        Hamonics_include = True
        
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
                Atoms.append(class2.Atom(Str[0], j, int(Str[1]), int(Str[2])))
                j += 1
                i += 1
    
    # determine number of atom types from list
    nAtTypes = 0        
    for i in range(0, len(Atoms), 1):
        if (Atoms[i].AtType > nAtTypes):
            nAtTypes = Atoms[i].AtType
    nAtTypes += 1        
    
    # replace atom symbols with numbers
    for i in range(0, len(DList), 1):
        for j in range(0, len(Atoms), 1):
            if DList[i][0] == Atoms[j].Symbol:
                DList[i][0] = Atoms[j].AtType 
            if DList[i][1] == Atoms[j].Symbol:
                DList[i][1] = Atoms[j].AtType    
    
    nAtoms = len(Atoms)
    Distances = [] # create list of distances from list of atoms
    for i in range(0, nAtoms, 1):
        for j in range(i+1, nAtoms, 1):
            Distances.append(class2.Distance(Atoms[i], Atoms[j]))
    
    # insert DiType in the list with distance descriptions
    for j in range(0, len(DList), 1):
        Inserted = False
        for i in range(0, len(Distances), 1):
            if Distances[i].isIntermolecular == DList[j][2]:
                if ((Distances[i].Atom1.AtType == DList[j][0]) and (Distances[i].Atom2.AtType == DList[j][1])) or ((Distances[i].Atom1.AtType == DList[j][1]) and (Distances[i].Atom2.AtType == DList[j][0])):
                    if not Inserted:
                        DList[j].insert(0, Distances[i].DiType)
                        Inserted = True
    
    # create list of DiTypes
    Types_list = []
    for i in range(0, len(DList), 1):
        Types_list.append(DList[i][0])
          
    # add default powers to DList if necessairy
    for i in range(0, len(Distances), 1):
        if Distances[i].DiType not in Types_list:
            DList.append(list())
            DList[-1].append(Distances[i].DiType)
            DList[-1].append(Distances[i].Atom1.AtType)
            DList[-1].append(Distances[i].Atom2.AtType)
            if Distances[i].isIntermolecular:
                DList[-1].append(1)
            else:
                DList[-1].append(0)
            DList[-1] = DList[-1] + PowersDefault 
            Types_list.append(Distances[i].DiType)
            
    DtP_list = []
    for i in range(0, len(Distances), 1):
        Powers = []
        idx = Types_list.index(Distances[i].DiType)
        Powers = list(DList[idx][4:])
        for j in Powers:
            DtP_list.append(class2.Distance_to_Power(Distances[i], j))
    
    FeaturesAll = [] # list of all features
    # make list of features with only one distance
    for i in range(0, len(DtP_list), 1):
        FeaturesAll.append(class2.Feature(1, DtP_list[i], DtP2=None, Harmonic=None))
        
# include r*r only O-H intermolecular        
    if max_number_of_distances_in_feature == 2:
        for i in range(0, len(DtP_list), 1):
            for j in range(i+1, len(DtP_list), 1):
                if DtP_list[i].Distance.isIntermolecular and  (DtP_list[i].Distance.Atom1.AtType != DtP_list[i].Distance.Atom2.AtType): # only O-H intermolecular first distance
                    if DtP_list[j].Distance.isIntermolecular and  (DtP_list[j].Distance.Atom1.AtType != DtP_list[j].Distance.Atom2.AtType): # only O-H intermolecular second distance
                        if DtP_list[i].Power != DtP_list[j].Power: # same powers are not included
                            FeaturesAll.append(class2.Feature(2, DtP_list[i], DtP2=DtP_list[j], Harmonic=None))
                
    if Hamonics_include:
        for i in range(0, len(DtP_list), 1):
            for j in range(0, len(Atoms), 1):
                if Atoms[j].Symbol == Center: # Atoms[j] - Center
                    for k in range(0, len(Atoms), 1):
                        if Atoms[k].MolecularIndex != Atoms[j].MolecularIndex: # Atoms[k] - external atom
                            for n in Degree_list:
                                for m in range(0, n+1, 1):
                                    h = class2.Harmonic(m, n, Atoms[j], Atoms[k])
                                    FeaturesAll.append(class2.Feature(1, DtP_list[i], Harmonic=h))

 
#    if max_number_of_distances_in_feature == 2:
#        for i in range(0, len(DtP_list), 1):
#            for j in range(i+1, len(DtP_list), 1):
#    # check if distances with different powers have same DiType
#                if DtP_list[i].Distance.DiType != DtP_list[j].Distance.DiType:
#                    FeaturesAll.append(class2.Feature(2, DtP_list[i], DtP2=DtP_list[j], Harmonic=None))


 
        
        
    # Make list of reduced features
    FeaturesReduced = []
    FeType_list = []
    for i in range(0, len(FeaturesAll), 1):
        if (FeaturesAll[i].FeType not in FeType_list) or (FeaturesAll[i].Harmonic is not None):
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
    
    Size = len(record_list) # N of observations
    BufSize = 20000 # depends on memory availabe
    size_list = []
    size_list_str = []
    nCPU = mp.cpu_count()
    print('Start Multiprocessing with ', nCPU, ' cores')
    pool = mp.Pool(processes=nCPU)
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
    j = 0
    ran = list(range(0, len(size_list_str), 1))
    if __name__=="__main__":
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
            
        library2.StoreFeaturesDescriprion(FileName, FeaturesAll, FeaturesReduced)
        print("DONE")

    
        
        
        