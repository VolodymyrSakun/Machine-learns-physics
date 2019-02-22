import numpy as np
import pandas as pd
import sklearn.metrics as skm
import os
import copy
import sys
from time import time
import shutil
import matplotlib.pyplot as plt
from project1 import IOfunctions
import random
from project1 import structure
from project1 import genetic
from project1 import regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from matplotlib.mlab import griddata


RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"
HARTREE_TO_KJMOL = 2625.49963865

def Print(*arg, color=RESET):
    s = ''.join(str(arg[i]) for i in range(0, len(arg), 1))
    sys.stdout.write(color)
    print(s)
    sys.stdout.write(RESET)
    return

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False
    
# redirect "print" functin to file
def RedirectPrintToFile(ProcedureFileName):
    f_old = sys.stdout
    f = open(ProcedureFileName, 'w')
    sys.stdout = f
    return f, f_old

# redirect "print" functin to console
def RedirectPrintToConsole(f, f_old):
    sys.stdout = f_old
    f.close()
    return

def argmaxabs(A):
    tmp = abs(A[0])
    idx = 0
    for i in range(1, len(A), 1):
        if abs(A[i]) > tmp:
            tmp = abs(A[i])
            idx = i
    return idx

def argminabs(A):
    tmp = abs(A[0])
    idx = 0
    for i in range(1, len(A), 1):
        if abs(A[i]) < tmp:
            tmp = abs(A[i])
            idx = i
    return idx

def Scaler_L2(X):
    X_new = np.zeros(shape=(X.shape[0], X.shape[1]), dtype = float)
    col = np.zeros(shape=(X.shape[0]), dtype = float)
    m = X.shape[0]# n of rows
    n = X.shape[1]# n of columns
    for j in range(0, n, 1):
        col[:] = X[:, j]
        Denom = np.sqrt(np.dot(np.transpose(col), col, out=None))
        for i in range(0, m, 1):
            X_new[i, j] = X[i, j] / Denom
    return X_new

def InInterval(n, List):
    if List is None:
        return -10
    for i in range(0, len(List), 1):
        if (n >= List[i][0]) and (n <= List[i][1]):
            return i
    return -10 # not in region

# Calculate variance inflation factor for all features
def CalculateVif(X):
# returns dataframe with variance inflation factors
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif = vif.round(1)
    return vif

def Swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b

# first list in *args governs sorting, others depend on first list 
def Sort(*args, direction='Lo-Hi'):
    idx = copy.deepcopy(args)
    n = len(idx[0])
    if direction == 'Lo-Hi':
        Dir = 1
    else:
        Dir = -1
# bubble sort according to direction
    while n != 0:
        newn = 0
        for i in range(1, n, 1):
            difference = (idx[0][i-1] - idx[0][i])*Dir
            if difference > 0:
                newn = i
                for j in range(0, len(idx), 1):
                    idx[j][i], idx[j][i-1] = Swap(idx[j][i], idx[j][i-1])
        n = newn
    return idx

def get_energy(Coef_tuple, FeaturesAll, FeaturesReduced, record_list, Record, nVariables):
    tuple_number = 0
    Found = False
    for i in range(0, len(Coef_tuple), 1):
        if Coef_tuple[i][0] == nVariables:
            tuple_number = i   
            Found = True
    if not Found:
        print('Reord with ', nVariables, ' variables does not exist')
        return
    features_set = Coef_tuple[tuple_number]
    k = Record # number of record
    E = 0
    for i in range(0, len(features_set[1]), 1): # for each nonzero coefficient
        current_feature_idx = features_set[1][i]
        current_feature_type = FeaturesReduced[current_feature_idx].FeType
        variable = 0
        for j in range(0, len(FeaturesAll), 1): # for each combined feature
            if FeaturesAll[j].FeType == current_feature_type:
                if FeaturesAll[j].nDistances == 1:
                    atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                    atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                    d = np.sqrt((record_list[k].atoms[atom1_index].x - record_list[k].atoms[atom2_index].x)**2 +\
                                (record_list[k].atoms[atom1_index].y - record_list[k].atoms[atom2_index].y)**2 +\
                                (record_list[k].atoms[atom1_index].z - record_list[k].atoms[atom2_index].z)**2)            
                    r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
                if FeaturesAll[j].nDistances == 2:
                    atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                    atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                    atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                    atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                    d1 = np.sqrt((record_list[k].atoms[atom11_index].x - record_list[k].atoms[atom12_index].x)**2 +\
                                 (record_list[k].atoms[atom11_index].y - record_list[k].atoms[atom12_index].y)**2 +\
                                 (record_list[k].atoms[atom11_index].z - record_list[k].atoms[atom12_index].z)**2)            
                    r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                    d2 = np.sqrt((record_list[k].atoms[atom21_index].x - record_list[k].atoms[atom22_index].x)**2 +\
                                 (record_list[k].atoms[atom21_index].y - record_list[k].atoms[atom22_index].y)**2 +\
                                 (record_list[k].atoms[atom21_index].z - record_list[k].atoms[atom22_index].z)**2)            
                    r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power
                    r = r1 * r2      
                variable += r
        E += variable * features_set[2][i] # combined features by coefficient
    return E

def replace_numbers(l1, l2, source_data):
    l1 = list(l1)
    l2 = list(l2) # those numbers will be replaced by numbers from l1
    if type(source_data) is not list: # replace single number
        try:
            idx = l2.index(source_data)
        except:
            return False # source data is not in mapping list
        dest = l1[idx]       
        return dest
    dest_data = copy.deepcopy(source_data)
    for i in range(0, len(source_data), 1):
        if (type(source_data[i]) is list): # 2D array
            for j in range(0, len(source_data[i]), 1):
                num_source = source_data[i][j]
                try:
                    idx = l2.index(num_source)
                except:
                    return False # source data is not in mapping list
                num_dest = l1[idx]
                dest_data[i][j] = num_dest
        else: # 1D array
            try:
                idx = l2.index(source_data[i])
            except:
                return False # source data is not in mapping list
            num_dest = l1[idx]
            dest_data[i] = num_dest
    return dest_data
    
def CreateGrid(GridStart, GridEnd, GridSpacing):
    Grid = [] 
    i = GridStart
    while round(i, 2) <= round((GridEnd-GridSpacing),2):
        Grid.append((round(i, 2), round(i+GridSpacing, 2)))
        i += GridSpacing
    return Grid

def FilterData(Files, Data):

    MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=Files['System descriptor'])
    if Data['Random state'] is not None:
        random.seed(Data['Random state'])
    else:
        random.seed()    
    GridTrain = CreateGrid(Data['Grid start'], Data['Grid end'], Data['Grid spacing']) # trained region bins
    GridTest = CreateGrid(Data['Grid start'], Data['Grid end'], Data['Grid spacing']) # non-trained and trained region bins
    N = np.zeros(shape=(len(GridTest)), dtype=int) # number of points in each grid  
    NTrain = list(np.zeros(shape=(len(GridTrain)), dtype=int)) # count train points
    NTest = list(np.zeros(shape=(len(GridTest)), dtype=int)) # count test points   
    setType = IOfunctions.getSetType(Files['Set'])  
    print(setType)
    if setType == 'Old':
        Records = IOfunctions.ReadRecordMoleculesOld(Files['Set'], MoleculePrototypes) # Read records
    else:
        Records = IOfunctions.ReadRecordMoleculesNew(Files['Set'], MoleculePrototypes) # Read records        
    DMin = 1000 # will be shortest distance
    DMax = 0 # will be most distant distance
    nMolecules = Records[0].nMolecules
# Determine DMin, DMax and fill N
    for record in Records:
        if nMolecules == 2:
            if record.R_Average > DMax:
                DMax = record.R_Average
            if record.R_Average < DMin:
                DMin = record.R_Average
            if record.R_Average >= Data['Grid start']:
                j = int((record.R_Average - Data['Grid start']) / Data['Grid spacing'])
                if j < len(N):
                    N[j] += 1
        else:
            if record.R_CenterOfMass_Average > DMax:
                DMax = record.R_CenterOfMass_Average
            if record.R_CenterOfMass_Average < DMin:
                DMin = record.R_CenterOfMass_Average
            if record.R_CenterOfMass_Average >= Data['Grid start']:
                j = int((record.R_CenterOfMass_Average - Data['Grid start']) / Data['Grid spacing'])
                if j < len(N):
                    N[j] += 1
# Estimate number of points per grid
    n = np.asarray(N.nonzero()).reshape(-1) # indices with nonzero records nonzero   
    nGrids = int(len(n) * Data['Confidence interval'])
    N_list = list(N)
    N_Reduced = []
    while len(N_Reduced) < nGrids:
        i = np.argmax(N_list)
        N_Reduced.append(N_list[i])
        del(N_list[i])
    nPointsGrid = N_Reduced[-1]
    nTestPointsGrid = int(nPointsGrid * Data['Test fraction'])
    nTotalTrainPointsGrid = nPointsGrid - nTestPointsGrid  
    nTrainPointsGrid = int(nTotalTrainPointsGrid * Data['Train fraction'])
    N_list = list(N)    
    i = 0
    while i < len(N_list): # remove regions where there are not enough points
# trained region        
        if (InInterval(GridTest[i][0], Data['Train intervals']) != -10)\
            and (InInterval(GridTest[i][1], Data['Train intervals']) != -10):
            if N_list[i] < nPointsGrid: # not enough points for training and test, discard
                del(N_list[i])
                del(NTrain[i])
                del(NTest[i])
                del(GridTrain[i])
                del(GridTest[i])   
            else:
                i += 1
        else: # test region            
            if N_list[i] < nTestPointsGrid: # not enough test points
                del(N_list[i])
                del(NTrain[i])
                del(NTest[i])
                del(GridTrain[i])
                del(GridTest[i]) 
            else:
                i += 1
    i = 0 # remove remaining train grid that not in training region
    while i < len(GridTrain):
        if (InInterval(GridTrain[i][0], Data['Train intervals']) != -10)\
            and (InInterval(GridTrain[i][1], Data['Train intervals']) != -10):
            i += 1
            continue
        else:
            del(GridTrain[i])
            del(NTrain[i])
# proceed records                               
    RecordsTrain = []
    RecordsTest = []
    while len(Records) > 0:
        r = random.randrange(0, len(Records), 1)  
        record = copy.deepcopy(Records[r])
        if nMolecules == 2:
            d = record.R_Average
        else:
            d = record.R_CenterOfMass_Average
        j = InInterval(d, GridTrain) 
        if j != -10: # in training region?
            if NTrain[j] < nTrainPointsGrid: # append to training set
                NTrain[j] += 1
                RecordsTrain.append(record)
            else:  # if it is full, append to test set
                j = InInterval(d, GridTest) # which interval?
                if j != -10:
                    if NTest[j] < nTestPointsGrid: 
                        NTest[j] += 1
                        RecordsTest.append(record)              
        else: # not training region
            j = InInterval(d, GridTest) # which interval?
            if j != -10:
                if NTest[j] < nTestPointsGrid: # append to test set only
                    NTest[j] += 1
                    RecordsTest.append(record)
        del(Records[r]) 

    IOfunctions.store_records(Files['Training set'], RecordsTrain) # store trained set
    IOfunctions.store_records(Files['Test set'], RecordsTest) # store test set
    IOfunctions.store_average_distances(Files['COM train'], RecordsTrain)
    IOfunctions.store_average_distances(Files['COM test'], RecordsTest)
    TestIntervals = [] # Define test regions
    if Data['Train intervals'][0][0] != 0:
        TestIntervals.append((0, Data['Train intervals'][0][0]))
    for i in range(0, (len(Data['Train intervals'])-1), 1):
        if Data['Train intervals'][i][1] != Data['Train intervals'][i+1][0]:
            TestIntervals.append((Data['Train intervals'][i][1], Data['Train intervals'][i+1][0]))
    if Data['Train intervals'][-1][1] < GridTest[-1][1]:
        TestIntervals.append((Data['Train intervals'][-1][1], GridTest[-1][1]))
    
    results = {'Initial dataset': Files['Set'],'Number of molecules per record': nMolecules,\
               'Train Intervals': Data['Train intervals'],'Test Intervals': TestIntervals,\
               'Train records number': len(RecordsTrain),'Train Grid': GridTrain,\
               'Test Grid': GridTest, 'Test records number': len(RecordsTest),\
               'Molecule prototypes': MoleculePrototypes, 'Max points per grid': nPointsGrid,\
               'Train points per grid': nTrainPointsGrid, 'Train Fraction Used': Data['Train fraction'],\
               'Test points per grid': nTestPointsGrid, 'Confidence Interval used': Data['Confidence interval'],\
               'Training Set': Files['Training set'], 'Test Set': Files['Test set'],\
               'COM Train': Files['COM train'], 'COM Test': Files['COM test']}
            
    return results
    
def ReadData(F_data, Atoms):
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
    record_list = []
    while i < len(data):
        s = data[i].split() # line of text separated in list
        if len(s) == 0: # empty line
            i += 1
            continue
    # record for energy value
        elif (len(s) == 1) and isfloat(s[0]): 
            record_list.append(structure.RecordAtoms(float(s[0]), atoms_list))
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
    return record_list

def GenerateFeatures(Files, Forecast=False): 

    def CreateDtPList(Distances, Description):
    # make list of distances raised to corresponding power    
        Powers = []
        
        for distance in Distances:
            Found = False
            for description in Description:
                if distance.isIntermolecular == description[2]:
                    if ((distance.Atom1.Symbol == description[0]) and (distance.Atom2.Symbol == description[1])) or\
                        ((distance.Atom1.Symbol == description[1]) and (distance.Atom2.Symbol == description[0])):
                        Powers.append(description[3])  
                        Found = True
                        break
            if not Found:
                print('Missing distances in SystemDescriptor')
                print(distance.Atom1.Symbol, '-', distance.Atom2.Symbol)                
        DtP_list = []
        for i in range(0, nDistances, 1):
            for power in Powers[i]:
                DtP_list.append(structure.Distance_to_Power(Distances[i], power, PowerDigits=2))
        return DtP_list
    
    def CreateSingleFeaturesAll(DtP_Single_list, FeType='Linear', nDistances=1, nConstants=1):    
        FeaturesAll = [] # single linear features list
        for DtP in DtP_Single_list:
            FeaturesAll.append(structure.Feature(FeType=FeType, DtP1=DtP, DtP2=None,\
                nDistances=nDistances, nConstants=nConstants)) 
        return FeaturesAll
    
    def CreateDoubleFeaturesAll(DtP_Double_list, FeType='Linear',\
            nConstants=1, IncludeOnlySamePower=False, IncludeSameType=True):    
        FeaturesAll = [] # double linear features list
        for i in range(0, len(DtP_Double_list), 1):
            for j in range(i+1, len(DtP_Double_list), 1):
                if DtP_Double_list[i].Power > DtP_Double_list[j].Power:
                    continue # skip duplicates
                if IncludeOnlySamePower: # pairs with same power will only be added               
                    if DtP_Double_list[i].Power != DtP_Double_list[j].Power:
                        continue # include only products with same powers                 
                if not IncludeSameType: # if this parameter is false
                    if DtP_Double_list[i].Distance.DiType == DtP_Double_list[j].Distance.DiType: 
                        continue # skip if distances of the same type
                FeaturesAll.append(structure.Feature(FeType=FeType,\
                    DtP1=DtP_Double_list[i], DtP2=DtP_Double_list[j],\
                    nDistances=2, nConstants=nConstants))
        return FeaturesAll

    def CreateTripleFeaturesAll(DtP_Triple_list, FeType='Linear',\
            nConstants=1, IncludeOnlySamePower=False, IncludeSameType=True):    
        FeaturesAll = [] # double linear features list
        for i in range(0, len(DtP_Triple_list), 1):
            for j in range(i+1, len(DtP_Triple_list), 1):
                for k in range(j+1, len(DtP_Triple_list), 1):
                    if (DtP_Triple_list[i].Power > DtP_Triple_list[j].Power) or\
                        DtP_Triple_list[j].Power > DtP_Triple_list[k].Power:
                        continue # skip duplicates
                    if IncludeOnlySamePower: # pairs with same power will only be added               
                        if not ((DtP_Triple_list[i].Power == DtP_Triple_list[j].Power) and\
                            (DtP_Triple_list[j].Power == DtP_Triple_list[k].Power)):
                            continue # include only products with same powers                 
                    if not IncludeSameType: # if this parameter is false
                        if (DtP_Triple_list[i].Distance.DiType == DtP_Triple_list[j].Distance.DiType) and\
                            (DtP_Triple_list[j].Distance.DiType == DtP_Triple_list[k].Distance.DiType): 
                            continue # skip if distances of the same type
                    FeaturesAll.append(structure.Feature(FeType=FeType,\
                        DtP1=DtP_Triple_list[i], DtP2=DtP_Triple_list[j], DtP3=DtP_Triple_list[k],\
                        nDistances=3, nConstants=nConstants))
        return FeaturesAll
    
    def CreateFeaturesReduced(FeaturesAll):
    # Make list of reduced features
        FeaturesReduced = []
        FeType_list = []
        for i in range(0, len(FeaturesAll), 1):
            if (FeaturesAll[i].FeType not in FeType_list):
                FeType_list.append(FeaturesAll[i].FeType)
                feature = copy.deepcopy(FeaturesAll[i])
                FeaturesReduced.append(feature)
    # store global indices for each reduced feature
        for i in range(0, len(FeaturesReduced), 1):
            for j in range(0, len(FeaturesAll), 1):
                if (FeaturesAll[j].FeType == FeaturesReduced[i].FeType):
                    if j not in FeaturesReduced[i].idx:
                        FeaturesReduced[i].idx.append(j)
        return FeaturesReduced
        
    F_SystemDescriptor = Files['System descriptor']
    Prototypes = IOfunctions.ReadMoleculeDescription(F_SystemDescriptor, keyword='MoleculeDescription')
    Atoms, Molecules = IOfunctions.ReadSystemDescription(F_SystemDescriptor, 'SYSTEM')        
    LinearSingle = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='LinearSingle')
    LinearDouble = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='LinearDouble')
    LinearTriple = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='LinearTriple')
    ExpSingle = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='ExpSingle')
    ExpDouble = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='ExpDouble')
    GaussianSingle = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='GaussianSingle')
    
    nAtoms = len(Atoms)
    nMolecules = len(Molecules)
    AtomTypesList = []
    for i in Atoms: # create atom types list
        if i.AtType not in AtomTypesList:
            AtomTypesList.append(i.AtType)
    nAtomTypes = len(AtomTypesList)
    Distances = [] # create list of distances from list of atoms
    for i in range(0, nAtoms, 1):
        for j in range(i+1, nAtoms, 1):
            Distances.append(structure.Distance(Atoms[i], Atoms[j]))
    DiTypeList = [] # create distances types list
    for i in Distances:
        if i.DiType not in DiTypeList:
            DiTypeList.append(i.DiType)
    nDistances = len(Distances)
    nDiTypes = len(DiTypeList)
    MoleculeNames = []
    for i in Prototypes: # create list of molecule names
        MoleculeNames.append(i.Name)
    for i in Molecules: # update data in Molecules from Prototype
        if i.Name in MoleculeNames:
            idx = MoleculeNames.index(i.Name)
            i.Mass = Prototypes[idx].Mass
            for j in range(0, len(i.Atoms), 1):
                i.Atoms[j].Mass = Prototypes[idx].Atoms[j].Mass
                i.Atoms[j].Radius = Prototypes[idx].Atoms[j].Radius
                i.Atoms[j].Bonds = Prototypes[idx].Atoms[j].Bonds
                i.Atoms[j].Bonds = replace_numbers(i.AtomIndex, Prototypes[idx].AtomIndex, i.Atoms[j].Bonds)
    for i in Molecules:
        i._refresh() # update data from prototype
    # store required data in system object    
    System = structure.System(Atoms=Atoms, Molecules=Molecules, Prototypes=Prototypes,\
        nAtoms=nAtoms, nAtomTypes=nAtomTypes, nMolecules=nMolecules, Distances=Distances,\
        nDistances=nDistances, nDiTypes=nDiTypes)
    
    if not Forecast:
        records_train_list = IOfunctions.ReadRecordAtoms(Files['Training set'], Atoms)
        records_test_list = IOfunctions.ReadRecordAtoms(Files['Test set'], Atoms)
        IOfunctions.StoreEnergy(Files['Response Train'], records_train_list)
        IOfunctions.StoreEnergy(Files['Response Test'], records_test_list)
    else:
        records_forecast_list = IOfunctions.ReadRecordAtoms(Files['Forecast set'], Atoms)
        IOfunctions.StoreEnergy(Files['Response Forecast'], records_forecast_list)        
        
    if LinearSingle['Include']:
        DtP_LinearSingle_list = CreateDtPList(Distances, LinearSingle['Description'])
        FeaturesLinearSingleAll = CreateSingleFeaturesAll(DtP_LinearSingle_list,\
            FeType='Linear', nDistances=1, nConstants=1)
        FeaturesLinearSingleReduced = CreateFeaturesReduced(FeaturesLinearSingleAll)
        if not Forecast:        
            IOfunctions.StoreLinearFeatures(Files['Linear Single Train'], FeaturesLinearSingleAll,\
                FeaturesLinearSingleReduced, records_train_list, Atoms)
            IOfunctions.StoreLinearFeatures(Files['Linear Single Test'], FeaturesLinearSingleAll,\
                FeaturesLinearSingleReduced, records_test_list, Atoms)
        else:
            IOfunctions.StoreLinearFeatures(Files['Linear Single Forecast'], FeaturesLinearSingleAll,\
                FeaturesLinearSingleReduced, records_forecast_list, Atoms)            
    else:
        FeaturesLinearSingleAll = None
        FeaturesLinearSingleReduced = None
        
    if LinearDouble['Include']:
        DtP_LinearDouble_list = CreateDtPList(Distances, LinearDouble['Description'])
        FeaturesLinearDoubleAll = CreateDoubleFeaturesAll(DtP_LinearDouble_list,\
            FeType='Linear', IncludeOnlySamePower=LinearDouble['IncludeOnlySamePower'],\
            IncludeSameType=LinearDouble['IncludeSameType'], nConstants=1)
        FeaturesLinearDoubleReduced = CreateFeaturesReduced(FeaturesLinearDoubleAll)
        if not Forecast:  
            IOfunctions.StoreLinearFeatures(Files['Linear Double Train'], FeaturesLinearDoubleAll,\
                FeaturesLinearDoubleReduced, records_train_list, Atoms)
            IOfunctions.StoreLinearFeatures(Files['Linear Double Test'], FeaturesLinearDoubleAll,\
                FeaturesLinearDoubleReduced, records_test_list, Atoms)    
        else:
            IOfunctions.StoreLinearFeatures(Files['Linear Double Forecast'], FeaturesLinearDoubleAll,\
                FeaturesLinearDoubleReduced, records_forecast_list, Atoms)             
    else:
        FeaturesLinearDoubleAll = None
        FeaturesLinearDoubleReduced = None

    if LinearTriple['Include']:
        DtP_LinearTriple_list = CreateDtPList(Distances, LinearTriple['Description'])
        FeaturesLinearTripleAll = CreateTripleFeaturesAll(DtP_LinearTriple_list,\
            FeType='Linear', IncludeOnlySamePower=LinearTriple['IncludeOnlySamePower'],\
            IncludeSameType=LinearTriple['IncludeSameType'], nConstants=1)
        FeaturesLinearTripleReduced = CreateFeaturesReduced(FeaturesLinearTripleAll)
        if not Forecast:
            IOfunctions.StoreLinearFeatures(Files['Linear Triple Train'], FeaturesLinearTripleAll,\
                FeaturesLinearTripleReduced, records_train_list, Atoms)
            IOfunctions.StoreLinearFeatures(Files['Linear Triple Test'], FeaturesLinearTripleAll,\
                FeaturesLinearTripleReduced, records_test_list, Atoms)    
        else:
            IOfunctions.StoreLinearFeatures(Files['Linear Triple Forecast'], FeaturesLinearTripleAll,\
                FeaturesLinearTripleReduced, records_forecast_list, Atoms)              
    else:
        FeaturesLinearTripleAll = None
        FeaturesLinearTripleReduced = None
                
    if ExpSingle['Include']:
        DtP_ExpSingle_list = CreateDtPList(Distances, ExpSingle['Description'])
        FeaturesExpSingleAll = CreateSingleFeaturesAll(DtP_ExpSingle_list,\
            FeType='Exp', nDistances=1, nConstants=2)
        if not Forecast:     
            IOfunctions.StoreExpSingleFeatures(Files['Exp Single Train D'],\
                Files['Exp Single Train D^n'], FeaturesExpSingleAll, records_train_list, Atoms)
            IOfunctions.StoreExpSingleFeatures(Files['Exp Single Test D'],\
                Files['Exp Single Test D^n'], FeaturesExpSingleAll, records_test_list, Atoms)
        else:
            IOfunctions.StoreExpSingleFeatures(Files['Exp Single Forecast D'],\
                Files['Exp Single Forecast D^n'], FeaturesExpSingleAll, records_forecast_list, Atoms)            
    else:
        FeaturesExpSingleAll = None
                
    if ExpDouble['Include']:    
        DtP_ExpDouble_list = CreateDtPList(Distances, ExpDouble['Description'])
        FeaturesExpDoubleAll = CreateDoubleFeaturesAll(DtP_ExpDouble_list, FeType='Exp',\
            nConstants=3, IncludeOnlySamePower=ExpDouble['IncludeOnlySamePower'],\
            IncludeSameType=ExpDouble['IncludeSameType'])
        if not Forecast: 
            IOfunctions.StoreExpDoubleFeatures(Files['Exp Double Train D1'],\
                Files['Exp Double Train D2'], Files['Exp Double Train D1^mD2^n'],\
                FeaturesExpDoubleAll, records_train_list, Atoms)
            IOfunctions.StoreExpDoubleFeatures(Files['Exp Double Test D1'],\
                Files['Exp Double Test D2'], Files['Exp Double Test D1^mD2^n'],\
                FeaturesExpDoubleAll, records_test_list, Atoms)
        else:
            IOfunctions.StoreExpDoubleFeatures(Files['Exp Double Forecast D1'],\
                Files['Exp Double Forecast D2'], Files['Exp Double Forecast D1^mD2^n'],\
                FeaturesExpDoubleAll, records_forecast_list, Atoms)            
    else:
        FeaturesExpDoubleAll = None

    if GaussianSingle['Include']:
        DtP_GaussianSingle_list = CreateDtPList(Distances, GaussianSingle['Description'])
        FeaturesGaussianSingleAll = CreateSingleFeaturesAll(DtP_GaussianSingle_list,\
            FeType='Gauss', nDistances=1, nConstants=1)
        FeaturesGaussianSingleReduced = CreateFeaturesReduced(FeaturesGaussianSingleAll)
        if not Forecast: 
            IOfunctions.StoreLinearFeatures(Files['Gaussian Train'],\
                FeaturesGaussianSingleAll, FeaturesGaussianSingleReduced, records_train_list, Atoms)
            IOfunctions.StoreLinearFeatures(Files['Gaussian Test'],\
                FeaturesGaussianSingleAll, FeaturesGaussianSingleReduced, records_test_list, Atoms)
        else:
            IOfunctions.StoreLinearFeatures(Files['Gaussian Forecast'],\
                FeaturesGaussianSingleAll, FeaturesGaussianSingleReduced, records_forecast_list, Atoms)            
    else:
        FeaturesGaussianSingleAll = None
        FeaturesGaussianSingleReduced = None
        
    Structure = {'System': System,\
        'FeaturesLinearSingleAll': FeaturesLinearSingleAll,\
        'FeaturesLinearSingleReduced': FeaturesLinearSingleReduced,\
        'FeaturesLinearDoubleAll': FeaturesLinearDoubleAll,\
        'FeaturesLinearDoubleReduced': FeaturesLinearDoubleReduced,\
        'FeaturesLinearTripleAll': FeaturesLinearTripleAll,\
        'FeaturesLinearTripleReduced': FeaturesLinearTripleReduced,\
        'FeaturesExpSingleAll': FeaturesExpSingleAll,\
        'FeaturesExpDoubleAll': FeaturesExpDoubleAll,\
        'FeaturesGaussianAll': FeaturesGaussianSingleAll,\
        'FeaturesGaussianReduced': FeaturesGaussianSingleReduced}
    
    if not Forecast: 
        IOfunctions.StoreStructure(Files['Structure'], Structure) # excel
    
    return Structure  
            
def GetFitGA(FilterDataResults, Files, Data, GenerateFeaturesResults):
    
    featuresDict = IOfunctions.ReadFeatures(Files, GenerateFeaturesResults, Forecast=False) 
    Y_train = featuresDict['Y_train']
    Y_test = featuresDict['Y_test']
    X_LinearSingle_train = featuresDict['X_LinearSingle_train']
    X_LinearSingle_test = featuresDict['X_LinearSingle_test']
    X_ExpSingleD_train = featuresDict['X_ExpSingleD_train']
    X_ExpSingleDn_train = featuresDict['X_ExpSingleDn_train']
    X_ExpSingleD_test = featuresDict['X_ExpSingleD_test']
    X_ExpSingleDn_test = featuresDict['X_ExpSingleDn_test']
    X_Linear_train = featuresDict['X_Linear_train']
    X_Linear_test = featuresDict['X_Linear_test']
    FeaturesLinearAll = featuresDict['FeaturesLinearAll']
    FeaturesLinearReduced = featuresDict['FeaturesLinearReduced']
    
    if Data['Use VIP'] and X_LinearSingle_train is not None:
        if Data['Number of VIP features'] is None:
            if GenerateFeaturesResults.System.nAtoms == 6: # two water molecules system
                nVIP = 5
            if GenerateFeaturesResults.System.nAtoms == 9: # three water molecules system
                nVIP = 3
        else:
            nVIP = Data['Number of VIP features']

        t = time()       
        ga = genetic.GA(PopulationSize=Data['GA population size'], ChromosomeSize=Data['GA chromosome size'],\
            MutationProbability=Data['GA mutation probability'], MutationInterval=Data['GA mutation interval'],\
            EliteFraction=Data['GA elite fraction'], MutationCrossoverFraction=Data['GA mutation crossover fraction'],\
            CrossoverFractionInterval=Data['GA crossover fraction interval'], PrintInterval=Data['GA generations per output'],\
            StopTime=Data['GA stop time'], RandomSeed=Data['Random state'], verbose=Data['GA verbose'],\
            UseCorrelationMutation=Data['GA use correlation for mutation'], MinCorrMutation=Data['GA min correlation for mutation'],\
            UseCorrelationBestFit=Data['A use correlation'], MinCorrBestFit=Data['A min correlation'])   

        if (Data['First algorithm'] == 'ENet'):
            print('Only linear features will be considered')
            Alphas = np.logspace(Data['ENet alpha min single'], Data['ENet alpha max single'], num=Data['ENet number of alphas single'],\
                endpoint=True, base=10.0, dtype=float)
            enet = regression.ENet(L1=Data['ENet ratio single'], nAlphas=None,\
                alphas=Alphas, random_state=Data['Random state'])
            print('Number of features go to elastic net regularisation = ', X_LinearSingle_train.shape[1])
            enet.fit(X_LinearSingle_train, Y_train, VIP_idx=None, Criterion=Data['ENet criterion'], normalize=True,\
                max_iter=Data['ENet max number of iterations single'], tol=0.0001, cv=Data['ENet cv number single'], n_jobs=1, selection='random', verbose=Data['ENet verbose'])
            enet.plot_path(1, F_ENet=Files['ENet path single'], FigSize=Data['Figure size'], FileFormat=Data['Figure file format'])
            idx = enet.idx
            Gene_list = []
            for i in idx:
                Gene_list.append(genetic.GA.Gene(i, Type=0))
            chromosome = ga.Chromosome(Gene_list)
            chromosome.erase_score()
            chromosome.score(x_expD_train=None, x_expDn_train=None, x_lin_train=X_LinearSingle_train,\
                y_train=Y_train, x_expD_test=None, x_expDn_test=None, x_lin_test=X_LinearSingle_train,\
                y_test=Y_test,\
                LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
            if ga.LinearSolver == 'statsmodels':
                chromosome.rank_sort_pValue()
            else:
                chromosome.rank(x_expD=None, x_expDn=None, x_lin=X_LinearSingle_train,\
                    y=Y_train,\
                    LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
                chromosome.sort(order='Most important first')  
                ga.n_lin = X_LinearSingle_train.shape[1]
            t_sec = time() - t
            print("\n", 'Elastic Net worked ', t_sec, 'sec')  
            print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
        if Data['First algorithm'] == 'GA':
            print('Genetic Algorithm for all features')
            ga.fit(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_LinearSingle_train,\
                y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_LinearSingle_test,\
                y_test=Y_test, idx_exp=None, idx_lin=None, VIP_idx_exp=None,\
                VIP_idx_lin=None, CrossoverMethod=Data['GA crossover method'], MutationMethod=Data['GA mutation method'],\
                LinearSolver=Data['LR linear solver'], cond=Data['LR scipy condition'], lapack_driver=Data['LR scipy driver'], nIter=Data['GA max generations'])
            t_sec = time() - t
            print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
            chromosome = ga.BestChromosome        
            print("\n", 'Features left for Backward Elimination and Search Alternative = ',\
                  chromosome.Size)

        while chromosome.Size > 1:
            if chromosome.Size <= ga.ChromosomeSize:
                chromosome = ga.BestFit2(chromosome, x_expD=X_ExpSingleD_train,\
                    x_expDn=X_ExpSingleDn_train, x_lin=X_Linear_train, y=Y_train,\
                    goal=Data['A goal'], epoch=Data['A stop time'], q_max=Data['A max queue'], model=Data['A selection criterion'], verbose=Data['A verbose'])            
                chromosome.score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_LinearSingle_train,\
                    y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_LinearSingle_test,\
                    y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver) 
                chromosome.Origin = 'Best Fit'
                chromosome_copy = copy.deepcopy(chromosome)
                chromosome_copy.print_score()
                ga.DecreasingChromosomes.append(chromosome_copy)
            chromosome = ga.RemoveWorstGene(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train,\
                x_lin=X_LinearSingle_train, y=Y_train, verbose=Data['BE verbose'])
        t_sec = time() - t
        for i in range(0, len(ga.BestFitPath), 1): # get fitness from test set
            ga.BestFitPath[i].score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_LinearSingle_train,\
                    y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_LinearSingle_test,\
                    y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)        

        ga.PlotChromosomes2(Files['GA path RMSE single'], 2, ga.DecreasingChromosomes,\
            XAxis='Nonzero', YAxis='RMSE', Title=None, PlotType='Line',\
            figsize=Data['Figure size'], marker_size=5, line_width=1, FileFormat=Data['Figure file format'])
        ga.PlotChromosomes2(Files['GA path R2 single'], 3, ga.DecreasingChromosomes,\
            XAxis='Nonzero', YAxis='R2', Title=None, PlotType='Line', FileFormat=Data['Figure file format'],\
            figsize=Data['Figure size'], marker_size=5, line_width=1)           
        ga.PlotChromosomes2(Files['BF path RMSE single'], 4, ga.BestFitPath, XAxis='Time',\
            YAxis='RMSE', Title=None, PlotType='Scatter', figsize=Data['Figure size'],\
            marker_size=5, line_width=1, FileFormat=Data['Figure file format'])
    
        print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
        ga.Results_to_xlsx('{} {}'.format(Files['Fit'], 'Single.xlsx'), \
            FeaturesNonlinear=GenerateFeaturesResults['FeaturesExpSingleAll'],\
            FeaturesAll=GenerateFeaturesResults['FeaturesLinearSingleAll'],\
            FeaturesReduced=GenerateFeaturesResults['FeaturesLinearSingleReduced'],\
            X_Linear=X_Linear_train)                              
        for i in ga.DecreasingChromosomes:
            if i.Size == nVIP:
                VIP_idx_exp = i.get_genes_list(Type=1)
                VIP_idx_lin = i.get_genes_list(Type=0)
                break
    else:
        VIP_idx_exp = []
        VIP_idx_lin = []      
# proceed all features 
    Print('{} {} {} {}'.format('VIP linear:', VIP_idx_lin, 'VIP exponential:', VIP_idx_exp), color=RED)
    t = time()    
    ga = genetic.GA(PopulationSize=Data['GA population size'], ChromosomeSize=Data['GA chromosome size'],\
        MutationProbability=Data['GA mutation probability'], MutationInterval=Data['GA mutation interval'],\
        EliteFraction=Data['GA elite fraction'], MutationCrossoverFraction=Data['GA mutation crossover fraction'],\
        CrossoverFractionInterval=Data['GA crossover fraction interval'], PrintInterval=Data['GA generations per output'],\
        StopTime=Data['GA stop time'], RandomSeed=Data['Random state'], verbose=Data['GA verbose'],\
        UseCorrelationMutation=Data['GA use correlation for mutation'], MinCorrMutation=Data['GA min correlation for mutation'],\
        UseCorrelationBestFit=Data['A use correlation'], MinCorrBestFit=Data['A min correlation'])   
       
# linear only for now
    if (Data['First algorithm'] == 'ENet'):
        print('Only linear features will be considered')
        Alphas = np.logspace(Data['ENet alpha min'], Data['ENet alpha max'], num=Data['ENet number of alphas'],\
            endpoint=True, base=10.0, dtype=float)
        enet = regression.ENet(L1=Data['ENet ratio'], nAlphas=None,\
            alphas=Alphas, random_state=Data['Random state'])
        print('Number of features go to elastic net regularisation = ', X_Linear_train.shape[1])
        enet.fit(X_LinearSingle_train, Y_train, VIP_idx=None, Criterion=Data['ENet criterion'],\
            normalize=True, max_iter=Data['ENet max number of iterations'], tol=0.0001,\
            cv=Data['ENet cv number'], n_jobs=1, selection='random', verbose=Data['ENet verbose'])               
        enet.plot_path(2, F_ENet=Files['ENet path'], FigSize=Data['Figure size'], FileFormat=Data['Figure file format'])
        idx = enet.idx
        Gene_list = []
        for i in idx:
            Gene_list.append(genetic.Gene(i, Type=0))
        chromosome = genetic.Chromosome(Gene_list)
        chromosome.erase_score()
        chromosome.score(x_expD_train=None, x_expDn_train=None, x_lin_train=X_Linear_train,\
            y_train=Y_train, x_expD_test=None, x_expDn_test=None, x_lin_test=X_Linear_test,\
            y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond,\
            lapack_driver=ga.lapack_driver)        
        chromosome.rank_sort(x_expD=None, x_expDn=None, x_lin=X_Linear_train, y=Y_train,\
            LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)        
        t_sec = time() - t
        ga.idx_lin = list(range(0, X_Linear_train.shape[1], 1))
        ga.n_lin = len(ga.idx_lin)
        x, _ = regression.Standardize(X_Linear_train)
        ga.C = np.cov(x, rowvar=False, bias=True)
        print("\n", 'Elastic Net worked ', t_sec, 'sec')  
        print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    if Data['First algorithm'] == 'GA':        
        print('Genetic Algorithm for all features')
        ga.fit(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_Linear_train, y_train=Y_train,\
            x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_Linear_test, y_test=Y_test,\
            idx_exp=None, idx_lin=None, VIP_idx_exp=VIP_idx_exp,\
            VIP_idx_lin=VIP_idx_lin, CrossoverMethod=Data['GA crossover method'], MutationMethod=Data['GA mutation method'],\
            LinearSolver=Data['LR linear solver'], cond=Data['LR scipy condition'], lapack_driver=Data['LR scipy driver'], nIter = Data['GA max generations'])

        t_sec = time() - t
        print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
        chromosome = copy.deepcopy(ga.BestChromosome)
        print("\n", 'Features left for Backward Elimination and Search Alternative = ',\
              chromosome.Size)
            
    ga.start_time = time()
    while chromosome.Size >= Data['BE min chromosome size']:
        if chromosome.Size <= ga.ChromosomeSize:
            chromosome = ga.BestFit2(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train, x_lin=X_Linear_train,\
                y=Y_train, goal=Data['A goal'], epoch=Data['A stop time'], q_max=Data['A max queue'], model=Data['A selection criterion'], verbose=Data['A verbose'])
            chromosome.score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_Linear_train,\
                y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_Linear_test,\
                y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond,\
                lapack_driver=ga.lapack_driver) 
            chromosome.Origin = 'Best Fit'
            chromosome_copy = copy.deepcopy(chromosome)
            chromosome_copy.print_score()
            ga.DecreasingChromosomes.append(chromosome_copy)
        # chromosome must be sorted before             
        chromosome = ga.RemoveWorstGene(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train,\
            x_lin=X_Linear_train, y=Y_train, verbose=True)
        if chromosome is None:
            break # number of genes in chromosome = number of VIP genes
    t_sec = time() - t
    for i in range(0, len(ga.BestFitPath), 1): # get fitness from test set
        ga.BestFitPath[i].score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_Linear_train,\
            y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_Linear_test,\
            y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond,\
            lapack_driver=ga.lapack_driver)        
    ga.PlotChromosomes2(Files['GA path RMSE'], 5, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='RMSE',\
        Title=None, PlotType='Line', figsize=Data['Figure size'], marker_size=5, line_width=1, FileFormat=Data['Figure file format'])
    ga.PlotChromosomes2(Files['GA path R2'], 6, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='R2',\
        Title=None, PlotType='Line', figsize=Data['Figure size'], marker_size=5, line_width=1, FileFormat=Data['Figure file format'])           
    ga.PlotChromosomes2(Files['BF path RMSE'], 7, ga.BestFitPath, XAxis='Time', YAxis='RMSE',\
        Title=None, PlotType='Scatter', figsize=Data['Figure size'], marker_size=7, line_width=1, FileFormat=Data['Figure file format'])
    
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    # append ethalon to the end of list            
    gene0 = genetic.Gene(0, Type=0, p_Value=None, rank=None)
    gene1 = genetic.Gene(15, Type=0, p_Value=None, rank=None)
    gene2 = genetic.Gene(30, Type=0, p_Value=None, rank=None)
    gene3 = genetic.Gene(5, Type=0, p_Value=None, rank=None)
    gene4 = genetic.Gene(11, Type=0, p_Value=None, rank=None)
    etalon = genetic.Chromosome([gene0, gene1, gene2, gene3, gene4])
    etalon.score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_Linear_train,\
        y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_Linear_test,\
        y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond,\
        lapack_driver=ga.lapack_driver)
    if etalon is not None:
        ga.BestFitPath.append(etalon)  
        ga.DecreasingChromosomes.append(etalon)   
    ga.Results_to_xlsx(Files['Fit'],\
        FeaturesNonlinear=GenerateFeaturesResults['FeaturesExpSingleAll'],\
        FeaturesAll=FeaturesLinearAll,\
        FeaturesReduced=FeaturesLinearReduced,\
        X_Linear=X_Linear_train)
    return ga

# linear search for constant noise
def GetFitGP(Files, GaussianPrecision=10, GaussianStart=0.01, GaussianEnd=20, GaussianLen=5):
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    # Gaussian fit    
    print('Gaussian started')      
    k = 0 # estimate best gaussian
    gpR2_array = np.zeros(shape=(GaussianLen), dtype=float)
    Start = GaussianStart
    End = GaussianEnd
    while k < GaussianPrecision:
        print('Start = ', Start)
        print('End = ', End)        
        grid = np.linspace(Start, End, GaussianLen)
        for i in range(0, GaussianLen, 1):
            if gpR2_array[i] == 0:
                kernel = RBF(length_scale=grid[i], length_scale_bounds=(1e-10, 1e+10)) + WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-10, 1e+1))# + RationalQuadratic(length_scale=1.2, alpha=0.78)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
                    n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
                gp.fit(X_Gaussian_train, Y_train) # set from distances
                gpR2 = gp.score(X_Gaussian_test, Y_test)
                print(grid[i], '  ', gpR2)
                gpR2_array[i] = gpR2
        index = np.argmax(gpR2_array)
        gpR2_new = np.zeros(shape=(GaussianLen), dtype=float)
        print('Index = ', index)
        print('length_scale ', grid[index])
        print('R2 ', gpR2_array[index])
        if index == 0:
            sys.stdout.write(RED)                                
            print('Check gaussian lower interval')
            sys.stdout.write(RESET)
            Start = grid[index]
            gpR2_new[0] = gpR2_array[index]
        else:
            Start = grid[index-1]
            gpR2_new[0] = gpR2_array[index-1]
        if index == (GaussianLen-1):
            sys.stdout.write(RED)                                
            print('Check gaussian upper interval')
            sys.stdout.write(RESET)
            End = grid[index]
            gpR2_new[-1] = gpR2_array[index]
        else:
            End = grid[index+1]
            gpR2_new[-1] = gpR2_array[index+1]
        k += 1
        gpR2_array = copy.deepcopy(gpR2_new)
   
    kernel = RBF(length_scale=grid[index], length_scale_bounds=(1e-10, 1e+10)) + WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
        n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
    gp.fit(X_Gaussian_train, Y_train) # set from distances
    gpR2 = gp.score(X_Gaussian_test, Y_test)
    Print(gpR2, color=RED)
    return gp
    
# basinhopping optimizer
def GetFitGP2(Files):
    
    def fun(x):
        kernel = RBF(length_scale=x[0], length_scale_bounds=(1e-3, 1e+2)) +\
            WhiteKernel(noise_level=x[1], noise_level_bounds=(1e-10, 1e+1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
            n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
        gp.fit(X_Gaussian_train, Y_train) # set from distances
        gpR2 = gp.score(X_Gaussian_test, Y_test)
        print('R2 = ', gpR2, 'scale = ', x[0], 'noise = ', x[1])
        return -gpR2

    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    # Gaussian fit 
    
    from scipy.optimize import basinhopping
    x0 = np.array([1, 1e-4])
    res = basinhopping(fun, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None,\
        take_step=None, accept_test=None, callback=None, interval=50,\
        disp=True, niter_success=2, seed=None)
    
    return res
    
# build in optimizer
def GetFitGP3(Files):
    print('Gaussian started') 
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-1, 1e+2)) +\
        WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-20, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer='fmin_l_bfgs_b',\
        n_restarts_optimizer=2, normalize_y=True, copy_X_train=True, random_state=101)
    gp.fit(X_Gaussian_train, Y_train) # set from distances
    gpR2 = gp.score(X_Gaussian_test, Y_test)
    print('R2 = ', gpR2)    
    return gp

# user defined kernel
def GetFitGP4(Files):
    print('Gaussian started') 
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-1, 1e+2)) +\
        WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-20, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer='fmin_l_bfgs_b',\
        n_restarts_optimizer=2, normalize_y=True, copy_X_train=True, random_state=101)
    gp.fit(X_Gaussian_train, Y_train) # set from distances
    gpR2 = gp.score(X_Gaussian_test, Y_test)
    print('R2 = ', gpR2)    
    return gp

# hill climbing for Gaussian Process
    
class NodeHill(dict):
    
    def __init__(self, parent=None, length_scale=None, noise_level=None,\
        length_scale_inc=None, noise_level_inc=None, length_scale_bounds=None,\
        noise_level_bounds=None, function=None):
        
        self.parent = parent
        self.children = []
        self.length_scale = length_scale
        self.noise_level = noise_level # y
        self.length_scale_inc = length_scale_inc
        self.noise_level_inc = noise_level_inc
        self.length_scale_bounds = length_scale_bounds
        self.noise_level_bounds = noise_level_bounds
        self.function=function
        self.fitness = self.get_fitness() 
        
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
        
    def get_fitness(self):
        return self.function(self.length_scale, self.noise_level)

    def create_children(self):    
        if self.children != []:
            return
        length_scale_new = self.length_scale - self.length_scale_inc
        if length_scale_new > self.length_scale_bounds[0]: # greater than left bound
            child = NodeHill(parent=self, length_scale=length_scale_new,\
                noise_level=self.noise_level, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)
            self.children.append(child)
        length_scale_new = self.length_scale + self.length_scale_inc
        if length_scale_new < self.length_scale_bounds[1]: # smaller than right bound
            child = NodeHill(parent=self, length_scale=length_scale_new,\
                noise_level=self.noise_level, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)            
            self.children.append(child)
        noise_level_new = self.noise_level - self.noise_level_inc
        if noise_level_new > self.noise_level_bounds[0]: # greater than left bound
            child = NodeHill(parent=self, length_scale=self.length_scale,\
                noise_level=noise_level_new, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)
            self.children.append(child)
        noise_level_new = self.noise_level + self.noise_level_inc
        if noise_level_new < self.noise_level_bounds[0]: # smaller than right bound
            child = NodeHill(parent=self, length_scale=self.length_scale,\
                noise_level=noise_level_new, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)
            self.children.append(child)            
        return
    
    def get_highest_valued_child(self):
        if self.children == []:
            return None
        best = self.children[0]
        for i in range(1, len(self.children), 1):
            if best.fitness < self.children[i].fitness:
                best = self.children[i]
        return best

def GetFitGP5(Files, Data):
    
    def f(x, y):
        kernel = RBF(length_scale=x, length_scale_bounds=None) +\
            WhiteKernel(noise_level=y, noise_level_bounds=None)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
            n_restarts_optimizer=0, normalize_y=True, copy_X_train=True,\
            random_state=Data['Random state'])
        gp.fit(X_Gaussian_train, Y_train) # set from distances
        gpR2 = gp.score(X_Gaussian_test, Y_test)
        return gpR2
    
    def hill(length_scale_start, noise_level_start):

        def get_root(node):
            while node.parent is not None:
                node = node.parent
            return node
        
        fitness = []
        length_scale = []
        noise_level = []

# recursive; goes through all tree and stores path into lists        
        def preorderTraversal(node):
            fitness.append(node.fitness)
            length_scale.append(node.length_scale)
            noise_level.append(node.noise_level)
            for child in node.children:
                preorderTraversal(child)
            return

        Length_scale_inc = Data['GP length scale increment'] * (Data['GP length scale bounds'][0] + Data['GP length scale bounds'][1]) / 2
        Length_scale_inc_min = Data['GP min length scale increment'] * (Data['GP length scale bounds'][0] + Data['GP length scale bounds'][1]) / 2
        Noise_level_inc = Data['GP noise level increment'] * (Data['GP noise level bounds'][0] + Data['GP noise level bounds'][1]) / 2
        Noise_level_inc_min = Data['GP min noise level increment'] * (Data['GP noise level bounds'][0] + Data['GP noise level bounds'][1]) / 2        
        current = NodeHill(parent=None, length_scale=length_scale_start,\
            noise_level=noise_level_start, length_scale_inc=Length_scale_inc,\
            noise_level_inc=Noise_level_inc, length_scale_bounds=Data['GP length scale bounds'],\
            noise_level_bounds=Data['GP noise level bounds'], function=f)
        count = 0 
        while True:
            count += 1
            print('R2=', current.fitness, 'length_scale=', current.length_scale,\
                 'noise_level=', current.noise_level, 'length_scale_inc=',\
                 current.length_scale_inc, 'noise_level_inc=', current.noise_level_inc)
            current.create_children()
            successor = current.get_highest_valued_child()            
            if successor.fitness <= current.fitness: # looking for max
                Finish = True
                current.children = [] # erase old children
                Length_scale_inc = Length_scale_inc / 2
                Noise_level_inc = Noise_level_inc / 2
                if Length_scale_inc > Length_scale_inc_min:
                    current.length_scale_inc = Length_scale_inc
                    Finish = False
                if Noise_level_inc > Noise_level_inc_min:
                    current.noise_level_inc = Noise_level_inc
                    Finish = False
                if Finish:
                    root = get_root(current)
                    preorderTraversal(root)
                    path = {'R2': fitness, 'length_scale': length_scale, 'noise_level': noise_level}
                    return current, count, path
                else:
                    continue # with old node but smaller intervals
            else:
                current = successor
    
    print('Gaussian started') 
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test']) 
    nodes = []
    node, count, Path = hill(Data['GP initial length scale'], Data['GP initial noise level'])
    nodes.append(node)
    if Data['GP hill simulations'] is not None:
        for i in range(0, Data['GP hill simulations'], 1):
            print('Simulation ', i, 'out of ', Data['GP hill simulations'])
            length_scale_start = random.random() * (Data['GP length scale bounds'][1] - Data['GP length scale bounds'][0]) + Data['GP length scale bounds'][0]
            noise_level_start = random.random() * (Data['GP noise level bounds'][1] - Data['GP noise level bounds'][0]) + Data['GP noise level bounds'][0]    
            node, count, path = hill(length_scale_start, noise_level_start)
            nodes.append(node)
            Path['R2'].extend(path['R2'])
            Path['length_scale'].extend(path['length_scale'])
            Path['noise_level'].extend(path['noise_level'])
    best_node = nodes.pop(0)
    while len(nodes) > 0:
        node = nodes.pop(0)
        if node.fitness > best_node.fitness:
            best_node = node                 
    kernel = RBF(length_scale=best_node.length_scale, length_scale_bounds=Data['GP length scale bounds']) +\
        WhiteKernel(noise_level=best_node.noise_level, noise_level_bounds=Data['GP noise level bounds'])
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
        n_restarts_optimizer=0, normalize_y=True, copy_X_train=True,\
        random_state=Data['Random state'])
    gp.fit(X_Gaussian_train, Y_train) # set from distances   
    gpR2 = gp.score(X_Gaussian_test, Y_test)
    Print('{} {} {} {} {} {}'.format('Best Gaussian R2:', str(gpR2), 'length_scale:',\
          best_node.length_scale, 'noise_level:', best_node.noise_level), color=RED)
    return gp, Path

def EraseFile(F):
    try:        
        os.remove(F)
    except:
        return False
    return True

def CopyFile(F, Dir):
    if os.path.isfile(F):
        shutil.copy2(F, Dir)    
        return True
    else:
        return False

def MoveFile(F, Dir):
    if os.path.isfile(F):
        shutil.move(F, Dir)    
        return True
    else:
        return False

############## PLOT ###########################################################

def get_bounds(*argv, adj=0.02):
# return Min and Max for boundaries. *argv - 1D array like 
    max_list = []
    min_list = []
    for arg in argv:
        if arg is not None:
            max_list.append(np.max(arg))
            min_list.append(np.min(arg))       
    YMax = max(max_list)
    YMin = min(min_list)
    YMinAdj = YMin - (YMax-YMin) * adj
    YMaxAdj = YMax + (YMax-YMin) * adj        
    return YMinAdj, YMaxAdj

def PlotHistogram(FileName=None, y_true=None, y_pred=None, FigNumber=1,\
        FigSize=(4,3), Bins='auto', xLabel=None, yLabel='Frequency',\
        FileFormat='eps', Resolution=100):
    
    if y_true.size != y_pred.size:
        return False
    error = np.zeros(shape=(y_true.size), dtype=float)      
    error[:] = y_pred[:] - y_true[:] 
    fig = plt.figure(FigNumber, figsize=FigSize)
    plt.hist(error, bins=Bins)
    if xLabel is not None:
        plt.xlabel(xLabel)
    plt.ylabel(yLabel)
#    plt.show(fig)
    if FileName is not None:
        F = '{}{}{}'.format(FileName, '.', FileFormat)
        plt.savefig(F, bbox_inches='tight', format=FileFormat, dpi=Resolution)
        plt.close(fig)         
    return 

def plot_contour(x, y, z, x_res=100, y_res=100, FileName=None,\
        FileFormat='eps', FigSize=(4,3), xTitle=None, yTitle=None,\
        barTitle=None, Resolution=100):
    
    fig = plt.figure(100, figsize=FigSize)
    x_min, x_max = get_bounds(x, adj=0.02)
    y_min, y_max = get_bounds(y, adj=0.02)
    xi = np.linspace(x_min, x_max, x_res)
    yi = np.linspace(y_min, y_max, y_res)
    zi = griddata(x, y, z, xi, yi, interp='linear')
    plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    plt.contourf(xi, yi, zi, 15, vmax=abs(zi).max(), vmin=-abs(zi).max())
    cbar = plt.colorbar()  # draw colorbar
    if barTitle is not None:
        cbar.ax.set_ylabel(barTitle)
    idx_max = np.asarray(z).argmax()
    plt.scatter(x[idx_max], y[idx_max], marker='o', s=5, zorder=10)
    if xTitle is not None:
        plt.xlabel(xTitle)
    if yTitle is not None:
        plt.ylabel(yTitle)    
#    plt.show()
    if FileName is not None:
        F = '{}{}{}'.format(FileName, '.', FileFormat)
        plt.savefig(F, bbox_inches='tight', format=FileFormat, dpi=Resolution)
        plt.close(fig)         
    return

def Plot(CoM=None, E_True=None, nFunctions=None, E_Predicted=None, xLabel='R, CoM (Å)', \
        yEnergyLabel='Average Energy (kJ/mol)', yErrorLabel='Average bin error (kJ/mol)',\
        Legend=None, Grid=None, GridTrained=None, TrainedIntervals=None,\
        NontrainedIntervals=None, F_Error='Error', F_Energy='Energy', F_R2='R2', figsize=(4, 3),\
        fig_format='eps', marker_size=1, line_width = 0.3, bounds=None, Resolution=100):

    """
    first description in Legend is true energy function
    """
    if bounds is not None:
        i = 0
        while len(Grid) > i:
            if bounds[0] > Grid[i][0]:
                del(Grid[i])
                continue
            if bounds[1] < Grid[i][1]:
                del(Grid[i])
                continue            
            i += 1
            
    marker_energy = '.'
# circle, square, star, thin diamond, hexagon,  point, pixel   
    marker_fun = ['o', 's', '*', 'd', 'h', '.', ',']
    color_train_energy = 'red'
    color_nontrain_enegry = 'blue'
    color_nontrain_fun = ['blue', 'green', 'violet']
    color_train_fun = ['blue', 'green', 'violet']
    if nFunctions == 0:
        E_Predicted = None
    E_Predicted = np.asarray(E_Predicted)
    Size = CoM.size # number of observations    
    F_R2 = '{}{}{}'.format(F_R2, '.', fig_format)
    F_Error = '{}{}{}'.format(F_Error, '.', fig_format)
    F_Energy = '{}{}{}'.format(F_Energy, '.', fig_format)       
    if Grid is None:
        RMin, RMax = get_bounds(CoM, adj=0)
        nIntervals = 30
        Grid = CreateGrid(round(RMin, 2), round(RMax, 2), round((round(RMax, 2) - round(RMin, 2)) / nIntervals, 2))
    else:
        nIntervals = len(Grid)
    if GridTrained is None:
        GridTrained = []
    n = np.zeros(shape=(nIntervals), dtype=float) # n[i] = number of observations in interval    
    X = np.zeros(shape=(nIntervals), dtype=float) # X on plot
    E_True_bin = np.zeros(shape=(nIntervals), dtype=float) 
    Trained = np.empty(shape=(nIntervals), dtype=bool) # True if bin in trained region    
    e_true_list = [[] for _ in range(nIntervals)] # true energy values arranged according to grid
    for i in range(0, Size, 1):# count number of observations 
        j = InInterval(CoM[i], Grid)
        if j != -10: # not in any of intervals
            E_True_bin[j] += E_True[i] # cumulative true energy per bin
            e_true_list[j].append(E_True[i])
            n[j] += 1 # number of observations in each bin
            X[j] += CoM[i]
    E_True_bin = np.divide(E_True_bin, n) # average value
    X = np.divide(X, n) # average value of distance COM
    Error_bin = np.zeros(shape=(nFunctions, nIntervals), dtype=float) # Y axis for error plot
    E_Predicted_bin = np.zeros(shape=(nFunctions, nIntervals), dtype=float) # Y axis for energy plot
# access:     error_list[Function number][grid number][point in grid]
# append point to grid: error_list[Function number][grid number].append(value)    
    error_list = [[[] for _ in range(nIntervals)] for _ in range(nFunctions)]
    e_pred_list = [[[] for _ in range(nIntervals)] for _ in range(nFunctions)]
    for k in range(0, nFunctions, 1): # for each function
        error = np.zeros(shape=(Size), dtype=float)
        for i in range(0, Size, 1):
            error[i] = abs(E_True[i] - E_Predicted[k, i])
        error_bin = np.zeros(shape=(nIntervals), dtype=float)
        e_Predicted_bin = np.zeros(shape=(nIntervals), dtype=float)  
        for i in range(0, Size, 1):
            j = InInterval(CoM[i], Grid)
            if j != -10: # not in any of intervals
                error_list[k][j].append(error[i])
                e_pred_list[k][j].append(E_Predicted[k, i])
                error_bin[j] += error[i] # cumulative error per bin for each function
                e_Predicted_bin[j] += E_Predicted[k, i] # cumulative energy per bin for each functione
        error_bin = np.divide(error_bin, n) # average error per bin
        e_Predicted_bin = np.divide(e_Predicted_bin, n) # average predicted energy per bin
        Error_bin[k, :] = error_bin[:]
        E_Predicted_bin[k, :] = e_Predicted_bin[:]        
    for i in range(0, nIntervals, 1): # separate trained region from non-trained
        if InInterval(X[i], GridTrained) != -10:
            Trained[i] = True
        else:
            Trained[i] = False
    nTrained_bins = np.count_nonzero(Trained) # cannot be 0
    nNontrained_bins = nIntervals - nTrained_bins # can be 0
    X_Trained = np.zeros(shape=(nTrained_bins), dtype=float)
    E_True_bin_trained = np.zeros(shape=(nTrained_bins), dtype=float)
    Error_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)    
    E_Predicted_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)   
    Std_true_energy_trained = np.zeros(shape=(nTrained_bins), dtype=float)
    Std_energy_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float) 
    Std_error_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float) 
    if nNontrained_bins != 0:
        X_Nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
        E_True_bin_nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
        Error_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
        E_Predicted_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
        Std_true_energy_nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
        Std_energy_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float) 
        Std_error_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float) 
    else:
        X_Nontrained, E_True_bin_nontrained, Error_bin_nontrained,\
            E_Predicted_bin_nontrained = None, None, None, None
            
    Std_true_energy = np.zeros(shape=(nIntervals), dtype=float)
    Std_energy = np.zeros(shape=(nFunctions, nIntervals), dtype=float)
    Std_error = np.zeros(shape=(nFunctions, nIntervals), dtype=float)    
    R2 = np.zeros(shape=(nFunctions, nIntervals), dtype=float)
    mse = np.zeros(shape=(nFunctions, nIntervals), dtype=float)

    for i in range(0, nFunctions, 1):
        for j in range(nIntervals):
            if i == 0:
                Std_true_energy[j] = np.std(e_true_list[j]) # STD true energy for each bin
            Std_energy[i, j] = np.std(e_pred_list[i][j]) # STD predicted energy for each bin
            Std_error[i, j] = np.std(error_list[i][j]) # STD error for each bin
            R2[i, j] = skm.r2_score(e_true_list[j], e_pred_list[i][j])
            mse[i, j] = skm.mean_squared_error(e_true_list[j], e_pred_list[i][j])
           
    j = 0 # trained index
    k = 0 # nontrained index
# separate trained and nontrained regions in different arrays
    for i in range(0, nIntervals, 1):
        if Trained[i]:
            X_Trained[j] = X[i]
            E_True_bin_trained[j] = E_True_bin[i]
            Std_true_energy_trained[j] = Std_true_energy[i]
            for l in range(0, nFunctions, 1):
                Error_bin_trained[l, j] = Error_bin[l, i]
                E_Predicted_bin_trained[l, j] = E_Predicted_bin[l, i]
                Std_energy_trained[l, j] = Std_energy[l, i]
                Std_error_trained[l, j] = Std_error[l, i]
            j += 1
        else:
            X_Nontrained[k] = X[i]
            E_True_bin_nontrained[k] = E_True_bin[i]
            Std_true_energy_nontrained[j] = Std_true_energy[i]
            for l in range(0, nFunctions, 1):
                Error_bin_nontrained[l, k] = Error_bin[l, i]
                E_Predicted_bin_nontrained[l, k] = E_Predicted_bin[l, i]
                Std_energy_nontrained[l, j] = Std_energy[l, i]
                Std_error_nontrained[l, j] = Std_error[l, i]                
            k += 1    
# group observations according to trained and test regions
    x_trained = []
    e_True_bin_trained = []
    
    for i in range(0, len(TrainedIntervals), 1):
        x_trained.append([])
        e_True_bin_trained.append([])
    if nNontrained_bins != 0:
        x_nontrained = []
        e_True_bin_nontrained = []
        for i in range(0, len(NontrainedIntervals), 1):
            x_nontrained.append([])
            e_True_bin_nontrained.append([])
    for i in range(0, nTrained_bins, 1):
        j = InInterval(X_Trained[i], TrainedIntervals)
        if j != -10:
            x_trained[j].append(X_Trained[i])
            e_True_bin_trained[j].append(E_True_bin_trained[i])
    if nNontrained_bins != 0:
        for i in range(0, nNontrained_bins, 1):
            j = InInterval(X_Nontrained[i], NontrainedIntervals)
            if j != -10:
                x_nontrained[j].append(X_Nontrained[i])
                e_True_bin_nontrained[j].append(E_True_bin_nontrained[i])

    if Legend is None: # assigne some text for empty Legend list
        Legend = []
        Legend.append('{}'.format('Reference energy'))
        for i in range(0, nFunctions, 1):
            Legend.append('{} {}'.format('Function', i+1))
            
# plot Error
    fig_error = plt.figure(1, figsize=figsize)
    xMin, xMax = get_bounds(Grid, adj=0.02)
    yMin, yMax = get_bounds(Error_bin_trained, Error_bin_nontrained, adj=0.02)    
    plt.xlim((xMin, xMax))
#    plt.ylim((yMin, yMax))
    for i in range(0, nFunctions, 1):
        plt.plot(X_Trained, Error_bin_trained[i, :], \
            c=color_train_fun[i], marker=marker_fun[i], label=Legend[i+1])
        if nNontrained_bins != 0:
            plt.plot(X_Nontrained, Error_bin_nontrained[i, :],\
                c=color_nontrain_fun[i], marker=marker_fun[i], label=None)

        plt.errorbar(X, Error_bin[i][:], yerr=Std_error[i, :], xerr=None, fmt='none',\
            ecolor=color_nontrain_fun[i], elinewidth=1, capsize=3, barsabove=False, lolims=False,\
            uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, hold=None, data=None)

    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yErrorLabel)
#    plt.show(fig_error)
    plt.savefig(F_Error, bbox_inches='tight', format=fig_format, dpi=Resolution)
    plt.close(fig_error)

# plot MSE    
    fig_mse = plt.figure(3, figsize=figsize)
    xMin, xMax = get_bounds(Grid, adj=0.02)   
    plt.xlim((xMin, xMax))
    for i in range(0, nFunctions, 1):
        plt.plot(X, mse[i, :], c=color_train_fun[i], marker=marker_fun[i], label=Legend[i+1])
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel('MSE')
#    plt.show(fig_mse)
    F_mse = 'mse.png'
    plt.savefig(F_mse, bbox_inches='tight', format=fig_format, dpi=Resolution)
    plt.close(fig_mse)
    
# plot R2    
    fig_R2 = plt.figure(4, figsize=figsize)
    xMin, xMax = get_bounds(Grid, adj=0.02)   
    plt.xlim((xMin, xMax))
    for i in range(0, nFunctions, 1):
        plt.plot(X, R2[i, :], c=color_train_fun[i], marker=marker_fun[i], label=Legend[i+1])
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel('R2')
#    plt.show(fig_R2)
    plt.savefig(F_R2, bbox_inches='tight', format=fig_format, dpi=Resolution)
    plt.close(fig_R2)
    
# plot Energy. x bounds are the same as prev. plot
    fig_energy = plt.figure(2, figsize=figsize)
    yMin, yMax = get_bounds(E_True_bin_trained, E_True_bin_nontrained,\
        E_Predicted_bin_trained, E_Predicted_bin_nontrained, adj=0.02)
    plt.xlim((xMin, xMax))
#    plt.ylim((yMin, yMax))
    for i in range(0, len(TrainedIntervals), 1): # plot true energy on trained region
        if i == 0: # plot legend only once
            legend = Legend[0]
        else:
            legend = None
        plt.plot(x_trained[i], e_True_bin_trained[i], c=color_train_energy,\
            markersize=marker_size, marker=marker_energy, label=legend, lw=line_width) # true energy on trained intervals
# bars for true energy            
    plt.errorbar(X, E_True_bin, yerr=Std_true_energy, xerr=None, fmt='none',\
        ecolor=color_train_energy, elinewidth=1, capsize=3, barsabove=False, lolims=False,\
        uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, hold=None, data=None)
        
    if nNontrained_bins != 0: # plot true energy on non-trained region without legend
        for i in range(0, len(NontrainedIntervals), 1):
            plt.plot(x_nontrained[i], e_True_bin_nontrained[i], markersize=marker_size,\
                c=color_nontrain_enegry, marker=marker_energy, label=None, lw=line_width) # true energy on nontrained intervals
    for i in range(0, nFunctions, 1): # plot functions on trained region
        plt.plot(X_Trained, E_Predicted_bin_trained[i], \
            c=color_train_fun[i], marker=marker_fun[i], label=Legend[i+1])
        if nNontrained_bins != 0: # plot functions on non-trained region
            plt.plot(X_Nontrained, E_Predicted_bin_nontrained[i], \
                c=color_nontrain_fun[i], marker=marker_fun[i], label=None)
# bars for predicted energy        
        plt.errorbar(X, E_Predicted_bin[i], yerr=Std_energy[i], xerr=None, fmt='none',\
            ecolor=color_nontrain_fun[i], elinewidth=1, capsize=3, barsabove=False, lolims=False,\
            uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, hold=None, data=None)


    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yEnergyLabel)
#    plt.show(fig_energy)    
    plt.savefig(F_Energy, bbox_inches='tight', format=fig_format, dpi=Resolution)
    plt.close(fig_energy)
    return
        
def Proceed(Files, Data):
# obtain fit for one particular configuration defined by Files and Data
    
# for set 6
# seed 10; ConfidenceInterval=0.95; GridSpacing=0.2; TrainIntervals=[(2.8, 5), (8, 10)]
# Best Gaussian R2: 0.448418157908 length_scale: 1.4697829746212416 noise_level: 0.06325484382297103

# set 4
# ConfidenceInterval=0.85
# set 2
# ConfidenceInterval=0.95 MinCorrBestFit=0.9 model='Parent'
# ChromosomeSize=15, StopTime=600, BestFitStopTime=100, nIter=200
    
    FilterDataDict = FilterData(Files, Data)
            
    f, f_old = RedirectPrintToFile(Files['Set params'])
    for i, j in FilterDataDict.items():
        print(i, ':', j)   
    RedirectPrintToConsole(f, f_old)
    
    FeaturesDict = GenerateFeatures(Files)
       
    ga = GetFitGA(FilterDataDict, Files, Data, FeaturesDict)
    print("Gaussian started")
    gp, Path = GetFitGP5(Files, Data)
# some mistake    
#    plot_contour(Path['length_scale'], Path['noise_level'], Path['R2'], 100, 100,\
#        FileName=Files['GP path'], FileFormat=Data['Figure file format'],\
#        FigSize=Data['Figure size'], xTitle='Length scale',\
#        yTitle='Noise level', barTitle='Gaussian R2', Resolution=Data['Figure resolution'])
    print("Read files")
    COM_test = IOfunctions.ReadCSV(Files['COM test']) 
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])   
    if FeaturesDict['FeaturesLinearSingleAll'] is not None:
        X_LinearSingle_train = IOfunctions.ReadCSV(Files['Linear Single Train'])
        X_LinearSingle_test = IOfunctions.ReadCSV(Files['Linear Single Test'])
    else:
        X_LinearSingle_train, X_LinearSingle_test = None, None
    if FeaturesDict['FeaturesLinearDoubleAll'] is not None:
        X_LinearDouble_train = IOfunctions.ReadCSV(Files['Linear Double Train'])
        X_LinearDouble_test = IOfunctions.ReadCSV(Files['Linear Double Test'])
    else:
        X_LinearDouble_train, X_LinearDouble_test = None, None        
    if FeaturesDict['FeaturesLinearTripleAll'] is not None:        
        X_LinearTriple_train = IOfunctions.ReadCSV(Files['Linear Triple Train'])
        X_LinearTriple_test = IOfunctions.ReadCSV(Files['Linear Triple Test'])
    else:
        X_LinearTriple_train, X_LinearTriple_test = None, None    
    if FeaturesDict['FeaturesExpSingleAll'] is not None:     
        X_ExpSingleD_test = IOfunctions.ReadCSV(Files['Exp Single Test D'])
    else:
        X_ExpSingleD_test = None
    if FeaturesDict['FeaturesExpDoubleAll'] is not None:          
        X_ExpSingleDn_test = IOfunctions.ReadCSV(Files['Exp Single Test D^n'])    
    else:
        X_ExpSingleDn_test = None
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Test'])
    print("Predict gaussian")
    y_pred = gp.predict(X_Gaussian_test)
    ga.gp_MSE = skm.mean_squared_error(Y_test, y_pred)
    ga.gp_R2 = gp.score(X_Gaussian_test, Y_test)
    print("Save objects")
    IOfunctions.SaveObject(Files['GA object'], ga)    
    IOfunctions.SaveObject(Files['GP object'], gp) 

    if (X_LinearSingle_train is not None) and (X_LinearDouble_train is not None) and (X_LinearTriple_train is not None): # all three exist
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearSingleAll'])
        FeaturesLinearAll.extend(FeaturesDict['FeaturesLinearDoubleAll'])
        FeaturesLinearAll.extend(FeaturesDict['FeaturesLinearTripleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearSingleReduced'])
        FeaturesLinearReduced.extend(FeaturesDict['FeaturesLinearDoubleReduced'])
        FeaturesLinearReduced.extend(FeaturesDict['FeaturesLinearTripleReduced'])
    elif X_LinearSingle_train is not None and X_LinearDouble_train is not None: # single + double exist
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearSingleAll'])
        FeaturesLinearAll.extend(FeaturesDict['FeaturesLinearDoubleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearSingleReduced'])
        FeaturesLinearReduced.extend(FeaturesDict['FeaturesLinearDoubleReduced'])
    elif X_LinearSingle_train is not None and X_LinearDouble_train is None: # only single
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearSingleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearSingleReduced'])
    elif X_LinearSingle_train is None and X_LinearDouble_train is not None: # only double
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearDoubleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearDoubleReduced'])
    else: # no linear features
        FeaturesLinearAll = None
        FeaturesLinearReduced = None
    if (X_LinearSingle_test is not None) and (X_LinearDouble_test is not None) and (X_LinearTriple_test is not None): # all exist
        X_Linear_test = np.concatenate((X_LinearSingle_test,X_LinearDouble_test,X_LinearTriple_test),axis=1)
    elif X_LinearSingle_test is not None and X_LinearDouble_test is not None: # single + double exist
        X_Linear_test = np.concatenate((X_LinearSingle_test,X_LinearDouble_test),axis=1)
    elif X_LinearSingle_test is not None and X_LinearDouble_test is None: # only single
        X_Linear_test = X_LinearSingle_test
    elif X_LinearSingle_test is None and X_LinearDouble_test is not None: # only double
        X_Linear_test = X_LinearDouble_test
    else: # no linear features
        X_Linear_test = None   
    
    y_pred_gp = gp.predict(X_Gaussian_test)
    # converte energy to kJ / mol
    y_test_kj = HARTREE_TO_KJMOL * Y_test
    y_pred_gp_kj = HARTREE_TO_KJMOL * y_pred_gp
    
    PlotHistogram(FileName='{} {} {}'.format(Files['GP energy error histogram'],\
        X_Gaussian_test.shape[1], 'predictors'), y_true=y_test_kj, y_pred=y_pred_gp_kj,\
        FigNumber=1, FigSize=Data['Figure size'], Bins='auto', FileFormat=Data['Figure file format'],\
        xLabel='GP energy error, kJ/mol', yLabel='Frequency', Resolution=Data['Figure resolution'])
            
    for chromosome in ga.DecreasingChromosomes:
        y_pred_ga = chromosome.predict(x_expD=X_ExpSingleD_test,\
            x_expDn=X_ExpSingleDn_test, x_lin=X_Linear_test)
        y_pred_ga_kj = HARTREE_TO_KJMOL * y_pred_ga    
        PlotHistogram(FileName='{} {} {}'.format(Files['GA energy error histogram'],\
            chromosome.Size, 'predictors'), y_true=y_test_kj, y_pred=y_pred_ga_kj,\
            FigNumber=2, FigSize=Data['Figure size'], Bins='auto', FileFormat=Data['Figure file format'],\
            xLabel='GA energy error, kJ/mol', yLabel='Frequency', Resolution=Data['Figure resolution'])
        Plot(CoM=COM_test, E_True=y_test_kj, nFunctions=2, xLabel='R, CoM (Å)', \
            yErrorLabel='Average Error (kJ/mol)', yEnergyLabel='Average Energy (kJ/mol)',\
            E_Predicted=[y_pred_ga_kj, y_pred_gp_kj],\
            Legend=['Reference', 'GA', 'GP'], Grid=FilterDataDict['Test Grid'],\
            GridTrained=FilterDataDict['Train Grid'],\
            TrainedIntervals=FilterDataDict['Train Intervals'],\
            NontrainedIntervals=FilterDataDict['Test Intervals'],\
            F_Error='{} {} {}'.format(Files['Plot error'], chromosome.Size, 'predictors'),\
            F_Energy='{} {} {}'.format(Files['Plot energy'], chromosome.Size, 'predictors'),\
            figsize=Data['Figure size'], fig_format=Data['Figure file format'], marker_size=3,\
            line_width = 1, bounds=(Data['Grid start'] , Data['Grid end']), Resolution=Data['Figure resolution'])

    return FilterDataDict, FeaturesDict
