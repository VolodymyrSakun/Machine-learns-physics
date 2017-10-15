from project1 import library
from project1 import IOfunctions
import random
import numpy as np
import pickle
#import shutil
import copy

if __name__ == '__main__':

    F = 'SET 6.x'
    F_MoleculesDescriptor = 'MoleculesDescriptor.'
    F_Train = 'Training Set.x'
    F_Test = 'Test Set.x'
    F_Filter = 'Filter.dat'
    F_Gaussian = 'datafile.x'

    TrainIntervals = [(2.4, 14)] # (Low, High)   
#    TrainIntervals = [(0, 5), (7, 9)] # (Low, High) 2 water big
    MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=F_MoleculesDescriptor)
    GridStart = 2.4 # initial estimation of min ang max average distances
    GridEnd = 14.0
    GridSpacing = 0.2
    ConfidenceInterval = 1 # same as PDF but middle part
    TestFraction = 0.2
    TrainFraction = 1
    RandomSeed = 101
    if RandomSeed is not None:
        random.seed(RandomSeed)
    else:
        random.seed()    
    GridTrain = [] # small intervals
    GridTest = [] # small intervals
    i = GridStart
    while i < (GridEnd-GridSpacing):
        GridTrain.append((round(i, 2), round(i+GridSpacing, 2)))
        GridTest.append((round(i, 2), round(i+GridSpacing, 2)))
        i += GridSpacing
    N = np.zeros(shape=(len(GridTest)), dtype=int) # number of points in each grid  
    NTrain = list(np.zeros(shape=(len(GridTrain)), dtype=int)) # count test points
    NTest = list(np.zeros(shape=(len(GridTest)), dtype=int)) # count test points        
    Records = IOfunctions.ReadRecords(F, MoleculePrototypes) # Read records
    DMin = 1000
    DMax = 0
    nMolecules = Records[0].nMolecules
# Determine DMin, DMax and fill N    
    for record in Records:
        if nMolecules == 2:
            if record.R_Average > DMax:
                DMax = record.R_Average
            if record.R_Average < DMin:
                DMin = record.R_Average
            if record.R_Average >= GridStart:
                j = int((record.R_Average - GridStart) / GridSpacing)
                if j < len(N):
                    N[j] += 1
        else:
            if record.R_CenterOfMass_Average > DMax:
                DMax = record.R_CenterOfMass_Average
            if record.R_CenterOfMass_Average < DMin:
                DMin = record.R_CenterOfMass_Average
            if record.R_CenterOfMass_Average >= GridStart:
                j = int((record.R_CenterOfMass_Average - GridStart) / GridSpacing)
                if j < len(N):
                    N[j] += 1
# Estimate number of points per grid
    n = np.asarray(N.nonzero()).reshape(-1) # indices with nonzero records nonzero   
    nGrids = int(len(n) * ConfidenceInterval)
    N_list = list(N)
    N_Reduced = []
    while len(N_Reduced) < nGrids:
        i = np.argmax(N_list)
        N_Reduced.append(N_list[i])
        del(N_list[i])
    nPointsGrid = N_Reduced[-1]
    nTestPointsGrid = int(nPointsGrid * TestFraction)
    nTotalTrainPointsGrid = nPointsGrid - nTestPointsGrid  
    nTrainPointsGrid = int(nTotalTrainPointsGrid * TrainFraction)
    N_list = list(N)    
    i = 0
    while i < len(N_list): # remove regions where there are not enough points
# training region        
        if (library.InInterval(GridTest[i][0], TrainIntervals) != -10) and (library.InInterval(GridTest[i][1], TrainIntervals) != -10):
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
        if (library.InInterval(GridTrain[i][0], TrainIntervals) != -10) and (library.InInterval(GridTrain[i][1], TrainIntervals) != -10):
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
        j = library.InInterval(d, GridTrain) 
        if j != -10: # in training region?
            if NTrain[j] < nTrainPointsGrid: # append to training set
                NTrain[j] += 1
                RecordsTrain.append(record)
            else:  # if it is full, append to test set
                j = library.InInterval(d, GridTest) # which interval?
                if j != -10:
                    if NTest[j] < nTestPointsGrid: 
                        NTest[j] += 1
                        RecordsTest.append(record)              
        else: # not training region
            j = library.InInterval(d, GridTest) # which interval?
            if j != -10:
                if NTest[j] < nTestPointsGrid: # append to test set only
                    NTest[j] += 1
                    RecordsTest.append(record)
        del(Records[r]) 

    IOfunctions.store_records(F_Train, RecordsTrain) # store trained set
    IOfunctions.store_records(F_Test, RecordsTest) # store test set
#    shutil.copyfile(F_Train, F_Gaussian) # copy train set
#    f_source = open(F_Test, "r") # read test set
#    re0 = f_source.readlines() 
#    f_source.close()
#    f_target = open(F_Gaussian, "a") # append test set to train set for gaussian
#    f_target.writelines(re0)
#    f_target.close()
    TestIntervals = [] # Define test regions
    if TrainIntervals[0][0] != 0:
        TestIntervals.append((0, TrainIntervals[0][0]))
    for i in range(0, (len(TrainIntervals)-1), 1):
        if TrainIntervals[i][1] != TrainIntervals[i+1][0]:
            TestIntervals.append((TrainIntervals[i][1], TrainIntervals[i+1][0]))
    if TrainIntervals[-1][1] != GridTest[-1][1]:
        TestIntervals.append((TrainIntervals[-1][1], GridTest[-1][1]))
    
    results = {'Initial dataset': F, 'Number of molecules per record': nMolecules,\
               'Train Intervals': TrainIntervals, 'Test Intervals': TestIntervals, 'Train records number': len(RecordsTrain),\
               'Train Grid': GridTrain, 'Test Grid': GridTest, 'Test records number': len(RecordsTest),\
               'Molecule prototypes': MoleculePrototypes, 'Max points per grid': nPointsGrid,\
               'Train points per grid': nTrainPointsGrid, 'Train Fraction Used': TrainFraction,\
               'Test points per grid': nTestPointsGrid, 'Confidence Interval used': ConfidenceInterval,\
               'Training Set': F_Train, 'Test Set': F_Test}
    
# save results
    f = open(F_Filter, "wb")
    pickle.dump(results, f)
    f.close()   
# store results in txt file    
    l = []
    a = results.keys()
    for i in a:
        s = str(i) + "\n"
        s1 = str(results[i]) + "\n"
        l.append(s)
        l.append(s1)
        l.append("\n")
    f = open('results.txt', "w")
    f.writelines(l)
    f.close()
    

    
    