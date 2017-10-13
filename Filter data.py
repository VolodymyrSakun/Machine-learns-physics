from project1 import library
from project1 import IOfunctions
import random
import numpy as np
import pickle

if __name__ == '__main__':

#    F = 'datafile1 from github gaussian process.x'
#    F = 'datafile3 2 water molecules.x'
    F = 'datafile4 3 water molecules small.x'
#    F = 'datafile5 3 water molecules big.x'    
    F_MoleculesDescriptor = 'MoleculesDescriptor.'
    F_Train = 'Training Set.x'
    F_Test = 'Test Set.x'
    F_Filter = 'Filter.dat'
    

#    Intervals = [(0, 4.3), (7.37, 100)] # (Low, High)   set 1 
#    Intervals = [(0, 4.2), (14.93, 100)] # (Low, High)    set 3
#    Intervals = [(0, 1.9), (2.07, 100)] # (Low, High)   set 4
#    Intervals = [(0, 1.95), (4.03, 100)] # (Low, High)    set 5

    Intervals = [(0, 1.9), (2.07, 100)] # (Low, High)   
    nRecordsTrain = np.zeros(shape=(len(Intervals)), dtype=float)
    MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=F_MoleculesDescriptor)
    TrainFraction = 0.9
    TestFitFraction = 0.1
    nTestPoints = 1000
    RandomSeed = 101
    if RandomSeed is not None:
        random.seed(RandomSeed)
    else:
        random.seed()
    Records = IOfunctions.ReadRecords(F, MoleculePrototypes)
    nMolecules = Records[0].nMolecules
    RecordsTrain = []
    RecordsTest = []
    for i in range(0, len(Records), 1):
        if nMolecules == 2:
            I = library.InInterval(Records[i].R_Average, Intervals)
            if I != -1:
                RecordsTrain.append(Records[i])
            else:
                RecordsTest.append(Records[i])
        elif nMolecules > 2:
            I = library.InInterval(Records[i].R_CenterOfMass_Average, Intervals)
            if I != -1:
                RecordsTrain.append(Records[i])
            else:
                RecordsTest.append(Records[i])
                
    Initial_train_points_number = len(RecordsTrain)
    Initial_test_points_number = len(RecordsTest)
    print('Train points number = ', Initial_train_points_number)
    print('Test points number = ', Initial_test_points_number)
    
    nTrainTotal = len(RecordsTrain)
    nTrainFinal = int(TrainFraction * nTrainTotal)
    
    RecordsTrainReduced = []
    for i in range(0, nTrainFinal, 1):
        r = random.randrange(0, len(RecordsTrain), 1)
        RecordsTrainReduced.append(RecordsTrain[r])
        del(RecordsTrain[r])
        
    for i in range(0, len(RecordsTrain), 1):
        RecordsTest.append(RecordsTrain[i])
    del(RecordsTrain)
        
    for i in range(0, len(RecordsTrainReduced), 1):
        if nMolecules == 2:
            I = library.InInterval(RecordsTrainReduced[i].R_Average, Intervals)
            if I != -1:
                nRecordsTrain[I] += 1
        elif nMolecules > 2:
            I = library.InInterval(RecordsTrainReduced[i].R_CenterOfMass_Average, Intervals)
            if I != -1:
                nRecordsTrain[I] += 1

               
    Final_train_points_number = len(RecordsTrainReduced)
    Final_test_points_number = len(RecordsTest)   
    print('Train points number = ', Final_train_points_number)
    print('Test points number = ', Final_test_points_number)
    print(nRecordsTrain)
    
    IOfunctions.store_records(F_Train, RecordsTrainReduced)
    IOfunctions.store_records(F_Test, RecordsTest)
    
    results = {'Initial dataset': F, 'Number of molecules per record': nMolecules,\
               'Intervals': Intervals, 'Molecule prototypes': MoleculePrototypes,\
               'Train set fraction': TrainFraction,\
               'Initial train points number': Initial_train_points_number,\
               'Initial test points number': Initial_test_points_number,\
               'Final train points number': Final_train_points_number,\
               'Final test points number': Final_test_points_number,\
               'Final train records number by interval': nRecordsTrain,\
               'Test Fraction for fit': TestFitFraction,\
               'Test Points number': nTestPoints,\
               'Training Set': F_Train, 'Test Set': F_Test}
    
# save results
    f = open(F_Filter, "wb")
    pickle.dump(results, f)
    f.close()
    
