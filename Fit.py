
from structure import library3
from structure.genetic3 import GA
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from multiprocessing import cpu_count
import shutil


if __name__ == '__main__':
# Global variables
    FirstAlgorithm = 'GA' # specifies algorithm that will give rough initial fit. 
    # Can be 'ENet' or 'GA' 
    UseVIP = False # if true fit will be found in two steps. First step - fit only
    # single distances, select most important (VIP) features which will be kept
# Elastic net parameters    
    L1_Single = 0.7
    eps_single = 1e-3
    n_alphas_single = 100
    L1 = 0.7
    eps = 1e-3
    n_alphas = 100
# Best Fit parameters
    MaxDepth = 100 # integer number - greater = slower but better fit
    LastIterationsToStoreSingle = 15 # Number of features when Best Fit starts
    LastIterationsToStore = 15 # Same but for double features
    UseCorrelationMatrix = False # specifies if correlation matrix will be used
    # for Best Fit algorithm. If False, all features will be used in trials 
    # to find Best Fit. Overwise, only most correlated features will be used.
    # If false, will be slow for big number of features
    MinCorr_Single = 0.8 # minimum correlation for single distances
    MinCorr = 0.9 # minimum correlation for double distances
    # needed only if UseCorrelationMatrix=True
    # used for creating list of most correlated features for Best Fit
    # float [0 .. 1]
    F_Out_Single = 'Single BF.xlsx'
    F_ENet_Single = 'Single ENet path.png'
    F_Plot_Single = 'Single Plot'
    F_Out = 'BF.xlsx'
    F_ENet = 'ENet path.png'
    F_Plot = 'Plot'
    Slope = 0.001 # stopping criterion for VIP features. Slope of RMSE graph
    VIP_number = 5 # stopping criterion for VIP features. Desired number of VIP features
# GA parameters
    GA_Method = 'Random' # How GA works. Can be 'Random' or 'p_Value'
    n_jobs = 1 # How many cores will be used by GA. -1 = all cores
    TribeSize = 100 # Population per CPU
    ChromosomeSize = 15
    if n_jobs == -1:
        nCPU = cpu_count()
    else:
        nCPU = n_jobs
    PopulationSize = TribeSize*nCPU # Total population
    MutationProbability = 0.3 # probability of mutation
    MutationInterval = [1, 3] # will be randomly chosen between min and max-1
    EliteFraction = 0.3 # fracrion of good chromosomes that will be used for crossover
    NumberOfGood = int(EliteFraction * TribeSize)
    FractionOfCrossover = 1 # int(fraction NumberOfGood / NumberOfCrossover)
    NumberOfCrossover = int(NumberOfGood / FractionOfCrossover) # number of crossover / mutation together
    CrossoverFractionInterval = [0.6, 0.4] # how many genes will be taken from first and second best chromosomes (fraction)
    IterationPerRecord = 10 # Display results of current fit after N iterations
    StopTime = 300 # How long in seconds GA works without improvement
    RandomSeed = 101
    
# Read features and structures from files stored by "Generate combined features"
    X, Y, FeaturesAll, FeaturesReduced, system = library3.ReadFeatures('Harmonic Features.csv', \
        'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', 'system.dat', verbose=False)
# split data in order to separate training set and test set
# all response variables Y must be 1D arrays
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RandomSeed)
    Y_train = Y_train.reshape(-1, 1)
    x_std = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
    y_std = scale(Y_train, axis=0, with_mean=True, with_std=False, copy=True)
    Y_train = Y_train.reshape(-1)
    y_std = y_std.reshape(-1)
    
    if UseVIP:
        if system.nAtoms == 6: # two water molecules system
            VIP_number = 5
        if system.nAtoms == 9: # three water molecules system
            VIP_number = 3
# create arrays for single distances only        
        SingleFeaturesAll = []
        for i in range(0, len(FeaturesAll), 1):
            if FeaturesAll[i].nDistances == 1:
                SingleFeaturesAll.append(FeaturesAll[i])
        count_singles = 0
        SingleFeaturesReduced = []
        for i in range(0, len(FeaturesReduced), 1):
            if FeaturesReduced[i].nDistances == 1:
                count_singles += 1
                SingleFeaturesReduced.append(FeaturesReduced[i])
        Single_X_train = np.zeros(shape=(X_train.shape[0], len(SingleFeaturesReduced)), dtype=float)
        Single_X_test = np.zeros(shape=(X_test.shape[0], len(SingleFeaturesReduced)), dtype=float)
        single_x_std = np.zeros(shape=(x_std.shape[0], len(SingleFeaturesReduced)), dtype=float)
        j = 0
        for i in range(0, len(FeaturesReduced), 1):
            if FeaturesReduced[i].nDistances == 1:
                Single_X_train[:, j] = X_train[:, i]
                Single_X_test[:, j] = X_test[:, i]
                single_x_std[:, j] = x_std[:, i]
                j += 1
        if FirstAlgorithm == 'ENet':
            enet = library3.ENet(F_ENet=F_ENet_Single, L1=0.7, eps=1e-3, nAlphas=100, alphas=None)
            t = time.time()
            print('Elastic Net Fit for single features only')
            print('L1 portion = ', enet.L1)
            print('Epsilon = ', enet.eps)
            print('Number of alphas =', enet.nAlphas)
            print('Number of features go to elastic net regularisation = ', len(SingleFeaturesReduced))
            enet.fit(single_x_std, y_std, VIP_idx=None)
            idx = enet.idx
            t_sec = time.time() - t
            print("\n", 'Elastic Net worked ', t_sec, 'sec')  
            print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
        if FirstAlgorithm == 'GA':
            ga = GA(n_jobs=n_jobs, TribeSize=TribeSize, ChromosomeSize=ChromosomeSize,\
            MutationProbability=MutationProbability, MutationInterval=MutationInterval, EliteFraction=EliteFraction,\
            FractionOfCrossover=FractionOfCrossover, CrossoverFractionInterval=CrossoverFractionInterval,\
            IterationPerRecord=IterationPerRecord, StopTime=StopTime, RandomSeed=RandomSeed, verbose=True) 
            t = time.time()
            print('Genetic Algorithm for single features only')
            ga.fit(single_x_std, y_std, Idx=None, VIP_idx=None, Method=GA_Method) 
            t_sec = time.time() - t
            print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
            idx = ga.idx 
            print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
        if UseCorrelationMatrix:
            print('MinCorrelation = ', MinCorr_Single)
        bf = library3.BF(LastIterationsToStore=LastIterationsToStoreSingle, UseCorrelationMatrix=UseCorrelationMatrix,\
                MinCorr=MinCorr_Single, F_Xlsx=F_Out_Single, F_Plot = F_Plot_Single,\
                Slope = 0.001, VIP_number=VIP_number)
        t = time.time()
        VIP_idx = bf.fit(single_x_std, y_std, Single_X_train, Y_train, Single_X_test,\
            Y_test, SingleFeaturesAll, SingleFeaturesReduced, idx=idx,\
            VIP_idx=None, Method='sequential', Criterion='MSE', GetVIP=True, Max=MaxDepth, verbose=False)
        t_sec = time.time() - t
        print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    else:
        VIP_idx = []
# proceed all features        
    if FirstAlgorithm == 'ENet':
        enet = library3.ENet(F_ENet=F_ENet, L1=0.7, eps=1e-3, nAlphas=100, alphas=None)
        t = time.time()
        print('Elastic Net Fit for all features')
        print('L1 portion = ', enet.L1)
        print('Epsilon = ', enet.eps)
        print('Number of alphas =', enet.nAlphas)
        print('Number of features go to elastic net regularisation = ', len(FeaturesReduced))
        enet.fit(x_std, y_std, VIP_idx=VIP_idx)
        idx = enet.idx
        t_sec = time.time() - t
        print("\n", 'Elastic Net worked ', t_sec, 'sec')  
        print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    if FirstAlgorithm == 'GA':
        ga = GA(n_jobs=n_jobs, TribeSize=TribeSize, ChromosomeSize=ChromosomeSize,\
        MutationProbability=MutationProbability, MutationInterval=MutationInterval, EliteFraction=EliteFraction,\
        FractionOfCrossover=FractionOfCrossover, CrossoverFractionInterval=CrossoverFractionInterval,\
        IterationPerRecord=IterationPerRecord, StopTime=StopTime, RandomSeed=RandomSeed, verbose=True) 
        t = time.time()
        print('Genetic Algorithm for all features')
        ga.fit(x_std, y_std, Idx=None, VIP_idx=VIP_idx, Method=GA_Method) 
        t_sec = time.time() - t
        print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
        idx = ga.idx 
        print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    if UseCorrelationMatrix:
        print('MinCorrelation = ', MinCorr)
    bf = library3.BF(LastIterationsToStore=LastIterationsToStore, UseCorrelationMatrix=UseCorrelationMatrix,\
            MinCorr=MinCorr, F_Xlsx=F_Out, F_Plot = F_Plot,\
            Slope = 0.001, VIP_number=None)
    t = time.time()
    bf.fit(x_std, y_std, X_train, Y_train, X_test,\
        Y_test, FeaturesAll, FeaturesReduced, idx=idx,\
        VIP_idx=VIP_idx, Method='sequential', Criterion='MSE', GetVIP=False, Max=MaxDepth, verbose=False)
    t_sec = time.time() - t
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
        
    directory = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    if not os.path.exists(directory):
        os.makedirs(directory)    
    if os.path.isfile('SystemDescriptor.'):
        shutil.copyfile('SystemDescriptor.', directory + '\\' + 'SystemDescriptor.')
    if os.path.isfile('Structure.xlsx'):    
        shutil.copyfile('Structure.xlsx', directory + '\\' + 'Structure.xlsx')
    if os.path.isfile('Harmonic Features Reduced List.xlsx'): 
        shutil.copyfile('Harmonic Features Reduced List.xlsx', directory + '\\' + 'Harmonic Features Reduced List.xlsx')
    if os.path.isfile(F_Out_Single):
        shutil.move(F_Out_Single, directory + '\\' + F_Out_Single)
    if os.path.isfile(F_ENet_Single):
        shutil.move(F_ENet_Single, directory + '\\' + F_ENet_Single)
    if os.path.isfile(F_Plot_Single + '_RMSE' + '.png'):
        shutil.move(F_Plot_Single + '_RMSE' + '.png', directory + '\\' + F_Plot_Single + '_RMSE' + '.png')
    if os.path.isfile(F_Plot_Single + '_R2' + '.png'):
        shutil.move(F_Plot_Single + '_R2' + '.png', directory + '\\' + F_Plot_Single + '_R2' + '.png')
    if os.path.isfile(F_Out):
        shutil.move(F_Out, directory + '\\' + F_Out)    
    if os.path.isfile(F_ENet):
        shutil.move(F_ENet, directory + '\\' + F_ENet)
    if os.path.isfile(F_Plot + '_RMSE' + '.png'):
        shutil.move(F_Plot + '_RMSE' + '.png', directory + '\\' + F_Plot + '_RMSE' + '.png')
    if os.path.isfile(F_Plot + '_R2' + '.png'):
        shutil.move(F_Plot + '_R2' + '.png', directory + '\\' + F_Plot + '_R2' + '.png')



