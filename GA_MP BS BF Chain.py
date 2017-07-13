# Multiprocessing Genetic algorithm + Backward Sequential + Best Fit
# Proceeds single distances, selects VIP features, than proceeds double features

import multiprocessing as mp
from structure import library3
from structure import genetic2
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import random
import copy
from joblib import Parallel, delayed
import os
import shutil

if __name__ == '__main__':
    # Global variables
    Method = 'Random'
    n_jobs = -1
    
    LastIterationsToStoreSingle = 15
    MinCorr_Single = 1e-7
    F_Out_Single = 'Single GA_MP BF.xlsx'
    F_Plot_Single = 'Single Plot'
    Slope = 0.001
    VIP_number = 5

    LastIterationsToStoreDouble = 15
    MinCorr_Double = 1e-7
    F_Out_Double = 'Double GA_MP BF.xlsx'
    F_Plot_Double = 'Double Plot'
    
    # Read features and structures from files stored by "Generate combined features"
    X, Y, FeaturesAll, FeaturesReduced, system = library3.ReadFeatures('Harmonic Features.csv', \
        'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', 'system.dat', verbose=False)
    if system.nAtoms == 6:
        VIP_number = 5
    if system.nAtoms == 9:
        VIP_number = 3
    # split data in order to separate training set and test set
    # all response variables Y must be 1D arrays
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)
    Y_train = Y_train.reshape(-1, 1)
    x_std = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
    y_std = scale(Y_train, axis=0, with_mean=True, with_std=False, copy=True)
    Y_train = Y_train.reshape(-1)
    y_std = y_std.reshape(-1)
# single distances
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
    idx = []
    for i in range(0, Single_X_train.shape[1], 1):
        idx.append(i)
    print('Genetic algorithm for single distances only')
    t = time.time()
    if n_jobs == -1:
        nCPU = mp.cpu_count()
    else:
        nCPU = n_jobs
    TribeSize = 100 # Population per CPU
    ChromosomeSize = 15
    PopulationSize = TribeSize*nCPU # Total population
    MutationProbability = 0.3 # probability of mutation
    MutationInterval = [1, 3] # will be randomly chosen between min and max-1
    EliteFraction = 0.3 # fracrion of good chromosomes that will be used for crossover
    NumberOfGood = int(EliteFraction * TribeSize)
    FractionOfCrossover = 1 # int(fraction NumberOfGood / NumberOfCrossover)
    NumberOfCrossover = int(NumberOfGood / FractionOfCrossover) # number of crossover / mutation together
    CrossoverFractionInterval = [0.6, 0.4]
    IterationPerRecord = 1
    StopTime = 600 # in seconds
    random.seed()
    Population = [] # list of chromosomes 
    # get initial population
    Population = genetic2.init_population(Population, PopulationSize, ChromosomeSize, idx, VIP_idx=None)
    # get fitness for initial population
    if Method == 'Random':
        for j in range(0, PopulationSize, 1):
            Population[j] = genetic2.get_fitness(Population[j], single_x_std, y_std)
    else:
        if Method == 'p_Value':
            for j in range(0, PopulationSize, 1):
                Population[j] = genetic2.get_fitness_pValue(Population[j], single_x_std, y_std)
    # rank initial population
    Population = genetic2.rank_population(Population)
    # get best chromosome for initial population
    BestChromosome = genetic2.get_best(Population)
    TimeLastImprovement = time.time()
    i = 0
    try: # loop will be interrupted by pressing Ctrl-C
        while (time.time() - TimeLastImprovement) < StopTime: # 1 hour without improvement than stop
            new_Population = []
            jobs = (delayed(genetic2.tribe_one_generation)(Population[m*TribeSize : (m+1)*TribeSize], NumberOfCrossover, MutationProbability, NumberOfGood, idx, MutationInterval, CrossoverFractionInterval, x_std, y_std, Method='Random', VIP_idx=None) for m in range(0, nCPU, 1))
            new_Population = Parallel(n_jobs=nCPU)(jobs)
            NewPopulation = []
            for m in range(0, nCPU, 1): # reshape population
                for n in range(0, TribeSize, 1):
                    NewPopulation.append(new_Population[m][n])
            del(new_Population)
    # rank initial population
            new_Population = genetic2.rank_population(NewPopulation)
    # get best chromosome for initial population
            BetterChromosome = genetic2.get_best(new_Population)
            if BetterChromosome.MSE < BestChromosome.MSE:
                BestChromosome = BetterChromosome
                TimeLastImprovement = time.time()
            Population = copy.deepcopy(new_Population)
            del(new_Population)
            i += 1
            if (i % IterationPerRecord) == 0:            
                print('Iteration = ', i, ' Best MSE = ', BestChromosome.MSE,\
                     'Best R2 = ', BestChromosome.R2, "\nIndices = ", [BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)],\
                     "\nTime since last improvement =", str(int((time.time() - TimeLastImprovement))), 'sec')    
        print([BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)])
    except KeyboardInterrupt: # interrupt by pressing Ctrl-C
        pass
    idx = [BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)]
    LastIterationsToStore = ChromosomeSize
    t_sec = time.time() - t
    print("\n", 'Genetic algorithm worked ', t_sec, 'sec')  
    print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    print('MinCorrelation = ', MinCorr_Single)
    print('Calculating Correlation matrix')
    C = np.cov(single_x_std, rowvar=False, bias=True)
    writeResults = pd.ExcelWriter(F_Out_Single, engine='openpyxl')
    mse_list = []
    rmse_list = []
    r2_list = []
    aIC_list = []
    nonzero_list = []
    features_list = []
    t = time.time()    
    print('Initial index set before backward elimination')
    print(idx)
    while len(idx) > 2:
        FeatureSize = len(idx)
        if FeatureSize > LastIterationsToStoreSingle:
            print('Features left: ', FeatureSize)
        idx_backward = library3.BackwardElimination(Single_X_train, Y_train, Single_X_test, Y_test,\
            FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
            N_of_features_to_select=FeatureSize-1, idx=idx, VIP_idx=None, PlotPath=False, StorePath=False, \
            FileName='Backward sequential', N_of_last_iterations_to_store=None, verbose=False)
        if len(idx_backward) > LastIterationsToStore:
            idx = idx_backward
            continue
    #    idx_corr = library2.ClassifyCorrelatedFeatures(single_x_std, idx_backward, MinCorrelation=MinCorr,\
    #        Model=1, Corr_Matrix=C, verbose=False)
        idx_corr = library3.get_full_list(idx, Single_X_train.shape[1])
        print('Features left: ', len(idx_backward))
        idx = library3.rank_features(single_x_std, y_std, idx_backward)
        idx_alternative = library3.FindBestSet(single_x_std, y_std, idx, idx_corr, VIP_idx=None, Method='MSE', verbose=True)
        library3.Results_to_xls3(writeResults, str(len(idx_alternative)),\
            idx_alternative, Single_X_train, Y_train, Single_X_test, Y_test, FeaturesAll, FeaturesReduced)
        idx = idx_alternative
        nonzero, mse, rmse, r2, aIC = library3.get_fit_score(Single_X_train, Single_X_test, Y_train, Y_test, idx=idx)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        aIC_list.append(aIC)
        nonzero_list.append(nonzero)
        features_list.append(idx)
    t_sec = time.time() - t
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    writeResults.save()
    library3.plot_rmse(F_Plot_Single, nonzero_list, rmse_list, r2_list)
    size_list = len(nonzero_list)
    for i in range(0, size_list-1, 1):
        mse = mse_list[size_list-i-1]
        mse_next =  mse_list[size_list-i-2]
        delta = mse - mse_next
        fraction = delta / mse_next
        if (fraction < 0) or (abs(fraction) < Slope) or (nonzero_list[size_list-i-2] > VIP_number):
            VIP_idx = features_list[size_list-i-1]
            if fraction < 0:
                print('VIP selection criteria: Slope became positive')
            if abs(fraction) < Slope:
                print('VIP selection criteria: Slope less than ', Slope)
            if nonzero_list[size_list-i-2] > VIP_number:
                print('VIP selection criteria: desired number of VIP features = ', VIP_number, 'has been reached')
            print('List of VIP features: ', VIP_idx)
            break
# double distances
   
    idx = []
    for i in range(0, X_train.shape[1], 1):
        idx.append(i)
    print('Genetic algorithm for all features. VIP_idx activated')
    t = time.time()
    if n_jobs == -1:
        nCPU = mp.cpu_count()
    else:
        nCPU = n_jobs
    TribeSize = 100 # Population per CPU
    ChromosomeSize = 15
    PopulationSize = TribeSize*nCPU # Total population
    MutationProbability = 0.3 # probability of mutation
    MutationInterval = [1, 3] # will be randomly chosen between min and max-1
    EliteFraction = 0.3 # fracrion of good chromosomes that will be used for crossover
    NumberOfGood = int(EliteFraction * TribeSize)
    FractionOfCrossover = 1 # int(fraction NumberOfGood / NumberOfCrossover)
    NumberOfCrossover = int(NumberOfGood / FractionOfCrossover) # number of crossover / mutation together
    CrossoverFractionInterval = [0.6, 0.4]
    IterationPerRecord = 1
    StopTime = 600 # in seconds
    random.seed()
    Population = [] # list of chromosomes 
    # get initial population
    Population = genetic2.init_population(Population, PopulationSize, ChromosomeSize, idx, VIP_idx=VIP_idx)
    # get fitness for initial population
    if Method == 'Random':
        for j in range(0, PopulationSize, 1):
            Population[j] = genetic2.get_fitness(Population[j], x_std, y_std)
    else:
        if Method == 'p_Value':
            for j in range(0, PopulationSize, 1):
                Population[j] = genetic2.get_fitness_pValue(Population[j], x_std, y_std)
    # rank initial population
    Population = genetic2.rank_population(Population)
    # get best chromosome for initial population
    BestChromosome = genetic2.get_best(Population)
    TimeLastImprovement = time.time()
    i = 0
    try: # loop will be interrupted by pressing Ctrl-C
        while (time.time() - TimeLastImprovement) < StopTime: # 1 hour without improvement than stop
            new_Population = []
            jobs = (delayed(genetic2.tribe_one_generation)(Population[m*TribeSize : (m+1)*TribeSize], NumberOfCrossover, MutationProbability, NumberOfGood, idx, MutationInterval, CrossoverFractionInterval, x_std, y_std, Method='Random', VIP_idx=VIP_idx) for m in range(0, nCPU, 1))
            new_Population = Parallel(n_jobs=nCPU)(jobs)
            NewPopulation = []
            for m in range(0, nCPU, 1): # reshape population
                for n in range(0, TribeSize, 1):
                    NewPopulation.append(new_Population[m][n])
            del(new_Population)
    # rank initial population
            new_Population = genetic2.rank_population(NewPopulation)
    # get best chromosome for initial population
            BetterChromosome = genetic2.get_best(new_Population)
            if BetterChromosome.MSE < BestChromosome.MSE:
                BestChromosome = BetterChromosome
                TimeLastImprovement = time.time()
            Population = copy.deepcopy(new_Population)
            del(new_Population)
            i += 1
            if (i % IterationPerRecord) == 0:            
                print('Iteration = ', i, ' Best MSE = ', BestChromosome.MSE,\
                     'Best R2 = ', BestChromosome.R2, "\nIndices = ", [BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)],\
                     "\nTime since last improvement =", str(int((time.time() - TimeLastImprovement))), 'sec')    
        print([BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)])
    except KeyboardInterrupt: # interrupt by pressing Ctrl-C
        pass
    idx = [BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)]
    LastIterationsToStore = ChromosomeSize
    for i in VIP_idx: # add missing VIP features 
        if i not in idx:
            idx.append(i)
    t_sec = time.time() - t
    print("\n", 'Genetic algorithm worked ', t_sec, 'sec')  
    print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    print('MinCorrelation = ', MinCorr_Double)
    print('Calculating Correlation matrix')
    C = np.cov(x_std, rowvar=False, bias=True)
    writeResults = pd.ExcelWriter(F_Out_Double, engine='openpyxl')
    mse_list = []
    rmse_list = []
    r2_list = []
    aIC_list = []
    nonzero_list = []
    features_list = []
    t = time.time()    
    print('Initial index set before backward elimination')
    print(idx)
    print('Start Backward sequential elimination')
    while len(idx) > (1+len(VIP_idx)):
        FeatureSize = len(idx)
        if FeatureSize > LastIterationsToStoreSingle:
            print('Features left: ', FeatureSize)
        idx_backward = library3.BackwardElimination(X_train, Y_train, X_test, Y_test,\
            FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
            N_of_features_to_select=FeatureSize-1, idx=idx, VIP_idx=VIP_idx, PlotPath=False, StorePath=False, \
            FileName=None, N_of_last_iterations_to_store=None, verbose=False)
        if len(idx_backward) > LastIterationsToStoreSingle:
            idx = idx_backward
            continue
        idx_corr = library3.get_full_list(idx, x_std.shape[1])
        print('Features left: ', len(idx_backward))
        idx = library3.rank_features(x_std, y_std, idx_backward)
        idx_alternative = library3.FindBestSet(x_std, y_std, idx, idx_corr, VIP_idx=VIP_idx, Method='MSE', verbose=True)
        library3.Results_to_xls3(writeResults, str(len(idx_alternative)),\
            idx_alternative, X_train, Y_train, X_test, Y_test, FeaturesAll, FeaturesReduced)
        idx = idx_alternative
        nonzero, mse, rmse, r2, aIC = library3.get_fit_score(X_train, X_test, Y_train, Y_test, idx=idx)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        aIC_list.append(aIC)
        nonzero_list.append(nonzero)
        features_list.append(idx)
    t_sec = time.time() - t
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    writeResults.save()
    library3.plot_rmse(F_Plot_Double, nonzero_list, rmse_list, r2_list)    
    directory = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    if not os.path.exists(directory):
        os.makedirs(directory)    
    try:
        shutil.copyfile('SystemDescriptor.', directory + '\\' + 'SystemDescriptor.')
        shutil.copyfile('Structure.xlsx', directory + '\\' + 'Structure.xlsx')
        shutil.copyfile('Harmonic Features Reduced List.xlsx', directory + '\\' + 'Harmonic Features Reduced List.xlsx')
        shutil.move(F_Out_Single, directory + '\\' + F_Out_Single)
        shutil.move(F_Plot_Single + '_RMSE' + '.png', directory + '\\' + F_Plot_Single + '_RMSE' + '.png')
        shutil.move(F_Plot_Single + '_R2' + '.png', directory + '\\' + F_Plot_Single + '_R2' + '.png')
        shutil.move(F_Out_Double, directory + '\\' + F_Out_Double)
        shutil.move(F_Plot_Double + '_RMSE' + '.png', directory + '\\' + F_Plot_Double + '_RMSE' + '.png')
        shutil.move(F_Plot_Double + '_R2' + '.png', directory + '\\' + F_Plot_Double + '_R2' + '.png')
    except:
        pass
    print('DONE')

