# Genetic algorithm only
import multiprocessing as mp
from structure import library2
from structure import genetic
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import random
import copy
from joblib import Parallel, delayed

if __name__ == '__main__':
    F_Out = 'GA_MP BF.xlsx'
    # Read features and structures from files stored by "Generate combined features"
    X, Y, FeaturesAll, FeaturesReduced = library2.ReadFeatures('Harmonic Features.csv', \
        'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', verbose=False)
    # split data in order to separate training set and test set
    # all response variables Y must be 1D arrays
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)
    Y_train = Y_train.reshape(-1, 1)
    x_std = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
    y_std = scale(Y_train, axis=0, with_mean=True, with_std=False, copy=True)
    Y_train = Y_train.reshape(-1)
    y_std = y_std.reshape(-1)
    print('Least squares for full set of features')
    print('Number of features = ', X_train.shape[1])
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
    lr.fit(X_train, Y_train)
    y_pred = lr.predict(X_train)
    mse = skm.mean_squared_error(Y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y_train, y_pred)
    print('MSE= ', mse)
    print('RMSE= ', rmse)
    print('R2= ', r2)
    
    idx = []
    for i in range(0, X_train.shape[1], 1):
        idx.append(i)
        
    print('Genetic algorithm')
    # Global variables
    Method = 'Random'
    n_jobs = -1
    if n_jobs == -1:
        nCPU = mp.cpu_count()
    else:
        nCPU = n_jobs
    TribeSize = 100 # Population per CPU
    ChromosomeSize = 6 # number of features to fit data
    PopulationSize = TribeSize*nCPU # Total population
    MutationProbability = 0.3 # probability of mutation
    MutationInterval = [1, 3] # will be randomly chosen between min and max-1
    EliteFraction = 0.3 # fracrion of good chromosomes that will be used for crossover
    NumberOfGood = int(EliteFraction * TribeSize)
    FractionOfCrossover = 1 # int(fraction NumberOfGood / NumberOfCrossover)
    NumberOfCrossover = int(NumberOfGood / FractionOfCrossover) # number of crossover / mutation together
    CrossoverFractionInterval = [0.6, 0.4]
    IterationPerRecord = 1
    StopTime = 1800
    random.seed()
    Population = [] # list of chromosomes 
    
    # get initial population
    Population = genetic.init_population(Population, PopulationSize, ChromosomeSize, idx)
#    Population = genetic.init_populationM(Population, PopulationSize, ChromosomeSize, idx, x_std, y_std, C)
    # get fitness for initial population
    if Method == 'Random':
        for j in range(0, PopulationSize, 1):
            Population[j] = genetic.get_fitness(Population[j], x_std, y_std)
    else:
        if Method == 'p_Value':
            for j in range(0, PopulationSize, 1):
                Population[j] = genetic.get_fitness_pValue(Population[j], x_std, y_std)
        else:
            quit()
    # rank initial population
    Population = genetic.rank_population(Population)
    # get best chromosome for initial population
    BestChromosome = genetic.get_best(Population)
    # TimeStart = datetime.datetime.now()
    # TimeLastImprovement = datetime.datetime.now()
    TimeLastImprovement = time.time()
    i = 0
    try: # loop will be interrupted by pressing Ctrl-C
        while (time.time() - TimeLastImprovement) < StopTime: # 1 hour without improvement than stop
            new_Population = []
            jobs = (delayed(genetic.tribe_one_generation)(Population[m*TribeSize : (m+1)*TribeSize], NumberOfCrossover, MutationProbability, NumberOfGood, idx, MutationInterval, CrossoverFractionInterval, x_std, y_std, Method='Random') for m in range(0, nCPU, 1))
            new_Population = Parallel(n_jobs=nCPU)(jobs)
    # for m in range(0, nCPU, 1):
    # new_Population.append(genetic.tribe_one_generation(Population[m*TribeSize : (m+1)*TribeSize], NumberOfCrossover, MutationProbability, NumberOfGood, idx, MutationInterval, CrossoverFractionInterval, x_std, y_std, Method='Random'))
            NewPopulation = []
            for m in range(0, nCPU, 1): # reshape population
                for n in range(0, TribeSize, 1):
                    NewPopulation.append(new_Population[m][n])
            del(new_Population)
    # rank initial population
            new_Population = genetic.rank_population(NewPopulation)
    # get best chromosome for initial population
            BetterChromosome = genetic.get_best(new_Population)
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

    idx_corr = library2.get_full_list(idx, X_train.shape[1])
    writeResults = pd.ExcelWriter(F_Out, engine='openpyxl')
    idx_alternative = library2.FindBestSet(x_std, y_std, idx, idx_corr, Method='MSE', verbose=True)
    library2.Results_to_xls3(writeResults, str(len(idx_alternative)),\
        idx_alternative, X_train, Y_train, X_test, Y_test, FeaturesAll, FeaturesReduced)
    writeResults.save()
    print('DONE')
