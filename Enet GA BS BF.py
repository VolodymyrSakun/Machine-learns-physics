# Elastic Net, GA, BS + BF
import multiprocessing as mp
from structure import library2
from structure import genetic
import pandas as pd
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import random
import copy

# Global variables
F_name = 'ENet GA BS BF.xlsx'
ProcedureFileName = 'procedure.txt'
L1 = 0.7
eps = 1e-3
n_alphas = 100
# Read features and structures from files stored by "Generate combined features"
X, Y, FeaturesAll, FeaturesReduced = library2.ReadFeatures('Harmonic Features.csv', \
    'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', verbose=False)
# split data in order to separate training set and test set
# all response variables Y must be 1D arrays
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
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

idx_initial = []
for i in range(0, X_train.shape[1], 1):
    idx_initial.append(i)
    
print('Elastic Net Fit')
print('L1 portion = ', L1)
print('Epsilon = ', eps)
print('Number of alphas =', n_alphas)
print('Number of features go to elastic net regularisation = ', len(FeaturesReduced))
alphas = np.logspace(-3, -6, num=100, endpoint=True, base=10.0)
t = time.time()
# get elastic net path
# select the best subset from elastic net path
# THIS SET WORKS
idx, alpha, mse, nonz = library2.SelectBestSubsetFromElasticNetPath(x_std, y_std,\
    Method='grid', MSE_threshold=None, R2_threshold=None, L1_ratio=L1, Eps=eps,\
    N_alphas=n_alphas, Alphas=None, max_iter=10000, tol=0.0001, cv=None, n_jobs=1, \
    selection='random', PlotPath=True, verbose=True)
t_sec = time.time() - t
print("\n", 'Elastic Net worked ', t_sec, 'sec')

idx = idx_initial # use initial set

print('Genetic algorithm')
# Global variables
Method = 'Random'
nCPU = mp.cpu_count()
ChromosomeSize = 15 # number of features to fit data
PopulationSize = 100*nCPU # number of parents in population
MutationProbability = 0.3 # probability of mutation
MutationInterval = [1, 3] # will be randomly chosen between min and max-1
EliteFraction = 0.3 # fracrion of good chromosomes that will be used for crossover
NumberOfGood = int(EliteFraction * PopulationSize)
FractionOfCrossover = 1 # int(fraction NumberOfGood / NumberOfCrossover)
NumberOfCrossover = int(NumberOfGood / FractionOfCrossover) # number of crossover / mutation together
CrossoverFractionInterval = [0.6, 0.4]
IterationPerRecord = 10
random.seed()
Population = [] # list of chromosomes 

# get initial population
Population = genetic.init_population(Population, PopulationSize, ChromosomeSize, idx)
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
    while (time.time() - TimeLastImprovement) < 1800: # 1 hour without improvement than stop
        new_Population = []
        for j in range(0, NumberOfCrossover):
            rand_MutationProbability = random.random()
            if rand_MutationProbability <= MutationProbability: # mutation
                rand = random.randrange(0, NumberOfGood, 1) # which chromosome to mutate
                new_Population.append(genetic.mutate_many(Population[rand], idx, MutationInterval)) # mutate one of good chromosome
            else: # crossover 
                p1 = rand = random.randrange(0, NumberOfGood, 1) # chose first chromosome for crossover
                p2 = rand = random.randrange(0, NumberOfGood, 1) # chose second chromosome for crossover
                if p1 > p2: # swap if necessary since chromosome1 if better than chromosome2
                    p1, p2 = p2, p1
                if p2 == p1: # if the same chromosome try to get another one
                    k = 0
                    while p1 == p2 and (k < 100): # finish later
                        p2 = rand = random.randrange(0, NumberOfGood, 1)
                        k += 1
                if Method == 'Random':
                    new_Population.append(genetic.crossover_random(Population[p1], Population[p2], idx, CrossoverFractionInterval=CrossoverFractionInterval))
                else:
                    new_Population.append(genetic.crossover_pValue(Population[p1], Population[p2], idx, CrossoverFractionInterval=CrossoverFractionInterval))
# add the remaining chromosomes from feature set            
        for j in range(len(new_Population), PopulationSize, 1):
            new_Population.append(genetic.generate_new_chromosome(ChromosomeSize, idx))
# get fitness for initial population
        for j in range(0, PopulationSize, 1):
            new_Population[j] = genetic.get_fitness(new_Population[j], x_std, y_std)
# rank initial population
        new_Population = genetic.rank_population(new_Population)
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

LastIterationsToStore = ChromosomeSize
MinCorr = 1e-7
print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
print('MinCorrelation = ', MinCorr)
print('Correlation matrix')
C = np.cov(x_std, rowvar=False, bias=True)
writeResults = pd.ExcelWriter(F_name, engine='openpyxl')
mse_list = []
rmse_list = []
r2_list = []
aIC_list = []
nonzero_list = []
t = time.time()        
while len(idx) > 2:
    print(len(idx))
    FeatureSize = len(idx)
    idx_backward = library2.BackwardElimination(X_train, Y_train, X_test, Y_test,\
        FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
        N_of_features_to_select=FeatureSize-1, idx=idx, PlotPath=False, StorePath=False, \
        FileName='BS', N_of_last_iterations_to_store=None, verbose=False)
    if len(idx_backward) > LastIterationsToStore:
        idx = idx_backward
        continue
    idx_corr = library2.ClassifyCorrelatedFeatures(x_std, idx_backward, MinCorrelation=MinCorr,\
        Model=1, Corr_Matrix=C, verbose=False)
    idx_alternative = library2.FindBestSet(x_std, y_std, idx_backward, idx_corr, Method='MSE', verbose=True)
    library2.Results_to_xls3(writeResults, str(len(idx_alternative)),\
        idx_alternative, X_train, Y_train, X_test, Y_test, FeaturesAll, FeaturesReduced)
    idx = idx_alternative
    nonzero, mse, rmse, r2, aIC = library2.get_fit_score(X_train, X_test, Y_train, Y_test, idx=idx)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)
    aIC_list.append(aIC)
    nonzero_list.append(nonzero)
t_sec = time.time() - t
print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
writeResults.save()
library2.plot_rmse("Results", nonzero_list, rmse_list, r2_list)
# return output back to console
# RedirectPrintToConsole(f_new, f_old)








        
        
        
        
        
        
        



