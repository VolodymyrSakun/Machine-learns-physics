# Genetic algorithm

from structure import library1
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import numpy as np
import random
import copy
import pandas as pd
import datetime

# Read features and structures from files stored by "Generate combined features"
X, Y, FeaturesAll, FeaturesReduced = library1.ReadFeatures('Features and energy two distances reduced.csv', \
    'FeaturesAll.dat', 'FeaturesReduced.dat', verbose=False)
# split data in order to separate training set and test set
# all response variables Y must be 1D arrays
Y = Y.reshape(-1, 1)
x_std = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
y_std = scale(Y, axis=0, with_mean=True, with_std=False, copy=True)
Y = Y.reshape(-1)
y_std = y_std.reshape(-1)
Size = x_std.shape[0]
NofFeatures =len(FeaturesReduced)

ChromosomeSize = 10 # number of features to fit data
PopulationSize = 80 # number of parents in population
MutationProbability = 0.2 # probability of mutation
MutationInterval = [1, 2] # will be randomly chosen between min and max-1
EliteFraction = 0.2 # fracrion of good chromosomes that will be used for crossover
CrossoverFractionInterval = [0.7, 0.3]
FractionOfCrossover = 1 # int(fraction NumberOfGood / NumberOfCrossover)
# NumberOfCrossover = int(NumberOfGood / FractionOfCrossover)
IterationPerRecord = 10
MinCorr = 0.8 # for ClassifyCorrelatedFeatures. Smalle value gives better result but works longer
FileName_xlsx = "Genetic plus alternative. Variables=" + str(ChromosomeSize) + '.xlsx' # store .xlsx

random.seed()
Population = [] # list of chromosomes 

class Chromosome:
    Genes = None # list of indices of features
    MSE = None 
    R2 = None
    Rank = None
    
    def __init__(self, genes):
        self.Genes = genes
        
# get fit for given chromosome (MSE and R2)
def get_fitness(chromosome):
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    chromosome.Genes
    x_sel = np.zeros(shape=(Size, ChromosomeSize), dtype=float)
# creating selected features array
    j = 0
    for i in chromosome.Genes:
        x_sel[:, j] = x_std[:, i] # copy selected features from initial set
        j += 1
    lr.fit(x_sel, y_std)
    y_pred = lr.predict(x_sel)
    mse = skm.mean_squared_error(y_std, y_pred)
    r2 = skm.r2_score(y_std, y_pred)
    chromosome.MSE = mse
    chromosome.R2 = r2
    return chromosome
    
def generate_new_chromosome():
    idx = []
    for i in range(0, ChromosomeSize, 1):
         rand = random.randrange(0, NofFeatures, 1) # get random number of feature
         idx.append(rand)
    chromosome = Chromosome(idx)
    return chromosome
    
def mutate(chromosome):
    rand1 = random.randrange(MutationInterval[0], MutationInterval[1], 1) # get random number mutations from interval
    for i in range(0, rand1, 1):
        rand2 = random.randrange(0, ChromosomeSize, 1) # index of gene in chromosome to replace
        rand3 = random.randrange(0, NofFeatures, 1) # index of feature from initial set that will replace unfortunate gene
        chromosome.Genes[rand2] = rand3 # assign new gene fo chromosome
    return chromosome
    
def display(best_chromosome):
    print('Best solution has MSE = ', best_chromosome.MSE, ' R2 = ', best_chromosome.R2)
    return

def rank(population): # assign Chromosome.Rank for all population
# sort population list
    swapped = True
    n = PopulationSize
    while swapped:
        swapped = False
        for i in range(1, n, 1):
            if population[i-1].MSE > population[i].MSE:
                swapped = True
                population.insert(i-1, population[i])
                del(population[i+1])
        n - n - 1
# Assign ranks to population
    for i in range(0, PopulationSize, 1):
        population[i].Rank = i
    return population

# return best chromosome (population must be sorted)
def get_best(population):
    return population[0]
    
def crossover(chromosome1, chromosome2): # will produce a child from two parents
# will be modified late by taking genes with the lovest p-Values
    CrossoverFractionInterval
# randon number in given range [min .. max]
    rand1 = (CrossoverFractionInterval[1] - CrossoverFractionInterval[0]) * random.random() + CrossoverFractionInterval[0]
    n1 = int(ChromosomeSize * rand1) # number of genes to be taken from chromosome 1
    n2 = ChromosomeSize - n1 # number of rendom genes to be taken from chromosome 2
    idx = []
# append features from first chromosone
    for i in range(0, n1, 1):
        rand2 = random.randrange(0, ChromosomeSize, 1)
        if chromosome1.Genes[rand2] not in idx:
            idx.append(chromosome1.Genes[rand2])
        else:
            k = 0
            while (chromosome1.Genes[rand2] in idx) and (k < 100):
                rand2 = random.randrange(0, ChromosomeSize, 1)
                k += 1
            idx.append(chromosome1.Genes[rand2])
# append features from second chromosone
    for i in range(0, n2, 1):
        rand2 = random.randrange(0, ChromosomeSize, 1)
        if chromosome2.Genes[rand2] not in idx:
            idx.append(chromosome2.Genes[rand2])
        else:
            k = 0
            while (chromosome2.Genes[rand2] in idx) and (k < 100):
                rand2 = random.randrange(0, ChromosomeSize, 1)
                k += 1
            idx.append(chromosome2.Genes[rand2])
    child = Chromosome(idx)    
    return child
    
def init_population(population):
    for i in range(0, PopulationSize, 1):
        population.append(generate_new_chromosome())
    return population

# get initial population
Population = init_population(Population)
# get fitness for initial population
for j in range(0, PopulationSize, 1):
    Population[j] = get_fitness(Population[j])
# rank initial population
Population = rank(Population)
# get best chromosome for initial population
BestChromosome = get_best(Population)
TimeStart = datetime.datetime.now()
TimeLastImprovement = datetime.datetime.now()
i = 0
try: # loop will be interrupted by pressing Ctrl-C
    while True:
        new_Population = []
        NumberOfGood = int(EliteFraction * PopulationSize)
        NumberOfCrossover = int(NumberOfGood / FractionOfCrossover) # number of crossover / mutation together
        for j in range(0, NumberOfCrossover):
            rand_MutationProbability = random.random()
            if rand_MutationProbability <= MutationProbability: # mutation
                rand = random.randrange(0, NumberOfGood, 1) # which chromosome to mutate
                new_Population.append(mutate(Population[rand])) # mutate one of good chromosome
            else: # crossover
                p1 = rand = random.randrange(0, NumberOfGood, 1) # chose first chromosome for crossover
                p2 = rand = random.randrange(0, NumberOfGood, 1) # chose second chromosome for crossover
                if p2 == p1: # if the same chromosome try to get another one
                    k = 0
                    while p1 == p2 and (k < 100):
                        p2 = rand = random.randrange(0, NumberOfGood, 1)
                        k += 1
                new_Population.append(crossover(Population[p1], Population[p2]))
# add the remaining chromosomes from feature set            
        for j in range(len(new_Population), PopulationSize, 1):
            new_Population.append(generate_new_chromosome())
# get fitness for initial population
        for j in range(0, PopulationSize, 1):
            new_Population[j] = get_fitness(new_Population[j])
# rank initial population
        new_Population = rank(new_Population)
# get best chromosome for initial population
        BetterChromosome = get_best(new_Population)
        if BetterChromosome.MSE < BestChromosome.MSE:
            BestChromosome = BetterChromosome
            TimeLastImprovement = datetime.datetime.now()
        Population = copy.deepcopy(new_Population)
        del(new_Population)
        i += 1
        if (i % IterationPerRecord) == 0:            
            print('Iteration = ', i, ' Best MSE = ', BestChromosome.MSE,\
                 'Best R2 = ', BestChromosome.R2, ' Indices = ', BestChromosome.Genes,\
                 'Time since last improvement = ', str(datetime.datetime.now() - TimeLastImprovement))
    print(BestChromosome.Genes)
except KeyboardInterrupt: # interrupt in the morning by pressing Ctrl-C
    pass
# store results
writeResults = pd.ExcelWriter(FileName_xlsx, engine='openpyxl')
print('Storing results from Genetic algorithm')
library1.Results_to_xls(writeResults, str(len(BestChromosome.Genes)),\
    BestChromosome.Genes, X, Y, FeaturesAll, FeaturesReduced)
print('Calculating correlation matrix')
idx_corr = library1.ClassifyCorrelatedFeatures(X, BestChromosome.Genes, MinCorrelation=MinCorr,\
    Model=1, Corr_Matrix=None, verbose=False)
print('Searching alternate solution')
idx_alternative, res = library1.FindAlternativeFit(X, Y, BestChromosome.Genes,\
    idx_corr, Method='MSE', verbose=False)
print('Storing results from Alternate solution')
library1.Results_to_xls(writeResults, str(len(BestChromosome.Genes)) + " Adj",\
    idx_alternative, X, Y, FeaturesAll, FeaturesReduced)

writeResults.save()    
    
print('DONE')
    
    