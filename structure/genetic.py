import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as skm
import statsmodels.regression.linear_model as sm
import random

class Gene:
    idx = None
    p_Value = None
    rank = None
    def __init__(self, genes):
        self.idx = genes
        
class Chromosome:
    Genes = None # list of indices of features
    MSE = None 
    R2 = None
    Rank = None
    Size = None
    def __init__(self, genes, size):
        self.Genes = genes
        self.Size = size
        
# get fit for given chromosome (MSE and R2)
# only for method Random
def get_fitness(chromosome, x_std, y_std):
    Size = x_std.shape[0]
    chromosome_size = chromosome.Size
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    x_sel = np.zeros(shape=(Size, chromosome_size), dtype=float)
# creating selected features array
    for i in range(0, chromosome_size, 1):
        idx = chromosome.Genes[i].idx
        x_sel[:, i] = x_std[:, idx] # copy selected features from initial set
    lr.fit(x_sel, y_std)
    y_pred = lr.predict(x_sel)
    mse = skm.mean_squared_error(y_std, y_pred)
    r2 = skm.r2_score(y_std, y_pred)
    chromosome.MSE = mse
    chromosome.R2 = r2
    return chromosome

# get fit for given chromosome (MSE and R2)
# for method p_Value
def get_fitness_pValue(chromosome, x_std, y_std):
    Size = x_std.shape[0]
    chromosome_size = chromosome.Size
    x_sel = np.zeros(shape=(Size, chromosome_size), dtype=float)
# creating selected features array
    for i in range(0, chromosome_size, 1):
        idx = chromosome.Genes[i].idx
        x_sel[:, i] = x_std[:, idx] # copy selected features from initial set
    ols = sm.OLS(endog = y_std, exog = x_sel, hasconst = False).fit()
    y_pred = ols.predict(x_sel)
    pvalues = ols.pvalues
    r2 = ols.rsquared
    mse = skm.mean_squared_error(y_std, y_pred)
    chromosome.MSE = mse
    chromosome.R2 = r2
    for i in range(0, chromosome_size, 1):
        chromosome.Genes[i].p_Value = pvalues[i]
    rank_chromosome(chromosome) # rank genes in chromosome according to p-Values
    return chromosome

def generate_new_chromosome(chromosome_size, idx):
# idx - list of preseleccted features from which new chromosome will be made
    Gene_list = []
    NofFeatures = len(idx)
    for i in range(0, chromosome_size, 1):
        Found = False
        while not Found:
            rand0 = random.randrange(0, NofFeatures, 1) # get random number of feature
            rand = idx[rand0]
            if rand not in Gene_list:
                Found = True
        Gene_list.append(Gene(rand))
    chromosome = Chromosome(Gene_list, chromosome_size)
    return chromosome

# mutate one gene
def mutate_one(chromosome, idx, index):
# idx - list of preseleccted features from which mutator will take one gene
    NofFeatures = len(idx)
    rand0 = random.randrange(0, NofFeatures, 1) # index of feature from initial set that will replace unfortunate gene
    rand = idx[rand0]
    chromosome.Genes[index].idx = rand # assign new gene fo chromosome
    return chromosome

# mutate more than one gene
def mutate_many(chromosome, idx, mutation_interval):
    chromosome_size = chromosome.Size
    NofFeatures = len(idx)
    rand1 = random.randrange(mutation_interval[0], mutation_interval[1], 1) # get random number mutations from interval
    for i in range(0, rand1, 1):
        rand2 = random.randrange(0, chromosome_size, 1) # index of gene in chromosome to replace
        rand3 = random.randrange(0, NofFeatures, 1) # index of feature from initial set that will replace unfortunate gene
        rand = idx[rand3]
        chromosome.Genes[rand2].idx = rand # assign new gene fo chromosome
    return chromosome

# only for p-Value method
def rank_chromosome(chromosome):
    chromosome_size = len(chromosome.Genes)
    swapped = True
    n = chromosome_size
    while swapped:
        swapped = False
        for i in range(1, n, 1):
            if chromosome.Genes[i-1].p_Value > chromosome.Genes[i].p_Value:
                swapped = True
                chromosome.Genes.insert(i-1, chromosome.Genes[i])
                del(chromosome.Genes[i+1])
        n - n - 1
# Assign ranks to chromosome
    for i in range(0, chromosome_size, 1):
        chromosome.Genes[i].rank = i
    return chromosome

def rank_population(population): # assign Chromosome.Rank for all population
# sort population list
    population_size = len(population)
    swapped = True
    n = population_size
# bubble sort from low to high
    while swapped:
        swapped = False
        for i in range(1, n, 1):
            if population[i-1].MSE > population[i].MSE:
                swapped = True
                population.insert(i-1, population[i])
                del(population[i+1])
        n - n - 1
# Assign ranks to population. Rank 0 has chromosome with lowest MSE
    for i in range(0, population_size, 1):
        population[i].Rank = i
    return population

# return the best chromosome (population must be sorted)
def get_best(population):
    return population[0]
    
def crossover_random(chromosome1, chromosome2, idx, CrossoverFractionInterval=[0.6, 0.4]): # will produce a child from two parents
    chromosome_size = chromosome1.Size
# randon number in given range [min .. max]
    rand1 = (CrossoverFractionInterval[1] - CrossoverFractionInterval[0]) * random.random() + CrossoverFractionInterval[0]
    n1 = int(chromosome_size * rand1) # number of genes to be taken from chromosome 1
    n2 = chromosome_size - n1 # number of rendom genes to be taken from chromosome 2
    Idx = [] # list of genes indices
    Genes_list = []
# append features from first chromosone
    for i in range(0, n1, 1):
        Found = False
        while not Found: # each new gene must be unique
            rand2 = random.randrange(0, chromosome_size, 1)
            if chromosome1.Genes[rand2].idx not in Idx:
                Idx.append(chromosome1.Genes[rand2].idx) # append index of gene
                Found = True
# append features from second chromosone
    for i in range(0, n2, 1):
        Found = False
        nTrials = 0
        while (not Found) and (nTrials < 10):
            rand2 = random.randrange(0, chromosome_size, 1)
            if chromosome2.Genes[rand2].idx not in Idx:
                Idx.append(chromosome2.Genes[rand2].idx)
                Found = True
            nTrials += 1
        if not Found:
            while not Found:
                rand = random.randrange(0, len(idx), 1)
                rand3 = idx[rand]
                if rand3 not in Idx:
                    Idx.append(rand3)# append new gene from set
                    Found = True
    for i in range(0, len(Idx), 1):
        Genes_list.append(Gene(Idx[i])) # create list of objects Gene
    child = Chromosome(Genes_list, chromosome_size) # make a child
    return child
    
def crossover_pValue(chromosome1, chromosome2, idx, CrossoverFractionInterval=[0.6, 0.4]): # will produce a child from two parents
# will be modified late by taking genes with the lovest p-Values
    chromosome_size = chromosome1.Size
# randon number in given range [min .. max]
    rand1 = (CrossoverFractionInterval[1] - CrossoverFractionInterval[0]) * random.random() + CrossoverFractionInterval[0]
    n1 = int(chromosome_size * rand1) # number of genes to be taken from chromosome 1
    n2 = chromosome_size - n1 # number of random genes to be taken from chromosome 2
    if n1 < n2: # swap if necessary since chromosome1 if better than chromosome2
        n1, n2 = n2, n1
    Idx = [] # list of genes indices
    Genes_list = []
# append features from first chromosone
    for i in range(0, n1, 1):
        Idx.append(chromosome1.Genes[i].idx) # append first n1 indices of genes from chromosome1
# append features from second chromosone
    nFound = 0
    i = 0
    while (nFound < n2) and (i < chromosome_size):
        if chromosome2.Genes[i].idx not in Idx:
            Idx.append(chromosome2.Genes[i].idx)
            nFound += 1
        i += 1
    if nFound < n2:
        while (nFound < n2):
            rand = random.randrange(0, len(idx), 1)
            rand3 = idx[rand]
            if rand3 not in Idx:
                Idx.append(rand3)# append new gene from set
                nFound += 1
    for i in range(0, len(Idx), 1):
        Genes_list.append(Gene(Idx[i])) # create list of objects Gene
    child = Chromosome(Genes_list, chromosome_size) # make a child
    return child

def init_population(population, population_size, chromosome_size, idx):
    for i in range(0, population_size, 1):
        population.append(generate_new_chromosome(chromosome_size, idx))
    return population

def tribe_one_generation(Tribe, NumberOfCrossover, MutationProbability, NumberOfGood, idx, MutationInterval, CrossoverFractionInterval, x_std, y_std, Method='Random'):
    new_Tribe = []
    TribeSize = len(Tribe)
    ChromosomeSize = Tribe[0].Size
    for j in range(0, NumberOfCrossover, 1):
        rand_MutationProbability = random.random()
        if rand_MutationProbability <= MutationProbability: # mutation
            rand = random.randrange(0, NumberOfGood, 1) # which chromosome to mutate
            new_Tribe.append(mutate_many(Tribe[rand], idx, MutationInterval)) # mutate one of good chromosome
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
                new_Tribe.append(crossover_random(Tribe[p1], Tribe[p2], idx, CrossoverFractionInterval))
            else:
                new_Tribe.append(crossover_pValue(Tribe[p1], Tribe[p2], idx, CrossoverFractionInterval))
# add the remaining chromosomes from feature set            
    while len(new_Tribe) < TribeSize:
        new_Tribe.append(generate_new_chromosome(ChromosomeSize, idx))
# get fitness 
    if Method == 'Random':
        for j in range(0, TribeSize, 1):
            new_Tribe[j] = get_fitness(new_Tribe[j], x_std, y_std)
    else:
        for j in range(0, TribeSize, 1):
            new_Tribe[j] = get_fitness_pValue(new_Tribe[j], x_std, y_std)

    return new_Tribe