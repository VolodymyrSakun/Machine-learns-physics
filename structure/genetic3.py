# Class GA (genetic algorithm)
# able to fit multiple linear regression with high multicollinearity
# classes:
# Gene
# Chromosome
# functions:
# get_fitness
# get_fitness_pValue
# generate_new_chromosome
# mutate_one
# mutate_many
# rank_chromosome - inactive
# rank_population
# get_best
# crossover_random
# crossover_pValue
# init_population
# tribe_one_generation
# fit

import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.metrics as skm
import statsmodels.regression.linear_model as sm
import random
from structure import library2
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from time import time
from copy import deepcopy

class GA:
    idx = None
    n_jobs = -1 # How many cores will be used by GA. -1 = all cores
    TribeSize = 100 # Population per CPU
    ChromosomeSize = 15
    nCPU = 1
    PopulationSize = None # Total population
    MutationProbability = 0.3 # probability of mutation
    MutationInterval = [1, 3] # will be randomly chosen between min and max-1
    EliteFraction = 0.3 # fracrion of good chromosomes that will be used for crossover
    NumberOfGood = None # how many genes considered to be good
    FractionOfCrossover = 1 # int(fraction NumberOfGood / NumberOfCrossover)
    NumberOfCrossover = None # number of crossover / mutation together
    CrossoverFractionInterval = [0.6, 0.4]
    IterationPerRecord = 1
    StopTime = 600 # in seconds
    verbose = True    
    RandomSeed = None
    
    class Gene:
        idx = None
        p_Value = None
        rank = None
        def __init__(self, gene):
            self.idx = gene
            
    class Chromosome:
        Genes = None # list of indices of features
        MSE = None 
        R2 = None
        Rank = None
        Size = None
        def __init__(self, genes, size):
            self.Genes = genes
            self.Size = size

    def __init__(self, n_jobs=-1, TribeSize=100, ChromosomeSize=15,\
            MutationProbability=0.3, MutationInterval=[1,3], EliteFraction=0.3,\
            FractionOfCrossover=1, CrossoverFractionInterval=[0.6,0.4],\
            IterationPerRecord=1, StopTime=600, RandomSeed=None, verbose=True):
        self.n_jobs = n_jobs
        if self.n_jobs == -1:
            self.nCPU = cpu_count()
        else:
            self.nCPU = self.n_jobs
        self.TribeSize = TribeSize
        self.ChromosomeSize = ChromosomeSize
        self.PopulationSize = self.TribeSize*self.nCPU
        self.MutationProbability = MutationProbability
        self.MutationInterval = MutationInterval
        self.EliteFraction = EliteFraction
        self.NumberOfGood = int(self.EliteFraction * self.TribeSize)
        self.FractionOfCrossover = FractionOfCrossover
        self.NumberOfCrossover = int(self.NumberOfGood / self.FractionOfCrossover)
        self.CrossoverFractionInterval = CrossoverFractionInterval
        self.IterationPerRecord = IterationPerRecord
        self.StopTime = StopTime
        self.verbose = verbose
        self.RandomSeed = RandomSeed
        return

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
        chromosome = GA.rank_chromosome(chromosome) # rank genes in chromosome according to p-Values
        return chromosome
    
    def generate_new_chromosome(chromosome_size, idx, VIP_idx=None):
# idx - list of preseleccted features from which new chromosome will be made
        if VIP_idx is None:
            VIP_idx = []
        Gene_list = []
        NofFeatures = len(idx)
        for i in range(0, len(VIP_idx), 1):
            Gene_list.append(GA.Gene(VIP_idx[i]))
        for i in range(len(VIP_idx), chromosome_size, 1):
            Found = False
            while not Found:
                rand0 = random.randrange(0, NofFeatures, 1) # get random number of feature
                rand = idx[rand0]
                if rand not in Gene_list:
                    Found = True
            Gene_list.append(GA.Gene(rand))
        chromosome = GA.Chromosome(Gene_list, chromosome_size)
        return chromosome
       
# mutate one gene
    def mutate_one(chromosome, idx, index, VIP_idx=None):
        if VIP_idx is None:
            VIP_idx = []    
# idx - list of preseleccted features from which mutator will take one gene
        NofFeatures = len(idx)
        rand0 = random.randrange(0, NofFeatures, 1) # index of feature from initial set that will replace unfortunate gene
        rand = idx[rand0] # global
        # index - local
        if chromosome.Genes[index].idx not in VIP_idx:
            chromosome.Genes[index].idx = rand # assign new gene fo chromosome
        return chromosome
    
# mutate more than one gene
    def mutate_many(chromosome, idx, mutation_interval, VIP_idx=None):
        if VIP_idx is None:
            VIP_idx = [] 
        chromosome_size = chromosome.Size
        NofFeatures = len(idx)
        rand1 = random.randrange(mutation_interval[0], mutation_interval[1], 1) # get random number mutations from interval
        for i in range(0, rand1, 1):
            rand2 = random.randrange(0, chromosome_size, 1) # index of gene in chromosome to replace
            rand3 = random.randrange(0, NofFeatures, 1) # index of feature from initial set that will replace unfortunate gene
            rand = idx[rand3]
            if chromosome.Genes[rand2].idx not in VIP_idx:
                chromosome.Genes[rand2].idx = rand # assign new gene fo chromosome
        return chromosome
    
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
    
# inactive
    def set_fitness_for_genes(chromosome, x_std, y_std):
        chromosome_size = chromosome.Size
        Size = x_std.shape[0]
        z = np.zeros(shape=(Size, 1), dtype=float)
        tmp = np.zeros(shape=(Size, 1), dtype=float)
        lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
        x_sel = np.zeros(shape=(Size, chromosome_size), dtype=float)
# creating selected features array
        for i in range(0, chromosome_size, 1):
            idx = chromosome.Genes[i].idx
            x_sel[:, i] = x_std[:, idx] # copy selected features from initial set
        lr.fit(x_sel, y_std)
        y_pred = lr.predict(x_sel)
        mse_full = skm.mean_squared_error(y_std, y_pred)
        for i in range(0, chromosome_size, 1):
            tmp[:, 0] = x_sel[:, i] # save column
            x_sel[:, i] = z[:, 0] # copy zeros to column
            lr.fit(x_sel, y_std)
            y_pred = lr.predict(x_sel)
            mse = skm.mean_squared_error(y_std, y_pred)
            drop = mse - mse_full # changes of mse after dropping one gene
            chromosome.Genes[i].p_Value = drop
            x_sel[:, i] = tmp[:, 0] # restore column
        return chromosome

# assign Chromosome.Rank for all population    
    def rank_population(population): 
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
        
    def crossover_random(chromosome1, chromosome2, idx, CrossoverFractionInterval=[0.6, 0.4], VIP_idx=None): 
# will produce a child from two parents
        if VIP_idx is None:
            VIP_idx = []
        Idx = [] # list of genes indices
        Genes_list = []
        for i in VIP_idx:
            Genes_list.append(GA.Gene(i)) # copy VIP genes
        chromosome_size = chromosome1.Size
        size_left = chromosome_size - len(VIP_idx)
# randon number in given range [min .. max]
        rand1 = (CrossoverFractionInterval[1] - CrossoverFractionInterval[0]) * random.random() + CrossoverFractionInterval[0]
        n1 = int(size_left * rand1) # number of genes to be taken from chromosome 1
        n2 = chromosome_size - n1 -len(VIP_idx) # number of rendom genes to be taken from chromosome 2
# append features from first chromosone
        for i in range(0, n1, 1):
            Found = False
            while not Found: # each new gene must be unique
                rand2 = random.randrange(0, chromosome_size, 1)
                if (chromosome1.Genes[rand2].idx not in Idx) and (chromosome1.Genes[rand2].idx not in VIP_idx):
                    Idx.append(chromosome1.Genes[rand2].idx) # append index of gene
                    Found = True
# append features from second chromosone
        for i in range(0, n2, 1):
            Found = False
            nTrials = 0
            while (not Found) and (nTrials < 10):
                rand2 = random.randrange(0, chromosome_size, 1)
                if (chromosome2.Genes[rand2].idx not in Idx) and (chromosome2.Genes[rand2].idx not in VIP_idx):
                    Idx.append(chromosome2.Genes[rand2].idx)
                    Found = True
                nTrials += 1
            if not Found:
                while not Found:
                    rand = random.randrange(0, len(idx), 1)
                    rand3 = idx[rand]
                    if (rand3 not in Idx) and (rand3 not in VIP_idx):
                        Idx.append(rand3)# append new gene from set
                        Found = True
        for i in range(0, len(Idx), 1):
            Genes_list.append(GA.Gene(Idx[i])) # create list of objects Gene
        child = GA.Chromosome(Genes_list, chromosome_size) # make a child
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
            Genes_list.append(GA.Gene(Idx[i])) # create list of objects Gene
        child = GA.Chromosome(Genes_list, chromosome_size) # make a child
        return child
    
    def init_population(population, population_size, chromosome_size, idx, VIP_idx=None):
        for i in range(0, population_size, 1):
            population.append(GA.generate_new_chromosome(chromosome_size, idx, VIP_idx=VIP_idx))
        return population
    
    def tribe_one_generation(Tribe, NumberOfCrossover, MutationProbability, NumberOfGood, \
                             idx, MutationInterval, CrossoverFractionInterval, x_std, y_std,\
                             Method='Random', VIP_idx=None):
        new_Tribe = []
        TribeSize = len(Tribe)
        ChromosomeSize = Tribe[0].Size
        for j in range(0, NumberOfCrossover, 1):
            rand_MutationProbability = random.random()
            if rand_MutationProbability <= MutationProbability: # mutation
                rand = random.randrange(0, NumberOfGood, 1) # which chromosome to mutate
                new_Tribe.append(GA.mutate_many(Tribe[rand], idx, MutationInterval)) # mutate one of good chromosome
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
                    new_Tribe.append(GA.crossover_random(Tribe[p1], Tribe[p2], idx, CrossoverFractionInterval, VIP_idx=VIP_idx))
                else:
                    new_Tribe.append(GA.crossover_pValue(Tribe[p1], Tribe[p2], idx, CrossoverFractionInterval))
    # add the remaining chromosomes from feature set            
        while len(new_Tribe) < TribeSize:
            new_Tribe.append(GA.generate_new_chromosome(ChromosomeSize, idx, VIP_idx=VIP_idx))
    # get fitness 
        if Method == 'Random':
            for j in range(0, TribeSize, 1):
                new_Tribe[j] = GA.get_fitness(new_Tribe[j], x_std, y_std)
        else:
            for j in range(0, TribeSize, 1):
                new_Tribe[j] = GA.get_fitness_pValue(new_Tribe[j], x_std, y_std)
    
        return new_Tribe
    
    def fit(self, x, y, Idx=None, VIP_idx=None, Method='Random'):
        # Method - How GA works. Can be 'Random' or 'p_Value'
        if VIP_idx is None:
            VIP_idx = []
        if Idx is None:
            Idx = []
            for i in range(0, x.shape[1], 1):
                Idx.append(i)
        if self.RandomSeed is None:
            random.seed()
        else:
            random.seed(self.RandomSeed)
        Population = [] # list of chromosomes 
        # get initial population
        Population = GA.init_population(Population, self.PopulationSize,\
            self.ChromosomeSize, Idx, VIP_idx=VIP_idx)
        # get fitness for initial population
        if Method == 'Random':
            for j in range(0, self.PopulationSize, 1):
                Population[j] = GA.get_fitness(Population[j], x, y)
        else:
            if Method == 'p_Value':
                for j in range(0, self.PopulationSize, 1):
                    Population[j] = GA.get_fitness_pValue(Population[j], x, y)
        # rank initial population
        Population = GA.rank_population(Population)
        # get best chromosome for initial population
        BestChromosome = GA.get_best(Population)
        TimeLastImprovement = time()
        i = 0
        try: # loop will be interrupted by pressing Ctrl-C
            while (time() - TimeLastImprovement) < self.StopTime:
                if self.nCPU != 1:
                    new_Population = []
                    jobs = (delayed(GA.tribe_one_generation)\
                        (Population[m*self.TribeSize : (m+1)*self.TribeSize],\
                        self.NumberOfCrossover, self.MutationProbability,\
                        self.NumberOfGood, Idx, self.MutationInterval,\
                        self.CrossoverFractionInterval, x, y, Method=Method,\
                        VIP_idx=VIP_idx) for m in range(0, self.nCPU, 1))
                    new_Population = Parallel(n_jobs=self.nCPU)(jobs)
                    NewPopulation = []
                    for m in range(0, self.nCPU, 1): # reshape population
                        for n in range(0, self.TribeSize, 1):
                            NewPopulation.append(new_Population[m][n])
                    del(new_Population)
                else:
                    NewPopulation = GA.tribe_one_generation(Population,\
                        self.NumberOfCrossover, self.MutationProbability,\
                        self.NumberOfGood, Idx, self.MutationInterval,\
                        self.CrossoverFractionInterval, x, y, Method=Method,\
                        VIP_idx=VIP_idx)
        # rank initial population
                new_Population = GA.rank_population(NewPopulation)
        # get best chromosome for initial population
                BetterChromosome = GA.get_best(new_Population)
                if BetterChromosome.MSE < BestChromosome.MSE:
                    BestChromosome = BetterChromosome
                    TimeLastImprovement = time()
                Population = deepcopy(new_Population)
                del(new_Population)
                i += 1
                if self.verbose and ((i % self.IterationPerRecord) == 0):            
                    print('Iteration = ', i, ' Best MSE = ', BestChromosome.MSE,\
                        'Best R2 = ', BestChromosome.R2, "\nIndices = ",\
                        [BestChromosome.Genes[j].idx for j in\
                        range(0, len(BestChromosome.Genes), 1)],\
                        "\nTime since last improvement =",\
                        str(int((time() - TimeLastImprovement))), 'sec')    
            if self.verbose:
                print([BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)])
        except KeyboardInterrupt: # interrupt by pressing Ctrl-C
            pass
        idx = [BestChromosome.Genes[j].idx for j in range(0, len(BestChromosome.Genes), 1)]
        self.idx = idx
        return

        
