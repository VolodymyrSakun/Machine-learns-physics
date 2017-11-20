import sys
import random
import numpy as np
from project1 import library
from project1 import regression
from time import time
import copy
import matplotlib.pyplot as plt
import pandas as pd

class Gene:
    
    def __init__(self, Idx, Type=0, p_Value=None, rank=None):
        self.Idx = Idx # global index
        self.Type = Type # 0 - linear, 1 - exponential  
        self.p_Value = p_Value
        self.Rank = rank # highest - most important
        self.Coef1 = None # linear (gamma) coefficient or exponential coefficient in power of exp (alpha)
        self.Coef2 = None # None or coefficiant in front of exponent (beta)
        
    def is_gene_exists(self, chromosome): # returns true if gene exists in Gene_list
        if chromosome is None:
            return False
        if chromosome.Genes is None:
            return False
        for i in chromosome.Genes:
            if (i.Idx == self.Idx) and (i.Type == self.Type):
                return True
        return False
    
    def print_gene(self):
        print('Index = ', self.Idx)
        if self.Type == 0:
            print('Type: Linear')
        if self.Type == 1:
            print('Type: Exponential')
        print('p-Value = ', self.p_Value)
        print('Rank = ', self.Rank)
        return

############################################################# end of class Gene

class Chromosome:
    
    def __init__(self, genes):
        self.Genes = genes # list of Genes
        self.Size = len(self.Genes)
        self.MSE_Train = None 
        self.RMSE_Train = None
        self.R2_Train = None
        self.R2Adj_Train = None
        self.MSE_Test = None 
        self.RMSE_Test = None
        self.R2_Test = None
        self.R2Adj_Test = None
        self.Rank = None
        self.Origin = None
        self.time = None            
        return

# return list of indices of specified type            
    def get_genes_list(self, Type=0):
        idx = []
        for i in range(0, self.Size, 1):
            if self.Genes[i].Type == Type: 
                idx.append(self.Genes[i].Idx)
        return idx

# return list of indices of specified type            
    def get_coeff_list(self, Type=0):
        coef = []
        for i in range(0, self.Size, 1):
            if self.Genes[i].Type != Type:
                continue
            coef.append(self.Genes[i].Coef1)
            if self.Genes[i].Type == 1: # nonlinear
                coef.append(self.Genes[i].Coef2)
        return coef
    
    def find_gene_idx(self, gene): # returns gene index in chromosome
        if self.Genes is None:
            return None
        for i in range(0, self.Size, 1) :
            if (self.Genes[i].Idx == gene.Idx) and (self.Genes[i].Type == gene.Type):
                return i
        return None

    def erase_score(self):
        self.MSE_Train = None 
        self.RMSE_Train = None
        self.R2_Train = None
        self.R2Adj_Train = None
        self.MSE_Test = None 
        self.RMSE_Test = None
        self.R2_Test = None
        self.R2Adj_Test = None
        self.Size = len(self.Genes)
        self.Origin = None
        self.time = None
        return
        
    def is_exist(self, Chromosome_list):
        if Chromosome_list is None:
            return False
        if type(Chromosome_list) is Chromosome:
            Chromosome_list = [Chromosome_list]
        idx_lin0 = self.get_genes_list(Type=0)
        idx_lin0.sort()
        idx_nonlin0 = self.get_genes_list(Type=1)
        idx_nonlin0.sort()
        for i in Chromosome_list:
            idx_lin1 = i.get_genes_list(Type=0)
            idx_lin1.sort()
            idx_nonlin1 = i.get_genes_list(Type=1)
            idx_nonlin1.sort()
            if (idx_lin0 == idx_lin1) and (idx_nonlin0 == idx_nonlin1):
                return True
        return
    def has_duplicates(self):
        if self.Size == 0:
            return False
        idx_lin = self.get_genes_list(Type=0)
        idx_nonlin = self.get_genes_list(Type=1)
        dup_lin = [x for x in idx_lin if idx_lin.count(x) > 1]
        dup_nonlin = [x for x in idx_nonlin if idx_nonlin.count(x) >= 2]
        if len(dup_lin) == 0 and len(dup_nonlin) == 0:
            return False
        print('Chash: Found duplicates in chromosome')
        return True
        
# calculate Coefficients, MSE, R2, R2 adjusted
    def score(self, x_expD_train=None, x_expDn_train=None, x_lin_train=None,\
            y_train=None, x_expD_test=None, x_expDn_test=None, x_lin_test=None,\
            y_test=None, LinearSolver='sklearn', cond=1e-20, lapack_driver='gelsy'):
        idx_lin = self.get_genes_list(Type=0) # returns [] or list
        idx_exp = self.get_genes_list(Type=1)
        if (len(idx_exp) != 0) and (x_expD_train is not None) and (x_expDn_train is not None): # there are nonlinear indices
            fit_results = regression.fit_exp(idx_exp, idx_lin,\
                x_expD_train, x_expDn_train, x_lin_train, y_train,\
                x_expD_test=x_expD_test, x_expDn_test=x_expDn_test,\
                x_lin_test=x_lin_test, y_test=y_test)
            if fit_results['Success']:
                for i in range(0, self.Size, 1):
                    idx = self.Genes[i].Idx
                    if self.Genes[i].Type == 1:# exp
                        self.Genes[i].Coef1 = fit_results['Coefficients exponential'][idx]['Coefficient'][0]
                        self.Genes[i].Coef2 = fit_results['Coefficients exponential'][idx]['Coefficient'][1]
                    if self.Genes[i].Type == 0:# linear
                        self.Genes[i].Coef1 = fit_results['Coefficients linear'][idx]['Coefficient']                        
        else: # only linear indices
            if (len(idx_lin) != 0) and (x_lin_train is not None): 
                fit_results = regression.fit_linear(idx_lin, x_lin_train,\
                    y_train, x_test=x_lin_test, y_test=y_test,\
                    normalize=True, LinearSolver=LinearSolver,\
                    cond=cond, lapack_driver=lapack_driver)
                for i in range(0, self.Size, 1):
                    idx = self.Genes[i].Idx
                    if self.Genes[i].Type == 0:# linear
                        self.Genes[i].Coef1 = fit_results['Coefficients'][idx]['Coefficient'] 
                        if fit_results['Coefficients'].shape[0] == 2:
                            self.Genes[i].p_Value = fit_results['Coefficients'][idx]['p-Value']
        self.MSE_Train = fit_results['MSE Train']
        self.RMSE_Train = fit_results['RMSE Train']
        self.R2_Train = fit_results['R2 Train']
        self.R2Adj_Train = fit_results['R2 Adjusted Train']
        self.MSE_Test = fit_results['MSE Test']
        self.RMSE_Test = fit_results['RMSE Test']
        self.R2_Test = fit_results['R2 Test']
        self.R2Adj_Test = fit_results['R2 Adjusted Test']
        if self.MSE_Train is None:
            self.MSE_Train = 1e100
        if self.R2_Train is None:
            self.R2_Train = 0
        return 

# assigne ranks and sort genes according to p-Values
    def rank_sort_pValue(self):
        swapped = True
        n = self.Size
        while swapped:
            swapped = False
            for i in range(1, n, 1): # most important first
                if self.Genes[i-1].p_Value > self.Genes[i].p_Value:
                    swapped = True
                    self.Genes.insert(i-1, self.Genes[i])
                    del(self.Genes[i+1])
            n -= 1
# Assign ranks to chromosome. Highest number for most important
        j = self.Size
        for i in range(0, self.Size, 1):
            self.Genes[i].Rank = j
            j -= 1                
        return

    def rank_sort(self, x_expD=None, x_expDn=None, x_lin=None, y=None,\
            LinearSolver='sklearn', cond=1e-20, lapack_driver='gelsy'): # Highest Rank - most important
        if LinearSolver == 'statsmodels': # sort according to p-Values  
            self.rank_sort_pValue()
            return
        Size = self.Size
        mse = np.zeros(shape=(Size), dtype=float)
        tmp_chromosome = copy.deepcopy(self)
        for i in range(0, Size, 1):
            tmp_gene = copy.deepcopy(tmp_chromosome.Genes[i])
            del(tmp_chromosome.Genes[i])
            tmp_chromosome.Size = len(tmp_chromosome.Genes)
            tmp_chromosome.score(x_expD_train=x_expD, x_expDn_train=x_expDn, x_lin_train=x_lin,\
                y_train=y, x_expD_test=None, x_expDn_test=None, x_lin_test=None,\
                y_test=None, LinearSolver=LinearSolver, cond=cond, lapack_driver=lapack_driver)
            mse[i] = tmp_chromosome.MSE_Train
            tmp_chromosome.Genes.insert(i, tmp_gene)
            tmp_chromosome.Size = len(tmp_chromosome.Genes)
        for i in range(0, Size, 1):
            self.Genes[i].Rank = mse[i]
        self.sort(order='Most important first')
        return
      
# sort genes according to their ranks      
# order='Least important first' or 'Most important first'
    def sort(self, order='Most important first'): 
        for i in range(0, self.Size, 1):
            if self.Genes[i].Rank is None:
                return
        swapped = True
        if order == 'Least important first':
            Direction = 1
        else:
            Direction = -1 # most important first
        n = self.Size
        while swapped:
            swapped = False
            for i in range(1, n, 1):
                diff = Direction * (self.Genes[i-1].Rank - self.Genes[i].Rank)
                if diff > 0:
                    swapped = True
                    self.Genes.insert(i-1, self.Genes[i])
                    del(self.Genes[i+1])
            n -= 1
# Assign ranks to chromosome. Highest number for most important
        j = self.Size
        for i in range(0, self.Size, 1):
            self.Genes[i].Rank = j
            j -= 1
        return
    
    def print_score(self):
        idx_lin = self.get_genes_list(Type=0)
        print('Linear indices: ', idx_lin)
        idx_exp = self.get_genes_list(Type=1)
        print('Exponential indices: ', idx_exp)
        print('Chromosome size = ', self.Size)
        print('MSE Train = ', self.MSE_Train)
        print('RMSE Train = ', self.RMSE_Train)
        print('R2 Train = ',self.R2_Train)
        print('R2 Adjusted Train = ', self.R2Adj_Train)   
        if self.MSE_Test is not None:
            print('MSE Test = ', self.MSE_Test)
            print('RMSE Test = ', self.RMSE_Test)
            print('R2 Test = ',self.R2_Test)
            print('R2 Adjusted Test = ', self.R2Adj_Test)   
        print('Rank = ', self.Rank)
        print('Origin: ', self.Origin)
        print('Created time: ', self.time)
        return
            
    def predict(self, x_expD=None, x_expDn=None, x_lin=None):
        y_pred = None
        if x_expD is not None and x_expDn is not None:
            y_pred = np.zeros(shape=(x_expD.shape[0]), dtype=float)
        elif x_lin is not None and y_pred is None:
            y_pred = np.zeros(shape=(x_lin.shape[0]), dtype=float)
        for i in range(0, self.Size, 1):
            idx = self.Genes[i].Idx
            if self.Genes[i].Type == 1: # exponential with 2 coefficients
                y_pred[:] += self.Genes[i].Coef2 * x_expDn[:, idx] * np.exp(self.Genes[i].Coef1 * x_expD[:, idx])
            #   E         += beta                * R**(-n)       * e**(   alpha               * R)  
            elif self.Genes[i].Type == 0: # linear with 1 coefficient
                y_pred[:] += self.Genes[i].Coef1 * x_lin[:, idx] 
            #   E         += gamma               * R**(-m)    
        return y_pred

####################################################### end of class Chromosome

class GA:
    
    def __init__(self, PopulationSize=100, ChromosomeSize=15, UseCorrelationMutation=True,\
            MinCorrMutation=0.9, UseCorrelationBestFit=False, MinCorrBestFit=0.9,\
            MutationProbability=0.3, MutationInterval=[1,3], EliteFraction=0.3,\
            CrossoverFractionInterval=[0.6,0.4], MutationCrossoverFraction=0.5,\
            PrintInterval=10, StopTime=600, RandomSeed=None, verbose=True):
        self.ChromosomeSize = ChromosomeSize
        self.PopulationSize = PopulationSize
        self.MutationProbability = MutationProbability
        self.MutationInterval = MutationInterval
        self.EliteFraction = EliteFraction
        self.nGood = int(EliteFraction * PopulationSize)
        if self.nGood < 2:
            self.nGood = 2
        self.nMutationCrossover = int(MutationCrossoverFraction * PopulationSize)
        self.CrossoverFractionInterval = CrossoverFractionInterval
        self.PrintInterval = PrintInterval
        self.StopTime = StopTime
        self.verbose = verbose
        self.RandomSeed = RandomSeed
        self.UseCorrelationMutation = UseCorrelationMutation
        self.MinCorrMutation = MinCorrMutation
        self.UseCorrelationBestFit = UseCorrelationBestFit
        self.MinCorrBestFit = MinCorrBestFit
        self.DecreasingChromosomes = [] # Best fit and RemoveWorth progress
        self.VIP_Chromosome = None
        self.Population = None 
        self.BestChromosomes = [] # GA progress
        self.BestChromosome = None
        self.idx_exp = None
        self.idx_lin = None
        self.VIP_idx_exp = None
        self.VIP_idx_lin = None
        self.MutationCrossoverFraction = MutationCrossoverFraction
        self.LinearSolver = 'sklearn' # 'sklearn', 'scipy', 'statsmodels'
        self.cond = 1e-20
        self.lapack_driver = 'gelsy'
        self.CrossoverMethod = 'Random' # 'Random', 'Best'
        self.MutationMethod = 'Random' # 'Random'. 'Correlated'
        self.n_lin = None
        self.n_exp = None
        self.n_VIP_lin = None
        self.n_VIP_exp = None
        self.nIter = 1000

# create random gene from pool that does not belong to chromosome
    def create_random_gene(self, chromosome=None, pool_nonlin=None, pool_lin=None):
        if (chromosome is None):
            if (self.VIP_Chromosome is not None):
                chromosome = self.VIP_Chromosome
            else:
                chromosome = Chromosome([])
        if (pool_nonlin is None):
            if (self.idx_exp is not None):
                pool_nonlin = self.idx_exp
            else:
                pool_nonlin = []
        if (pool_lin is None):
            if (self.idx_lin is not None):
                pool_lin = self.idx_lin
            else:
                pool_lin = []
        i_lin = chromosome.get_genes_list(Type=0)
        if i_lin is None:
            i_lin = []
        i_nonlin = chromosome.get_genes_list(Type=1)
        if i_nonlin is None:
            i_nonlin = []
        pool_lin_free = []
        pool_nonlin_free = []
        for i in pool_nonlin:
            if i not in i_nonlin:
                pool_nonlin_free.append(i)
        for i in pool_lin:
            if i not in i_lin:
                pool_lin_free.append(i)                
        n_nonlin = len(pool_nonlin_free)
        n_lin = len(pool_lin_free)
        if (n_nonlin + n_lin) == 0:
            sys.stdout.write(library.RED)
            print('Crash: create_random_gene. No pool.')
            sys.stdout.write(library.RESET)
            return None
        rand = random.randrange(0, n_nonlin+n_lin, 1) # get random number of feature
        if rand < n_nonlin: # non-linear feature
            new_gene = Gene(pool_nonlin_free[rand], Type=1, p_Value=None, rank=None)
        elif n_lin > 0: # linear feature
            rand -= n_nonlin
            new_gene = Gene(pool_lin_free[rand], Type=0, p_Value=None, rank=None)
        return new_gene

    def set_VIP_Chromosome(self):          
        Gene_list = []
        if self.VIP_idx_exp is None:
            self.n_VIP_exp = 0
        else:
            self.n_VIP_exp = len(self.VIP_idx_exp)
        for i in range(0, self.n_VIP_exp, 1): # non-linear genes
            Gene_list.append(Gene(self.VIP_idx_exp[i], Type=1, p_Value=None, rank=None))
        if self.VIP_idx_lin is None:
            self.n_VIP_lin = 0
        else:
            self.n_VIP_lin = len(self.VIP_idx_lin)            
        for i in range(0, self.n_VIP_lin, 1): # linear genes
            Gene_list.append(Gene(self.VIP_idx_lin[i], Type=0, p_Value=None, rank=None))
        if len(Gene_list) > 0:
            self.VIP_Chromosome = Chromosome(Gene_list)
        else:
            self.VIP_Chromosome = None
        return

    def generate_new_chromosome(self):
        if self.VIP_Chromosome is not None:
            Gene_list = copy.deepcopy(self.VIP_Chromosome.Genes)
        else:
            Gene_list = []
        chromosome = Chromosome(Gene_list)
        start = len(Gene_list)
        for i in range(start, self.ChromosomeSize, 1):
            new_gene = self.create_random_gene(chromosome=chromosome)
            if new_gene is not None:
                chromosome.Genes.append(new_gene)
                chromosome.erase_score()
        if (len(chromosome.Genes) == self.ChromosomeSize) and (not chromosome.has_duplicates()):
            chromosome.Size = self.ChromosomeSize
            chromosome.Origin = 'Pool'
            return chromosome
        else:
            sys.stdout.write(library.RED)
            print('Crash: generate_new_chromosome')
            sys.stdout.write(library.RESET)
            return None

# mutate one gene
    def mutate(self, chromosome, MutationMethod='Random'):
        Gene_list = []
#        print(chromosome.Size)
#        chromosome.print_score()
        i = 0
        while i < len(chromosome.Genes):
            if chromosome.Genes[i].is_gene_exists(self.VIP_Chromosome):
                i_copy = copy.deepcopy(chromosome.Genes[i])
                Gene_list.append(i_copy) # copy VIP genes
                del(chromosome.Genes[i]) # remove VIP gene from chromosome
            i += 1
        chromosome.erase_score() # recalculate size
        gene_idx = random.randrange(0, chromosome.Size, 1)
        old_gene = chromosome.Genes[gene_idx] 
        for i in range(0, chromosome.Size, 1):
            if (chromosome.Genes[i].Idx == gene_idx) and (chromosome.Genes[i].Type == old_gene.Type):
                continue
            i_copy = copy.deepcopy(chromosome.Genes[i])
            Gene_list.append(i_copy) # copy VIP genes
        chromosome = Chromosome(Gene_list)
        if (old_gene.Type == 0) and (MutationMethod == 'Correlated'):
            corr_idx = self.get_correlated_features_list(chromosome, Model=2,\
                UseCorrelationMatrix = self.UseCorrelationMutation, MinCorr = self.MinCorrMutation)
            idx = []
            for i in range(0, len(corr_idx), 1):
                if corr_idx[i][0] == old_gene.Idx:
                    idx = corr_idx[i]
                    break
            if len(idx) > 1:
                new_gene = self.create_random_gene(chromosome=chromosome, pool_nonlin = [], pool_lin=idx)
                if new_gene is not None:
                    chromosome.Genes.append(new_gene)
                    chromosome.erase_score()
                    chromosome.Origin = 'Mutation'
                    if not chromosome.has_duplicates():
                        return chromosome
                else:
                    sys.stdout.write(library.RED)
                    print('Crash: mutate Correlated')
                    sys.stdout.write(library.RESET)
        new_gene = self.create_random_gene(chromosome=chromosome)
        if new_gene is not None:
            chromosome.Genes.append(new_gene)
            chromosome.erase_score()
            chromosome.Origin = 'Mutation'
            if not chromosome.has_duplicates():
                return chromosome
            else:
                sys.stdout.write(library.RED)
                print('Chash: Duplicates in mutation')
                sys.stdout.write(library.RESET)
        else:
            sys.stdout.write(library.RED)
            print('Crash: mutate. Cannot create random gene')
            sys.stdout.write(library.RESET)
            return None
               
# mutate more than one gene
    def mutate_many(self, chromosome, nMutations, MutationMethod='Random'):
        n = 0
        ch = copy.deepcopy(chromosome)
        while n < nMutations:
            ch = self.mutate(ch, MutationMethod=MutationMethod)
            if ch.has_duplicates():
                ch = copy.deepcopy(chromosome)
                n = 0
                continue
            n += 1
        return chromosome
        
# assign Chromosome.Rank for all population    
    def sort(self, order='Most important first'): 
# sort population list
        swapped = True
        n = self.PopulationSize
        if order == 'Least important first':
            Direction = -1
        else:
            Direction = 1
# bubble sort from low to high. lowest MSE corresponds to best chromosome
        while swapped:
            swapped = False
            for i in range(1, n, 1):
                diff = Direction * (self.Population[i-1].MSE_Train - self.Population[i].MSE_Train)
                if diff > 0:
                    swapped = True
                    self.Population.insert(i-1, self.Population[i])
                    del(self.Population[i+1])
            n - n - 1
# Assign ranks to population. Highest number for most important
            j = self.PopulationSize
            for i in range(0, self.PopulationSize, 1):
                self.Population[i].Rank = j
                j -= 1
        return 
    
# return the best chromosome (population must be sorted)
    def get_best_chromosome(self):
        return self.Population[0]
        
    def crossover(self, chromosome1, chromosome2):
# will produce a child from two parents
        if self.VIP_Chromosome is None:
            Gene_list = []
        else:
            Gene_list = copy.deepcopy(self.VIP_Chromosome.Genes)
        chromosome = Chromosome(Gene_list)
        size_VIP = len(Gene_list)
        size_left = self.ChromosomeSize - size_VIP
# randon number in given range [min .. max]
        rand = (self.CrossoverFractionInterval[1] - self.CrossoverFractionInterval[0]) * random.random() + self.CrossoverFractionInterval[0]
        n1 = int(size_left * rand) # number of genes to be taken from chromosome 1
        n2 = self.ChromosomeSize - n1 - size_VIP # number of random genes to be taken from chromosome 2
        idx_lin1 = chromosome1.get_genes_list(Type=0) # linear indices of first chromosome
        idx_nonlin1 = chromosome1.get_genes_list(Type=1) # non-linear indices of first chromosome
        idx_lin2 = chromosome2.get_genes_list(Type=0) # linear indices of second chromosome
        idx_nonlin2 = chromosome2.get_genes_list(Type=1) # non-linear indices of second chromosome
        if (self.CrossoverMethod == 'Random'):
# append features from first chromosone
            for i in range(0, n1, 1):
                new_gene = self.create_random_gene(chromosome=chromosome, pool_nonlin=idx_nonlin1, pool_lin=idx_lin1)
                if new_gene is not None:
                    chromosome.Genes.append(new_gene)
                    chromosome.erase_score()
                else:
                    sys.stdout.write(library.RED)
                    print('Crash: crossover n1')
                    sys.stdout.write(library.RESET)
                    break
# append features from first chromosone
            for i in range(0, n2, 1):
                new_gene = self.create_random_gene(chromosome=chromosome, pool_nonlin=idx_nonlin2, pool_lin=idx_lin2)
                if new_gene is not None:
                    chromosome.Genes.append(new_gene)
                    chromosome.erase_score()
                else:
                    sys.stdout.write(library.RED)
                    print('Crash: crossover n2')
                    print('pool_nonlin: ', idx_nonlin2)
                    print('pool_lin: ', idx_lin2)
                    sys.stdout.write(library.RESET)
                    chromosome.print_score()
                    break
        if (self.CrossoverMethod == 'Best'):
            count = 0 # add most important genes from first chromosome
            for i in range(0, self.ChromosomeSize, 1):
                new_gene = chromosome1.Genes[i]
                if not new_gene.is_gene_exists(chromosome):
                    chromosome.Genes.append(new_gene)
                    chromosome.erase_score()
                    count += 1
                    if count >= n1:
                        break
            count = 0 # add most important genes from first chromosome
            for i in range(0, self.ChromosomeSize, 1):
                new_gene = chromosome2.Genes[i]
                if not new_gene.is_gene_exists(chromosome):
                    chromosome.Genes.append(new_gene)
                    chromosome.erase_score()
                    count += 1
                    if count >= n2:
                        break
# add remaining genes from pool if needed
        while len(chromosome.Genes) < self.ChromosomeSize:
            new_gene = self.create_random_gene(chromosome=chromosome)
            if new_gene is not None: 
                chromosome.Genes.append(new_gene)
                chromosome.erase_score()
            else:
                sys.stdout.write(library.RED)
                print('Crash: crossover')
                sys.stdout.write(library.RESET)
                return None
        chromosome.erase_score()
        chromosome.Origin = 'Crossover'
        if not chromosome.has_duplicates():
            return chromosome # return a new child
        else:
            sys.stdout.write(library.RED)
            print('Crash: crossover produces duplicates')
            sys.stdout.write(library.RESET)
            return None
   
    def init_population(self):
        self.Population = []
        k = 0
        while (len(self.Population) < self.PopulationSize) and (k < 2*self.PopulationSize):
            k += 1
            chromosome = self.generate_new_chromosome()
            if (chromosome is not None) and (not chromosome.has_duplicates()):
                self.Population.append(chromosome)
        if len(self.Population) != self.PopulationSize:
            sys.stdout.write(library.RED)
            print('Crash: init_population')
            sys.stdout.write(library.RESET)
        return 
    
    def fit(self, x_expD_train=None, x_expDn_train=None, x_lin_train=None,\
            y_train=None, x_expD_test=None, x_expDn_test=None,\
            x_lin_test=None, y_test=None, idx_exp=None, idx_lin=None,\
            VIP_idx_exp=None, VIP_idx_lin=None, CrossoverMethod='Random',\
            MutationMethod='Random', LinearSolver = 'sklearn', cond=1e-20,\
            lapack_driver='gelsy', nIter=1000): 
        self.CrossoverMethod = CrossoverMethod
        self.MutationMethod = MutationMethod
        self.LinearSolver = LinearSolver
        self.cond = cond
        self.lapack_driver = lapack_driver
        self.VIP_idx_lin = VIP_idx_lin
        self.VIP_idx_exp = VIP_idx_exp
        self.set_VIP_Chromosome()
        self.nIter = nIter
        self.C = None 
        
        if (x_expD_train is None) or (x_expDn_train is None): # use only linear features
            self.idx_exp = []
            x_expD_train = None  
            x_expDn_train = None
            self.n_exp = 0
        else: # linear and non-linear features
            if (idx_exp is None) and (x_expD_train is not None) and (x_expDn_train is not None):
                self.idx_exp = list(range(0, x_expD_train.shape[1], 1))
                self.n_exp = len(self.idx_exp)
        if (idx_lin is None) and (x_lin_train is not None):
            if (x_lin_train is not None):
                self.C = np.cov(x_lin_train, rowvar=False, bias=True)
                self.idx_lin = list(range(0, x_lin_train.shape[1], 1))
                self.n_lin = len(self.idx_lin)
            else:
                self.idx_lin = []   
                self.n_lin = 0
        if self.RandomSeed is None:
            random.seed()
        else:
            random.seed(self.RandomSeed)
        # get initial population
        self.init_population()
        sys.stdout.write(library.RED)        
        print('VIP: ', self.VIP_idx_lin)
        if self.VIP_Chromosome is not None:
            self.VIP_Chromosome.print_score()
        sys.stdout.write(library.RESET)            
        StartTime = time()
        BestChromosome = None
        LastChangeTime = time()
        nIter = 0
        try: # loop will be interrupted by pressing Ctrl-C
            while ((time() - LastChangeTime) < self.StopTime) and (nIter < self.nIter):
                nIter += 1
                # get fitness for population
                for i in range(0, self.PopulationSize, 1): # calculate MSE and R2 for each chromosome
                    self.Population[i].score(x_expD_train=x_expD_train, x_expDn_train=x_expDn_train,\
                        x_lin_train=x_lin_train, y_train=y_train, x_expD_test=x_expD_test, x_expDn_test=x_expDn_test,\
                        x_lin_test=x_lin_test, y_test=y_test, LinearSolver=self.LinearSolver,\
                        cond=self.cond, lapack_driver=self.lapack_driver) 
                self.sort(order='Most important first') # sort chromosomes in population based on just obtained score
                if (CrossoverMethod == 'Best'):
                    for i in range(0, self.PopulationSize, 1):
                        self.Population[i].rank_sort(x_expD=x_expD_train, x_expDn=x_expDn_train,\
                            x_lin=x_lin_train, y=y_train, LinearSolver=self.LinearSolver,\
                            cond=self.cond, lapack_driver=self.lapack_driver) # rank genes using MSE
                BetterChromosome = self.get_best_chromosome() # get best chromosome  
                if BestChromosome is None:
                    BestChromosome = self.get_best_chromosome()
                if BetterChromosome.MSE_Train < BestChromosome.MSE_Train:
                    BestChromosome.rank_sort(x_expD=x_expD_train, x_expDn=x_expDn_train,\
                        x_lin=x_lin_train, y=y_train, LinearSolver=self.LinearSolver,\
                        cond=self.cond, lapack_driver=self.lapack_driver)                    
                    BestChromosome = copy.copy(BetterChromosome)
                    BestChromosome.time = time() - StartTime  
                    self.BestChromosomes.append(BestChromosome)
                    del(BetterChromosome)
                    LastChangeTime = time()
                    print('Iteration = ', nIter)
                    BestChromosome.print_score()
                if self.verbose and ((nIter % self.PrintInterval) == 0):  
                    print('Iteration = ', nIter, "\nTime since last improvement =",\
                        str(int((time() - LastChangeTime))), 'sec')
                new_population = []
                for j in range(0, self.nMutationCrossover, 1):
                    rand_MutationProbability = random.random()
                    if rand_MutationProbability <= self.MutationProbability: # mutation
                        rand = random.randrange(0, self.nGood, 1) # which chromosome to mutate
                        nMutations = random.randrange(self.MutationInterval[0], self.MutationInterval[1], 1)
                        chromosome = copy.copy(self.Population[rand])
                        chromosome = self.mutate_many(chromosome, nMutations,\
                            MutationMethod=self.MutationMethod) # mutate some of good chromosome
                    else: # crossover 
                        c1 = 0
                        c2 = 0
                        k = 0
                        while (c1 == c2) and (k < 10):
                            c1 = random.randrange(0, self.nGood, 1) # chose first chromosome for crossover
                            c2 = random.randrange(0, self.nGood, 1) # chose second chromosome for crossover
                            if c1 == c2:
                                k += 1
                                continue
                            chromosome1 = copy.copy(self.Population[c1])
                            chromosome2 = copy.copy(self.Population[c2])
                            if chromosome1.is_exist(chromosome2):
                                c1 = c2
                                k += 1
                        if k >= 100:
                            sys.stdout.write(library.RED)
                            print('Crash: fit main loop. crossover')
                            print('Might want to increase EliteFracrion')
                            sys.stdout.write(library.RESET)
                            chromosome = self.generate_new_chromosome()
                        else:
                            chromosome = self.crossover(chromosome1, chromosome2)
                    if chromosome is None:
                        sys.stdout.write(library.RED)
                        print('Crash: crossover')
                        sys.stdout.write(library.RESET)
                        chromosome = self.generate_new_chromosome()
                    new_population.append(chromosome)
                while len(new_population) < self.PopulationSize:
                    new_population.append(self.generate_new_chromosome())
                del(self.Population)
                self.Population = copy.copy(new_population)
        except KeyboardInterrupt: # interrupt by pressing Ctrl-C
            pass
        self.BestChromosome = BestChromosome
        return
    
    def PlotChromosomes(self, Fig_Number, ChromosomesList, XAxis='#',\
                        YAxis='MSE Train', PlotType='Line', F=None):
        fig = plt.figure(Fig_Number, figsize=(19,10))
        if XAxis == '#':
            XPlot = list(range(0, len(ChromosomesList), 1))
            plt.xlabel('Number of best chromosome')
        elif XAxis == 'time':
            XPlot = []
            for i in ChromosomesList:
                XPlot.append(i.time)
            plt.xlabel('Time')
        elif XAxis == 'Nonzero':
            XPlot = []
            for i in ChromosomesList:
                XPlot.append(i.Size)
            plt.xlabel('Nonzero coefficiens')
        YPlot = []       
        for i in ChromosomesList:       
            if YAxis == 'MSE Train':
                YPlot.append(i.MSE_Train)                
            elif YAxis == 'MSE Test':
                YPlot.append(i.MSE_Test)  
            elif YAxis == 'RMSE Train':
                YPlot.append(i.RMSE_Train)
            elif YAxis == 'RMSE Test':
                YPlot.append(i.RMSE_Test)
            elif YAxis == 'R2 Train':
                YPlot.append(i.R2_Train)  
            elif YAxis == 'R2 Test':
                YPlot.append(i.R2_Test)  
            elif YAxis == 'R2 Adjusted Train':
                YPlot.append(i.R2Adj_Train)                 
            elif YAxis == 'R2 Adjusted Test':
                YPlot.append(i.R2Adj_Test)                   
            else:
                sys.stdout.write(library.RED)
                print('Crash: PlotChromosomes')
                sys.stdout.write(library.RESET)
                return
        plt.ylabel(YAxis)
        if PlotType == 'Line':
            plt.plot(XPlot, YPlot)
        if PlotType == 'Scatter':
            plt.scatter(XPlot, YPlot)
        plt.title('Progress of Genetic Algorithm')       
        plt.axis('tight')
        plt.show(fig)
        if F is None:
            return
        if F is 'Default':
            F = YAxis
        else:
            F += ' ' + YAxis 
        plt.savefig(F, bbox_inches='tight')
        plt.close(fig)
        return
    
    def RemoveWorstGene(self, chromosome, x_expD=None, x_expDn=None, x_lin=None, y=None, verbose=True):
        if self.VIP_Chromosome is None: 
            del(chromosome.Genes[-1])
            chromosome.Size -= 1
            chromosome.erase_score()
        elif self.VIP_Chromosome.Size >= chromosome.Size:
            print('Cannot remove any gene since all of them are VIP')
            return None
        else:
            for i in range(0, chromosome.Size, 1):
                gene = chromosome.Genes[chromosome.Size-i-1]
                if not gene.is_gene_exists(self.VIP_Chromosome):
                    del(chromosome.Genes[chromosome.Size-i-1])
                    chromosome.Size -= 1
                    chromosome.erase_score()
                    break
        chromosome.score(x_expD_train=x_expD, x_expDn_train=x_expDn, x_lin_train=x_lin, y_train=y,\
            LinearSolver=self.LinearSolver, cond=self.cond, lapack_driver=self.lapack_driver)
        return chromosome
    
    def get_correlated_features_list(self, chromosome, Model=1,\
            UseCorrelationMatrix=True, MinCorr=0.9):
# returns nested list of correlated features idx_corr[i][j]
# each list [j] contains indices of features that have correlation <= to MinCorr
# with 0 element that corresponds to [i] index
# correlation value is given by MinCorr
# Model 1: correlated features may overlap each other
# Model 2: correlated features do not overlap each other. Works faster
    
        def InList(List, Value):
    # returns True if Value is in List        
            for i in range(0, len(List), 1):
                for j in range(0, len(List[i]), 1):
                    if List[i][j] == Value:
                        return True
            return False
        
        if not UseCorrelationMatrix:
            return self.get_all_features_list(chromosome)
        idx_lin = chromosome.get_genes_list(Type=0)
        if idx_lin is []:
            return []        
        list1 = []
        list2 = []
        for i in range(0, len(idx_lin), 1):
            idx = idx_lin[i]
            list1.append(list(''))
            list2.append(list(''))
            list1[i].append(idx)
            list2[i].append(idx)
        k = 0
        for i in idx_lin:
            for j in range(0, self.n_lin, 1):
                if i == j:
                    continue
                if self.C[i, j] > MinCorr:
                    if j not in list1:
                        list1[k].append(j)
                    if not InList(list2, j):
                        list2[k].append(j)
            k += 1
        if Model == 1:
            return list1
        elif Model == 2:
            return list2
    
    def get_all_features_list(self, chromosome):
        idx_lin = chromosome.get_genes_list(Type=0) # only linear features
        if idx_lin == []:
            return []
        idx_corr = []
        n = np.array(range(0, self.n_lin, 1), dtype=int)
        for i in range(0, len(idx_lin), 1):
            idx_corr.append(n)
        return idx_corr
    
    def BestFit(self, chromosome, x_expD=None, x_expDn=None, x_lin=None, y=None, verbose=True):        
        Best = copy.deepcopy(chromosome)        
        Best.rank_sort(x_expD=x_expD, x_expDn=x_expDn, x_lin=x_lin, y=y,\
            LinearSolver=self.LinearSolver, cond=self.cond, lapack_driver=self.lapack_driver)
        Best.score(x_expD_train=x_expD, x_expDn_train=x_expDn, x_lin_train=x_lin, y_train=y,\
            LinearSolver=self.LinearSolver, cond=self.cond, lapack_driver=self.lapack_driver)        
        Found = True
        idx_corr = self.get_correlated_features_list(Best, Model=1,\
            UseCorrelationMatrix=self.UseCorrelationBestFit, MinCorr=self.MinCorrBestFit)
        if idx_corr == []: # only for linear features
            return Best
        while Found:
            Found = False
            for i in range(0, len(idx_corr), 1):
                Chromosome_list = []    
                idx_list = Best.get_genes_list(Type=0)
                ch = copy.deepcopy(Best)
                old_gene = copy.deepcopy(ch.Genes[i])
                if (not old_gene.is_gene_exists(self.VIP_Chromosome)) and (old_gene.Type == 0):
                    for j in range(0, len(idx_corr[i]), 1):
                        new_idx = idx_corr[i][j]
                        if new_idx in idx_list:
                            continue # check if new index exists in chromosome
                        ch.Genes[i].Idx = idx_corr[i][j]
                        ch.erase_score()
                        ch.score(x_expD_train=x_expD, x_expDn_train=x_expDn, x_lin_train=x_lin, y_train=y,\
                            LinearSolver=self.LinearSolver, cond=self.cond, lapack_driver=self.lapack_driver)
                        if not ch.is_exist(Chromosome_list):  
                            ch.Origin = 'Best Fit'
                            chromosome_copy = copy.deepcopy(ch)                    
                            Chromosome_list.append(chromosome_copy)
                    for j in Chromosome_list:
                        if j.MSE_Train < Best.MSE_Train:
                            Best = copy.deepcopy(j)    
                            Found = True
                            if verbose:
                                sys.stdout.write(library.GREEN)
                                print('Best fit new chromosome')
                                sys.stdout.write(library.RESET)
                                Best.print_score()
            if Found:                
                Best.rank_sort(x_expD=x_expD, x_expDn=x_expDn, x_lin=x_lin, y=y,\
                    LinearSolver=self.LinearSolver, cond=self.cond, lapack_driver=self.lapack_driver)
                Best.score(x_expD_train=x_expD, x_expDn_train=x_expDn, x_lin_train=x_lin, y_train=y,\
                    LinearSolver=self.LinearSolver, cond=self.cond, lapack_driver=self.lapack_driver)

        return Best
        
    # store in sheet of .xlsx file description of fit with real coefficients
    def Results_to_xlsx(self, FileName, chromosome_list, FeaturesNonlinear=None,\
            FeaturesAll=None, FeaturesReduced=None):
        writeResults = pd.ExcelWriter(FileName, engine='openpyxl')
        if type(chromosome_list) is list:
            nChromosomes = len(chromosome_list)
        else:
            nChromosomes = 1 # chromosome_list can be one chromosome
            chromosome_list = [chromosome_list] # make a list with 1 record
        sheet_names = []
        for i in chromosome_list:
            sheet_names.append(i.Size)
        if len(sheet_names) != len(set(sheet_names)): # have duplicates
            sheet_names = list(range(0, len(chromosome_list), 1))
        for k in range(0, nChromosomes, 1): # for each excel sheet
            SheetName = str(sheet_names[k]) #replace with number of genes later
            idx_nonlin = chromosome_list[k].get_genes_list(Type=1)
            idx_lin = chromosome_list[k].get_genes_list(Type=0)
            
            Results = pd.DataFrame(np.empty(shape = (chromosome_list[k].Size, 22)).astype(str), \
                columns=['Feature index','Feature type','Bond 1','Power 1','Intermolecular 1',\
                'Distance 1 type','Bond 2','Power 2', 'Intermolecular 2','Distance 2 type',\
                'Number of distances in feature', 'Number of constants', 'Coefficients',\
                'Coefficients 2', 'MSE Train','RMSE Train', 'R2 Train', 'R2 Adjusted Train',\
                'MSE Test','RMSE Test','R2 Test', 'R2 Adjusted Test'], dtype=str)
            Results[:][:] = ''
            max_distances_in_feature = 1
            i = 0
            while i < len(idx_nonlin):
                index = idx_nonlin[i]
                gene = Gene(index, Type=1)
                I = chromosome_list[k].find_gene_idx(gene)
                Results.loc[i]['Feature index'] = index
                Results.loc[i]['Feature type'] = FeaturesNonlinear[index].FeType
                Results.loc[i]['Bond 1'] = FeaturesNonlinear[index].DtP1.Distance.Atom1.Symbol + '-' + FeaturesNonlinear[index].DtP1.Distance.Atom2.Symbol
                Results.loc[i]['Power 1'] = FeaturesNonlinear[index].DtP1.Power
                Results.loc[i]['Distance 1 type'] = str(FeaturesNonlinear[index].DtP1.DtpType)
                if FeaturesNonlinear[index].DtP1.Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
                    Results.loc[i]['Intermolecular 1'] = 'Yes'
                else:
                    Results.loc[i]['Intermolecular 1'] = 'No'           
                Results.loc[i]['Number of distances in feature'] = FeaturesNonlinear[index].nDistances
                Results.loc[i]['Number of constants'] = FeaturesNonlinear[index].nConstants
                Results.loc[i]['Coefficients'] = chromosome_list[k].Genes[I].Coef1 # power of exponent
                Results.loc[i]['Coefficients 2'] = chromosome_list[k].Genes[I].Coef2 # in front of exp
                i += 1
            j = 0 # counts linear genes
            while (j < len(idx_lin)): # counts rows in excel sheet
                index = idx_lin[j]
                gene = Gene(index, Type=0)
                I = chromosome_list[k].find_gene_idx(gene)
                Results.loc[i]['Feature index'] = index
                Results.loc[i]['Feature type'] = FeaturesReduced[index].FeType                
                if FeaturesReduced[index].nDistances >= 1:
                    Results.loc[i]['Bond 1'] = FeaturesReduced[index].DtP1.Distance.Atom1.Symbol + '-' + FeaturesReduced[index].DtP1.Distance.Atom2.Symbol
                    Results.loc[i]['Power 1'] = FeaturesReduced[index].DtP1.Power
                    Results.loc[i]['Distance 1 type'] = FeaturesReduced[index].DtP1.DtpType
                    if FeaturesReduced[index].DtP1.Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
                        Results.loc[i]['Intermolecular 1'] = 'Yes'
                    else:
                        Results.loc[i]['Intermolecular 1'] = 'No'
                else:
                    Results.loc[i]['Bond 1'] = ''
                    Results.loc[i]['Power 1'] = ''
                    Results.loc[i]['Intermolecular 1'] = ''
                    Results.loc[i]['Distance 1 type'] = ''            
                if FeaturesReduced[index].nDistances >= 2:
                    Results.loc[i]['Bond 2'] = FeaturesReduced[index].DtP2.Distance.Atom1.Symbol + '-' + FeaturesReduced[index].DtP2.Distance.Atom2.Symbol
                    Results.loc[i]['Power 2'] = FeaturesReduced[index].DtP2.Power
                    Results.loc[i]['Distance 2 type'] = FeaturesReduced[index].DtP2.DtpType
                    if max_distances_in_feature < 2:
                        max_distances_in_feature = 2
                    if FeaturesReduced[index].DtP2.Distance.isIntermolecular: # 1 = Intermolecular 0 = Intramolecular
                        Results.loc[i]['Intermolecular 2'] = 'Yes'
                    else:
                        Results.loc[i]['Intermolecular 2'] = 'No'
                else:
                    Results.loc[i]['Bond 2'] = ''
                    Results.loc[i]['Power 2'] = ''
                    Results.loc[i]['Intermolecular 2'] = ''
                    Results.loc[i]['Distance 2 type'] = ''                    
                counter = 0
                current_feature_type = FeaturesReduced[index].FeType
                for m in range(0, len(FeaturesAll), 1):
                    if FeaturesAll[m].FeType == current_feature_type:
                        counter += 1
                Results.loc[i]['Number of distances in feature'] = counter
                Results.loc[i]['Number of constants'] = 1
                Results.loc[i]['Coefficients'] = chromosome_list[k].Genes[I].Coef1  
                j += 1
                i += 1  
            Results.loc[0]['MSE Train'] = chromosome_list[k].MSE_Train
            Results.loc[0]['RMSE Train'] = chromosome_list[k].RMSE_Train
            Results.loc[0]['R2 Train'] = chromosome_list[k].R2_Train
            Results.loc[0]['R2 Adjusted Train'] = chromosome_list[k].R2Adj_Train
            Results.loc[0]['MSE Test'] = chromosome_list[k].MSE_Test
            Results.loc[0]['RMSE Test'] = chromosome_list[k].RMSE_Test
            Results.loc[0]['R2 Test'] = chromosome_list[k].R2_Test
            Results.loc[0]['R2 Adjusted Test'] = chromosome_list[k].R2Adj_Test   
            if max_distances_in_feature == 1:
                del(Results['Bond 2'])
                del(Results['Power 2'])
                del(Results['Intermolecular 2'])
                del(Results['Distance 2 type'])
            Results.to_excel(writeResults, sheet_name=SheetName)            
        writeResults.save()
        return

# only for linear now
    def BestFitTree(self, chromosome, x_nonlin=None, x_lin=None, y=None, verbose=True):       
        active_features = chromosome.get_genes_list(Type=0)
        VIP_idx = self.VIP_idx_lin
        MaxLoops=10000
        MaxBottom=1000
        x_std, _ = regression.Standardize(x_lin)
        y_std = y
        
        corr_list = self.get_correlated_features_list(chromosome, Model=2,\
            UseCorrelationMatrix=self.UseCorrelationBestFit, MinCorr=self.MinCorrBestFit)
        
        idx = library.FindBestSetTree(x_std, y_std, active_features, corr_list, VIP_idx=VIP_idx,\
            MaxLoops=MaxLoops, MaxBottom=MaxBottom, verbose=verbose)
        
        Genes_list = []
        for i in range(0, len(idx), 1):
            Idx = idx[i]
            gene = Gene(Idx, Type=0, p_Value=None, rank=None)
            Genes_list.append(gene)
        chromosome = Chromosome(Genes_list)
        return chromosome
        