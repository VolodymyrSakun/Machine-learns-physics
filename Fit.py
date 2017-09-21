from project1 import library
from project1.genetic import GA
from project1 import IOfunctions
from project1 import regression
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
import shutil
import copy
import pickle 
from sklearn.utils import shuffle

if __name__ == '__main__':
# Global variables
    DesiredNumberVariables = 15
    FirstAlgorithm = 'GA' # specifies algorithm that will give rough initial fit. 
    # Can be 'ENet' or 'GA' 
    UseVIP = False # if true fit will be found in two steps. First step - fit only
    # single distances, select most important (VIP) features which will be kept
    test_size = 1e-20 # fraction of test size
# Elastic net parameters    
    L1_Single = 0.7
    eps_single = 1e-3
    n_alphas_single = 100
    L1 = 0.7
    eps = 1e-3
    n_alphas = 100
# Best Fit parameters
#    BestFitMethod = 'Fast' # can be 'Tree' or 'Fast'
#    MaxLoops = 1000 # integer number - greater = slower but better fit
#    MaxBottom = 50 # integer - number of finished branches
    UseCorrelationMutation=True
    MinCorrMutation=0.9
    UseCorrelationBestFit=False
    MinCorrBestFit=0.8
    
    # for Best Fit algorithm. If False, all features will be used in trials 
    # to find Best Fit. Overwise, only most correlated features will be used.
    # If false, will be slow for big number of features
    # needed only if UseCorrelationMatrix=True
    # used for creating list of most correlated features for Best Fit
    # float [0 .. 1]
    F_Out_Single = 'Single BE.xlsx'
    F_ENet_Single = 'Single ENet path'
    F_GA_Single = 'Single GA path'
    F_BE_Single = 'Single BE path'
    F_Out = 'BE.xlsx'
    F_ENet = 'ENet path'
    F_GA = 'GA path'
    F_ga_structure = 'GA_structure.dat'
    F_MoleculesDescriptor = 'MoleculesDescriptor.'
    Slope = 0.001 # stopping criterion for VIP features. Slope of RMSE graph
    VIP_number = 5 # stopping criterion for VIP features. Desired number of VIP features
# GA parameters
    PopulationSize = 100 # Population
    ChromosomeSize = DesiredNumberVariables
    MutationProbability = 0.3 # probability of mutation
    MutationInterval = [1, 3] # will be randomly chosen between min and max-1
    EliteFraction = 0.4 # fracrion of good chromosomes that will be used for crossover
    MutationCrossoverFraction = 0.3
    CrossoverFractionInterval = [0.6, 0.4] # how many genes will be taken from first and second best chromosomes (fraction)
    IterationPerRecord = 100 # Display results of current fit after N iterations
    StopTime = 600 # How long in seconds GA works without improvement
    nIter = 1000 # How many populations will be used before GA stops
    RandomSeed = 101
    LinearSolver = 'sklearn' # 'sklearn', 'scipy', 'statsmodels'
    cond=1e-03 # for scipy solver
    lapack_driver='gelsy' # 'gelsd', 'gelsy', 'gelss', for scipy solver
    NonlinearFunction = 'exp'
    UseNonlinear = False # if true, only linear regression will be used
    CrossoverMethod = 'Random' # 'Random' or 'Best'
    MutationMethod = 'Correlated' # 'Random' or 'Correlated'
    verbose = True

    
# Read features and structures from files stored by "Generate combined features"
    FilterResults = IOfunctions.ReadFeatures(\
        F_Nonlinear='Distances.csv', F_linear_Train='LinearFeaturesTrain.csv',\
        F_Response_Train='ResponseTrain.csv', F_linear_Test='LinearFeaturesTest.csv',\
        F_Response_Test='ResponseTest.csv', F_NonlinearFeatures = 'NonlinearFeatures.dat',\
        F_FeaturesAll='LinearFeaturesAll.dat', F_FeaturesReduced='LinearFeaturesReduced.dat',\
        F_System='system.dat', F_Records=None, verbose=False)
    
    X_train_nonlin = FilterResults['X Nonlinear']
    X_train_lin = FilterResults['X Linear Train']
    Y_train = FilterResults['Response Train']
    X_test_lin = FilterResults['X Linear Test']
    Y_test = FilterResults['Response Test']
    FeaturesNonlinear = FilterResults['Nonlinear Features']
    FeaturesAll = FilterResults['Linear Features All']
    FeaturesReduced = FilterResults['Linear Features Reduced']
    system = FilterResults['System']
#    records = results['Records']

# split data in order to separate training set and test set
# all response variables Y must be 1D arrays

    if (X_train_nonlin is not None) and (X_train_lin is None):
        UseNonlinear = True
        print('Linear features are not provides. Use non-linear regression only')
        X_train_lin = None
        X_test_lin = None
    if ((X_train_nonlin is None) and (X_train_lin is not None)) or (not UseNonlinear):
        UseNonlinear = False
        if not UseNonlinear:
            print('Activated linear features only')
        else:
            print('Non-linear features are not provided. Use linear regression only')
        X_train_nonlin = None
        X_test_nonlin = None
    if ((X_train_nonlin is not None) and (X_train_lin is not None)) and UseNonlinear:
        print('Linear and non-linear features are provided')        
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
        SingleFeaturesReduced = []
        for i in range(0, len(FeaturesReduced), 1):
            if FeaturesReduced[i].nDistances == 1:
                SingleFeaturesReduced.append(FeaturesReduced[i])
        Single_X_train = np.zeros(shape=(X_train_lin.shape[0], len(SingleFeaturesReduced)), dtype=float)
        Single_X_test = np.zeros(shape=(X_test_lin.shape[0], len(SingleFeaturesReduced)), dtype=float)
        j = 0
        for i in range(0, len(FeaturesReduced), 1):
            if FeaturesReduced[i].nDistances == 1:
                Single_X_train[:, j] = X_train_lin[:, i]
                Single_X_test[:, j] = X_test_lin[:, i]
                j += 1
        t = time.time()       
        ga = GA(PopulationSize=PopulationSize, ChromosomeSize=ChromosomeSize,\
            MutationProbability=MutationProbability, MutationInterval=MutationInterval,\
            EliteFraction=EliteFraction, MutationCrossoverFraction=MutationCrossoverFraction,\
            CrossoverFractionInterval=CrossoverFractionInterval, IterationPerRecord=IterationPerRecord,\
            StopTime=StopTime, RandomSeed=RandomSeed, verbose=verbose,\
            UseCorrelationMutation=UseCorrelationMutation, MinCorrMutation=MinCorrMutation,\
            UseCorrelationBestFit=UseCorrelationBestFit, MinCorrBestFit=MinCorrBestFit)
    # linear only for now
        idx = list(range(0, Single_X_train.shape[1], 1))
        results = regression.fit_linear(idx, Single_X_train, Y_train, x_test=Single_X_test, y_test=Y_test,\
            MSEall_train=None, MSEall_test=None, normalize=True, LinearSolver='sklearn')
        ga.MSEall_train = results['MSE Train']
        ga.MSEall_test = results['MSE Test']    

        if (FirstAlgorithm == 'ENet'):
            print('Only linear features will be considered')
            Alphas = np.logspace(-7, -3.5, num=100, endpoint=True, base=10.0, dtype=None)
            enet = regression.ENet(L1=0.7, eps=1e-3, nAlphas=100, alphas=Alphas, random_state=None)
            print('Elastic Net Fit for all features')
            print('L1 portion = ', enet.L1)
            print('Epsilon = ', enet.eps)
            print('Number of alphas =', enet.nAlphas)
            print('Number of features go to elastic net regularisation = ', len(FeaturesReduced))
            enet.fit(Single_X_train, Y_train, VIP_idx=None, Criterion='Mallow', normalize=True,\
                max_iter=1000, tol=0.0001, cv=None, n_jobs=1, selection='random', verbose=True)
            enet.plot_path(1, FileName=F_ENet_Single)
            idx = enet.idx
            Gene_list = []
            for i in idx:
                Gene_list.append(GA.Gene(i, Type=0))
            chromosome = ga.Chromosome(Gene_list)
            chromosome.erase_score()
            # chromosome contains only linear indices
            # use standardized for training set and regular for testing
            chromosome.score(x_nonlin_train=None, x_lin_train=Single_X_train,\
                y_train=Y_train, x_nonlin_test=None, x_lin_test=Single_X_test,\
                y_test=Y_test, NonlinearFunction=ga.NonlinearFunction,\
                LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
            if ga.LinearSolver == 'statsmodels':
                chromosome.rank_sort_pValue()
            else:
                chromosome.rank(x_nonlin=X_train_nonlin, x_lin=Single_X_train,\
                    y=Y_train, NonlinearFunction=ga.NonlinearFunction,\
                    LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
                chromosome.sort(order='Most important first')  
                ga.n_lin = Single_X_train.shape[1]
            t_sec = time.time() - t
            print("\n", 'Elastic Net worked ', t_sec, 'sec')  
            print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
        if FirstAlgorithm == 'GA':
            print('Genetic Algorithm for all features')
            ga.fit(x_nonlin_train=X_train_nonlin, x_lin_train=Single_X_train, y_train=Y_train,\
                x_nonlin_test=X_test_nonlin, x_lin_test=Single_X_test, y_test=Y_test,\
                idx_nonlin=None, idx_lin=None, VIP_idx_nonlin=None,\
                VIP_idx_lin=None, CrossoverMethod=CrossoverMethod, MutationMethod=MutationMethod,\
                UseNonlinear=UseNonlinear, LinearSolver=LinearSolver,\
                cond=cond, lapack_driver=lapack_driver, NonlinearFunction=NonlinearFunction,\
                nIter=nIter)
            t_sec = time.time() - t
            print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
            chromosome = ga.BestChromosome        
            print("\n", 'Features left for Backward Elimination and Search Alternative = ',\
                  chromosome.Size)
            ga.PlotChromosomes(2, ga.BestChromosomes, XAxis='time', YAxis='R2 Train',\
                PlotType='Scatter', F=F_GA)
        while chromosome.Size > 1:
            if chromosome.Size <= ga.ChromosomeSize:
                chromosome = ga.BestFit(chromosome, x_nonlin=X_train_nonlin, x_lin=Single_X_train,\
                    y=Y_train, verbose=True) # returns ranked and sorted chromosome
# get score for test set                
                chromosome.score(x_nonlin_train=X_train_nonlin, x_lin_train=Single_X_train,\
                    y_train=Y_train, x_nonlin_test=X_test_nonlin, x_lin_test=Single_X_test,\
                    y_test=Y_test, NonlinearFunction=ga.NonlinearFunction,\
                    LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver) 
                chromosome.Origin = 'Best Fit'
                chromosome_copy = copy.deepcopy(chromosome)
                chromosome_copy.print_score()
                ga.DecreasingChromosomes.append(chromosome_copy)
            chromosome = ga.RemoveWorstGene(chromosome, x_nonlin=X_train_nonlin,\
                x_lin=Single_X_train, y=Y_train, verbose=True)
        t_sec = time.time() - t
        ga.PlotChromosomes(3, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='RMSE Train',\
            PlotType='Line', F='Single')
        ga.PlotChromosomes(4, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='R2 Adjusted Train',\
            PlotType='Line', F='Single')
        ga.PlotChromosomes(5, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='Mallow statistics Train',\
            PlotType='Line', F='Single')
        print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
        ga.Results_to_xlsx(F_Out_Single, ga.DecreasingChromosomes, FeaturesNonlinear=FeaturesNonlinear,\
            FeaturesAll=FeaturesAll, FeaturesReduced=FeaturesReduced)        
        for i in ga.DecreasingChromosomes:
            if i.Size == VIP_number:
                VIP_idx_nonlin = i.get_genes_list(Type=1)
                VIP_idx_lin = i.get_genes_list(Type=0)
    else:
        VIP_idx_nonlin = []
        VIP_idx_lin = []
# proceed all features 
    t = time.time()       
    ga = GA(PopulationSize=PopulationSize, ChromosomeSize=ChromosomeSize,\
        MutationProbability=MutationProbability, MutationInterval=MutationInterval,\
        EliteFraction=EliteFraction, MutationCrossoverFraction=MutationCrossoverFraction,\
        CrossoverFractionInterval=CrossoverFractionInterval, IterationPerRecord=IterationPerRecord,\
        StopTime=StopTime, RandomSeed=RandomSeed, verbose=verbose,\
        UseCorrelationMutation=UseCorrelationMutation, MinCorrMutation=MinCorrMutation,\
        UseCorrelationBestFit=UseCorrelationBestFit, MinCorrBestFit=MinCorrBestFit)
# linear only for now
    idx = list(range(0, X_train_lin.shape[1], 1))
    ga.n_lin = X_train_lin.shape[1]
    results = regression.fit_linear(idx, X_train_lin, Y_train, x_test=X_test_lin, y_test=Y_test,\
        MSEall_train=None, MSEall_test=None, normalize=True, LinearSolver='sklearn')
    ga.MSEall_train = results['MSE Train']
    ga.MSEall_test = results['MSE Test']    
    if (FirstAlgorithm == 'ENet'):
        print('Only linear features will be considered')
        Alphas = np.logspace(-7, -3.5, num=100, endpoint=True, base=10.0, dtype=None)
        enet = regression.ENet(L1=0.7, eps=1e-3, nAlphas=100, alphas=Alphas, random_state=None)
        print('Elastic Net Fit for all features')
        print('L1 portion = ', enet.L1)
        print('Epsilon = ', enet.eps)
        print('Number of alphas =', enet.nAlphas)
        print('Number of features go to elastic net regularisation = ', len(FeaturesReduced))
        enet.fit(X_train_lin, Y_train, VIP_idx=VIP_idx_lin, Criterion='Mallow', normalize=True,\
            max_iter=1000, tol=0.0001, cv=None, n_jobs=1, selection='random', verbose=True)
        enet.plot_path(1, FileName=F_ENet)
        idx = enet.idx
        Gene_list = []
        for i in idx:
            Gene_list.append(GA.Gene(i, Type=0))
        chromosome = ga.Chromosome(Gene_list)
        chromosome.erase_score()
        # chromosome contains only linear indices
        # use standardized for training set and regular for testing
        chromosome.score(x_nonlin_train=None, x_lin_train=X_train_lin,\
            y_train=Y_train, x_nonlin_test=None, x_lin_test=X_test_lin,\
            y_test=Y_test, NonlinearFunction=ga.NonlinearFunction,\
            LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
        chromosome.rank_sort(x_nonlin=X_train_nonlin, x_lin=X_train_lin, y=Y_train, NonlinearFunction=ga.NonlinearFunction,\
            LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
        t_sec = time.time() - t
        print("\n", 'Elastic Net worked ', t_sec, 'sec')  
        print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    if FirstAlgorithm == 'GA':
        print('Genetic Algorithm for all features')
        ga.fit(x_nonlin_train=X_train_nonlin, x_lin_train=X_train_lin, y_train=Y_train,\
            x_nonlin_test=X_test_nonlin, x_lin_test=X_test_lin, y_test=Y_test,\
            idx_nonlin=None, idx_lin=None, VIP_idx_nonlin=VIP_idx_nonlin,\
            VIP_idx_lin=VIP_idx_lin, CrossoverMethod=CrossoverMethod, MutationMethod=MutationMethod,\
            UseNonlinear=UseNonlinear, LinearSolver=LinearSolver,\
            cond=cond, lapack_driver=lapack_driver, NonlinearFunction=NonlinearFunction,\
            nIter = nIter)
        t_sec = time.time() - t
        print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
        chromosome = ga.BestChromosome        
        print("\n", 'Features left for Backward Elimination and Search Alternative = ',\
              chromosome.Size)
        ga.PlotChromosomes(2, ga.BestChromosomes, XAxis='time', YAxis='R2 Train',\
            PlotType='Scatter', F=F_GA)
    while chromosome.Size > 1:
        if chromosome.Size <= ga.ChromosomeSize:
            chromosome = ga.BestFit(chromosome, x_nonlin=X_train_nonlin, x_lin=X_train_lin,\
                y=Y_train, verbose=True)
#            chromosome = ga.BestFitTree(chromosome, x_nonlin=None, x_lin=X_train_lin, y=Y_train, verbose=True)
# calculate both train and test set score
            chromosome.score(x_nonlin_train=X_train_nonlin, x_lin_train=X_train_lin,\
                y_train=Y_train, x_nonlin_test=X_test_nonlin, x_lin_test=X_test_lin,\
                y_test=Y_test, NonlinearFunction=ga.NonlinearFunction,\
                LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver) 
            chromosome.Origin = 'Best Fit'
            chromosome_copy = copy.deepcopy(chromosome)
            chromosome_copy.print_score()
            ga.DecreasingChromosomes.append(chromosome_copy)
# chromosome must be sorted before             
        chromosome = ga.RemoveWorstGene(chromosome, x_nonlin=X_train_nonlin,\
            x_lin=X_train_lin, y=Y_train, verbose=True)
    t_sec = time.time() - t
    ga.PlotChromosomes(3, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='RMSE Train',\
        PlotType='Line', F='Default')
    ga.PlotChromosomes(4, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='R2 Adjusted Train',\
        PlotType='Line', F='Default')
    ga.PlotChromosomes(5, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='Mallow statistics Train',\
        PlotType='Line', F='Default')
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    ga.Results_to_xlsx(F_Out, ga.DecreasingChromosomes, FeaturesNonlinear=FeaturesNonlinear,\
            FeaturesAll=FeaturesAll, FeaturesReduced=FeaturesReduced)
    f = open(F_ga_structure, "wb") # save GA structure
    pickle.dump(ga, f, pickle.HIGHEST_PROTOCOL)
    f.close() 
    directory = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    if not os.path.exists(directory):
        os.makedirs(directory)    
    if os.path.isfile('SystemDescriptor.'):
        shutil.copyfile('SystemDescriptor.', directory + '\\' + 'SystemDescriptor.')
    if os.path.isfile(F_MoleculesDescriptor):
        shutil.copyfile(F_MoleculesDescriptor, directory + '\\' + F_MoleculesDescriptor)
    if os.path.isfile('Structure.xlsx'):    
        shutil.copyfile('Structure.xlsx', directory + '\\' + 'Structure.xlsx')
    if os.path.isfile('Linear Features Reduced List.xlsx'): 
        shutil.copyfile('Linear Features Reduced List.xlsx', directory + '\\' +\
                        'Linear Features Reduced List.xlsx')
    if os.path.isfile('Nonlinear Features List.xlsx'): 
        shutil.copyfile('Nonlinear Features List.xlsx', directory + '\\' +\
                        'Nonlinear Features List.xlsx')
    if os.path.isfile(F_Out_Single):
        shutil.move(F_Out_Single, directory + '\\' + F_Out_Single)
    if os.path.isfile(F_Out):
        shutil.move(F_Out, directory + '\\' + F_Out)   
# copy all *.dat files        
    l = os.listdir('./')
    for name in l:
        if name.endswith('.dat'):
            if os.path.isfile(name):
                shutil.copy2(name, directory)  # copy2, copyfile      
# move all *.png images        
    l = os.listdir('./')
    for name in l:
        if name.endswith('.png'):
            if os.path.isfile(name):
                shutil.move(name, directory)

           