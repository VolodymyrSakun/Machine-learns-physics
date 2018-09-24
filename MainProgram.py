from project1 import library
from project1 import IOfunctions
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import datetime
import shutil
import glob

if __name__ == '__main__':
    
    Files = {'Response Train': 'ResponseTrain.csv',\
            'Response Test': 'ResponseTest.csv',\
            'Linear Single Train': 'LinearSingleTrain.csv',\
            'Linear Single Test': 'LinearSingleTest.csv',\
            'Linear Double Train': 'LinearDoubleTrain.csv',\
            'Linear Double Test': 'LinearDoubleTest.csv',\
            'Linear Triple Train': 'LinearTripleTrain.csv',\
            'Linear Triple Test': 'LinearTripleTest.csv',\
            'Exp Single Train D': 'ExpSingleTrainD.csv',\
            'Exp Single Test D': 'ExpSingleTestD.csv',\
            'Exp Single Train D^n': 'ExpSingleTrainD^n.csv',\
            'Exp Single Test D^n': 'ExpSingleTestD^n.csv',\
            'Exp Double Train D1': 'ExpDoubleTrainD1.csv',\
            'Exp Double Test D1': 'ExpDoubleTestD1.csv',\
            'Exp Double Train D2': 'ExpDoubleTrainD2.csv',\
            'Exp Double Test D2': 'ExpDoubleTestD2.csv',\
            'Exp Double Train D1^mD2^n': 'ExpDoubleTrainD1^mD2^n.csv',\
            'Exp Double Test D1^mD2^n': 'ExpDoubleTestD1^mD2^n.csv',\
            'Gaussian Train': 'GaussianTrain.csv',\
            'Gaussian Test': 'GaussianTest.csv',\
            'Structure': 'Structure',\
            'Set': 'SET 6.x',\
            'System descriptor': 'SystemDescriptor.',\
            'Training set': 'Training Set.x',\
            'Test set': 'Test Set.x',\
            'COM train': 'COM Train.csv',\
            'COM test': 'COM Test.csv',\
            'Set params': 'SetParams.txt',\
            'Filter data': 'FilterData.dat',\
            'Generate features': 'GenerateFeatures.dat',\
            'Fit': 'Fit',\
            'GA object': 'ga.dat',\
            'GP object': 'gp.dat',\
    # plots        
            'ENet path single': 'ENetPathSingle',\
            'ENet path': 'ENetPath',\
            'GA path': 'GApath',\
            'GP path': 'GPpath',\
            'GA path RMSE single': 'GA_PATH_RMSE_Single',\
            'GA path R2 single': 'GA_PATH_R2_Single',\
            'BF path RMSE single': 'BF_PATH_RMSE_Single',\
            'GA path RMSE': 'GA_PATH_RMSE',\
            'GA path R2': 'GA_PATH_R2',\
            'BF path RMSE': 'BF_PATH_RMSE',\
            'GP energy error histogram': 'GP_energy_error_histogram',\
            'GA energy error histogram': 'GA_energy_error_histogram',\
            'Plot error': 'Error',\
            'Plot energy': 'Energy'}
    
    Data = {'Proceed fractions': False,\
            'Train intervals': [(2.8, 5.6)],\
            'Grid start': 2.8,\
            'Grid end': 5.6,\
            'Grid spacing': 0.2,\
            'Confidence interval': 0.95,\
            'Test fraction': 0.2,\
            'Train fraction': 1,\
            'Train fractions': np.linspace(0.2, 1, 9, endpoint=True, dtype=float),\
            'Random state': 1001,\
            'Figure file format': 'png',\
            'Figure size': (19, 10),\
            'Use VIP': False,\
            'Number of VIP features': 5,\
            # ENet or GA
            'First algorithm': 'GA',\
    # 'sklearn', 'scipy', 'statsmodels': which library to use for OLS. 'statsmodels' also provides p-values
            'LR linear solver': 'sklearn',\
    # for 'scipy' solver      
            'LR scipy condition': 1e-03,\
    # 'gelsd', 'gelsy', 'gelss', for scipy solver only
            'LR scipy driver': 'gelsy',\
    ################################# Elastic Net parameters ######################        
    # 'Mallow' - mallow statistics selection criteria, 'MSE' - best mse, 'CV' - for ElasticNetCV
            'ENet criterion': 'MSE',\
    # Elastic net parameters for single distances if VIP is to be used        
    # L1 / L2 ration        
            'ENet ratio single': 0.7,\
    # number of L1 constants for enet_path
            'ENet number of alphas single': 100,\
    # log10 scale min L1. -2 corresponds to 0.01
            'ENet alpha min single': -7,\
    # log10 scale max L1. -2 corresponds to 0.01
            'ENet alpha max single': -3,\
    # number of cross validation. only for 'CV' criterion        
            'ENet cv number single': 30,\
            'ENet max number of iterations single': 1000,\
    # Elastic net parameters for all cases except single for VIP        
    # L1 / L2 ration        
            'ENet ratio': 0.7,\
    # number of L1 constants for enet_path
            'ENet number of alphas': 100,\
    # log10 scale min L1. -2 corresponds to 0.01
            'ENet alpha min': -7,\
    # log10 scale max L1. -2 corresponds to 0.01
            'ENet alpha max': -3,\
    # number of cross validation. only for 'CV' criterion        
            'ENet cv number': 30,\
            'ENet max number of iterations': 1000,\
            'ENet verbose': True,\
    ###################### Genetic algorithm parameters ###########################
            'GA population size': 100,\
            'GA chromosome size': 15,\
            'GA stop time': 600,\
            'GA max generations': 200,\
            'GA mutation probability': 0.1,\
            'GA mutation interval': [1, 4],\
            'GA elite fraction': 0.5,\
            'GA mutation crossover fraction': 0.5,\
            'GA crossover fraction interval': [0.6, 0.4],\
            'GA use correlation for mutation': False,\
    # applicable if 'GA use correlation for mutation' is True
            'GA min correlation for mutation': 0.8,\
    # 'Random' or 'Best' - rank genes in each chromosome and take most important for crossover
            'GA crossover method': 'Random',\
    # 'Random' or 'Correlated' - pool with candidates for possible replacement created through correlation matrix rule
            'GA mutation method': 'Random',\
            'GA verbose': True,\
    # if verbose, every n generations print
            'GA generations per output': 10,\
    # minimal chromosome size for backwards elimination
            'BE min chromosome size': 2,\
            'BE verbose': True,\
    # A* - best first search tree algorithm parameters
    # A* stop when its fitness function reaches this number        
            'A goal': 1e-18,\
            'A stop time': 200,\
            'A max queue': 1000,\
            'A use correlation': False,\
            'A min correlation': 0.9,\
            'A verbose': True,\
    # 'Fast' - accept only child better than current best,
    # 'Parent' - all children that better than parent,
    # 'Level' - accept child that better then best on its level,
    # 'Level and Parent' - useless
    # 'Slow' - bredth first search
            'A selection criterion': 'Level',\
    ######################## Gaussian process parameters ##########################   
            'GP initial length scale': 1,\
            'GP initial noise level': 1e-5,\
            'GP length scale bounds': (1e-3, 10),\
            'GP noise level bounds': (1e-20, 0.6),\
    # fraction of interval determined by bounds        
            'GP length scale increment': 0.1,\
            'GP noise level increment': 0.1,\
            'GP min length scale increment': 0.1,\
            'GP min noise level increment': 0.1,\
    # None of number of random climbing hill simulations
            'GP hill simulations': 0}
    
# main block
    
    if not Data['Proceed fractions']:
        fractions = [Data['Train fraction']] # one fit
    else:
        fractions = Data['Train fractions']# full analysis
    for fraction in fractions:
        Data['Train fraction'] = fraction
        FilterDataDict, FeaturesDict = library.Proceed(Files, Data)
        if fraction == fractions[0]: # first run
            parent_dir = os.getcwd() # get current directory
    # generate string subdir
            main_dir = '{} {} {} {} {}'.format(FeaturesDict['System']['nMolecules'], 'molecules', FilterDataDict['Initial dataset'].split(".")[0], re.sub('\,|\[|\]|\;|\:', '', str(FilterDataDict['Train Intervals'])), datetime.datetime.now().strftime("%H-%M %B %d %Y"))
            os.mkdir(main_dir) # make subdir
            main_dir = '{}{}{}'.format(parent_dir, '\\', main_dir) # main_dir full path  
            subdirs = []
        subdir = '{:03.0f}{}'.format(Data['Train fraction']*100, '%') # generate string for subdir
        subdir = '{}{}{}'.format(main_dir, '\\', subdir) # subdir full path   
        os.mkdir(subdir) # make subdir
        subdirs.append(subdir) 
        files = [] # move files into subdir
        for file in glob.glob('{}{}'.format('*.', Data['Figure file format'])):
            files.append('{}{}{}'.format(parent_dir, '\\', file)) # all plots    
        files.append('{}{}{}'.format(parent_dir, '\\', Files['GA object'])) # ga.dat
        files.append('{}{}{}'.format(parent_dir, '\\', Files['GP object'])) # gp.dat
        files.append('{}{}{}'.format(parent_dir, '\\', Files['Set params'])) # txt
        files.append('{}{}{}{}'.format(parent_dir, '\\', Files['Structure'], '.xlsx')) # structure xlsx
        for file in glob.glob('{}{}'.format(Files['Fit'], '*.xlsx')): # fit results xlsx
            files.append('{}{}{}'.format(parent_dir, '\\', file))
        i = 0 # check if files exist
        while i < len(files):
            if os.path.exists(files[i]):
                i += 1
                continue
            else:
                del(files[i]) # erase from list if does not exist
        for file in files: # move files
            shutil.move(file, subdir)
            
    ########################### plot goodness of fit vs. fracrion of training poins
    if Data['Proceed fractions']: 
        if len(subdirs) != len(Data['Train fractions']):
            library.Print('Nunber of catalogs not equal to number of fractions', color=library.RED)
            # quit()
    
        ga = IOfunctions.LoadObject('{}{}{}'.format(subdirs[0], '\\', Files['GA object']))
        nPlots = len(ga.DecreasingChromosomes) # number of plots
        nPredictors = []
        
        x = np.zeros(shape=(len(Data['Train fractions'])), dtype=int)
        for i in range(0, x.size, 1):
            x[i] = int(Data['Train fractions'][i]*100)
        y_rmse_ga = np.zeros(shape=(x.size, nPlots), dtype=float)
        y_R2_ga = np.zeros(shape=(x.size, nPlots), dtype=float)
        y_rmse_gp = np.zeros(shape=(x.size, nPlots), dtype=float)
        y_R2_gp = np.zeros(shape=(x.size, nPlots), dtype=float)
        for i in range(0, len(subdirs), 1): # x 
            subdir = subdirs[i]
            ga = IOfunctions.LoadObject('{}{}{}'.format(subdir, '\\', Files['GA object']))
            gp = IOfunctions.LoadObject('{}{}{}'.format(subdir, '\\', Files['GP object']))        
            for j in range(0, nPlots, 1):
                y_rmse_ga[i, j] = library.HARTREE_TO_KJMOL * np.sqrt(ga.DecreasingChromosomes[j].MSE_Test) # GA RMSE
                y_R2_ga[i, j] = ga.DecreasingChromosomes[j].R2_Test # GA R2
                y_rmse_gp[i, j] = library.HARTREE_TO_KJMOL * np.sqrt(ga.gp_MSE) # gaussian RMSE
                y_R2_gp[i, j] = ga.gp_R2 # gaussian R2
                if i == 0:
                    nPredictors.append(ga.DecreasingChromosomes[j].Size)
    
    # plot fitness vs. % of training set
        color_train_energy = 'red'
        marker_energy = '.'      
        for j in range(0, nPlots, 1):
            fig = plt.figure(j, figsize=Data['Figure size'])    
            yMin, yMax = library.get_bounds(y_rmse_ga[:, j], y_rmse_gp[:, j], adj=0.02)
            xMin, xMax = library.get_bounds(x, adj=0.02)        
            plt.xlim((xMin, xMax))
            plt.ylim((yMin, yMax))
            plt.plot(x, y_rmse_ga[:, j], c='red',\
                markersize=5, marker='.', label='GA', lw=0.5)
            plt.plot(x, y_rmse_gp[:, j], c='blue',\
                markersize=5, marker='.', label='GP', lw=0.5)
            plt.legend()
            plt.xlabel('% of training set used')
            plt.ylabel('Average error (kJ/mol)')
            plt.show(fig)
            plt.savefig('{} {}{}{}'.format(nPredictors[j], 'predictors. RMSE vs. percentage of training set used', '.', Data['Figure file format']), bbox_inches='tight', format=Data['Figure file format'], dpi=1000)
            plt.close(fig)
    
            fig = plt.figure(j, figsize=Data['Figure size'])    
            yMin, yMax = library.get_bounds(y_R2_ga[:, j], y_R2_gp[:, j], adj=0.02)
            xMin, xMax = library.get_bounds(x, adj=0.02)        
            plt.xlim((xMin, xMax))
            plt.ylim((yMin, yMax))
            plt.plot(x, y_R2_ga[:, j], c='red',\
                markersize=5, marker='.', label='GA', lw=0.5)
            plt.plot(x, y_R2_gp[:, j], c='blue',\
                markersize=5, marker='.', label='GP', lw=0.5)
            plt.legend()
            plt.xlabel('% of training set used')
            plt.ylabel('Coefficient of determination R2')
            plt.show(fig)
            plt.savefig('{} {}{}{}'.format(nPredictors[j], 'predictors. R2 vs. percentage of training set used', '.', Data['Figure file format']), bbox_inches='tight', format=Data['Figure file format'], dpi=1000)
            plt.close(fig)
    
        files = [] # move plots into subdir
        for file in glob.glob('{}{}'.format('*.', Data['Figure file format'])):
            files.append('{}{}{}'.format(parent_dir, '\\', file)) # all plots         
    
        for file in files: # move all plots 
            shutil.move(file, main_dir)