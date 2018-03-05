from project1 import library
from project1 import IOfunctions
from project1 import regression
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
from matplotlib.mlab import griddata
import os
import re
import datetime
import shutil
import glob
import sklearn.metrics as skm

# return Min and Max for boundaries. *argv - 1D array like 
def get_bounds(*argv, adj=0.02):
    max_list = []
    min_list = []
    for arg in argv:
        if arg is not None:
            max_list.append(np.max(arg))
            min_list.append(np.min(arg))       
    YMax = max(max_list)
    YMin = min(min_list)
    YMinAdj = YMin - (YMax-YMin) * adj
    YMaxAdj = YMax + (YMax-YMin) * adj        
    return YMinAdj, YMaxAdj



def PlotHistogram(FileName=None, y_true=None, y_pred=None, FigNumber=1,\
        FigSize=(4,3), Bins='auto', xLabel=None, yLabel='Frequency', FileFormat='eps'):
    
    if y_true.size != y_pred.size:
        return False
    error = np.zeros(shape=(y_true.size), dtype=float)
#    error[:] = np.abs(y_pred[:] - y_true[:])        
    error[:] = y_pred[:] - y_true[:] 
    fig = plt.figure(FigNumber, figsize=FigSize)
    plt.hist(error, bins=Bins)
    if xLabel is not None:
        plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show(fig)
    if FileName is not None:
        F = '{}{}{}'.format(FileName, '.', FileFormat)
        plt.savefig(F, bbox_inches='tight', format=FileFormat, dpi=1000)
        plt.close(fig)         
    return 

def plot_contour(x, y, z, x_res=100, y_res=100, FileName=None,\
        FileFormat='eps', FigSize=(4,3), xTitle=None, yTitle=None, barTitle=None):
    
    fig = plt.figure(100, figsize=FigSize)
    x_min, x_max = get_bounds(x, adj=0.02)
    y_min, y_max = get_bounds(y, adj=0.02)
    xi = np.linspace(x_min, x_max, x_res)
    yi = np.linspace(y_min, y_max, y_res)
    zi = griddata(x, y, z, xi, yi, interp='linear')
    plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    plt.contourf(xi, yi, zi, 15, vmax=abs(zi).max(), vmin=-abs(zi).max())
    cbar = plt.colorbar()  # draw colorbar
    if barTitle is not None:
        cbar.ax.set_ylabel(barTitle)
    idx_max = np.asarray(z).argmax()
    plt.scatter(x[idx_max], y[idx_max], marker='o', s=5, zorder=10)
    if xTitle is not None:
        plt.xlabel(xTitle)
    if yTitle is not None:
        plt.ylabel(yTitle)    
    plt.show()
    if FileName is not None:
        F = '{}{}{}'.format(FileName, '.', FileFormat)
        plt.savefig(F, bbox_inches='tight', format=FileFormat, dpi=1000)
        plt.close(fig)         
    return

###############################################################################
    
def Plot(CoM=None, E_True=None, nFunctions=None, E_Predicted=None, xLabel='R, CoM (Å)', \
        yEnergyLabel='Average Energy (kJ/mol)', yErrorLabel='Average bin error (kJ/mol)',\
        Legend=None, Grid=None, GridTrained=None, TrainedIntervals=None,\
        NontrainedIntervals=None, F_Error='Error', F_Energy='Energy', figsize=(4, 3),\
        fig_format='eps', marker_size=1, line_width = 0.3, bounds=None):

    """
    first description in Legend is true energy function
    """
    if bounds is not None:
        i = 0
        while len(Grid) > i:
            if bounds[0] > Grid[i][0]:
                del(Grid[i])
                continue
            if bounds[1] < Grid[i][1]:
                del(Grid[i])
                continue            
            i += 1
            
    marker_energy = '.'
# circle, square, star, thin diamond, hexagon,  point, pixel   
    marker_fun = ['o', 's', '*', 'd', 'h', '.', ',']
    color_train_energy = 'red'
    color_nontrain_enegry = 'blue'
    color_nontrain_fun = ['magenta', 'orange', 'violet', ]
    color_train_fun = ['green', 'black', 'brown']  
    if nFunctions == 0:
        E_Predicted = None
    E_Predicted = np.asarray(E_Predicted)
    Size = CoM.size # number of observations    
    F_Error = '{}{}{}'.format(F_Error, '.', fig_format)
    F_Energy = '{}{}{}'.format(F_Energy, '.', fig_format)       
    if Grid is None:
        RMin, RMax = get_bounds(CoM, adj=0)
        nIntervals = 30
        Grid = library.CreateGrid(round(RMin, 2), round(RMax, 2), round((round(RMax, 2) - round(RMin, 2)) / nIntervals, 2))
    else:
        nIntervals = len(Grid)
    if GridTrained is None:
        GridTrained = []
    n = np.zeros(shape=(nIntervals), dtype=float) # n[i] = number of observations in interval    
    X = np.zeros(shape=(nIntervals), dtype=float) # X on plot
    E_True_bin = np.zeros(shape=(nIntervals), dtype=float) 
    Trained = np.empty(shape=(nIntervals), dtype=bool) # True if bin in trained region    
    for i in range(0, Size, 1):# count number of observations in each bin
        j = library.InInterval(CoM[i], Grid)
        if j != -10: # not in any of intervals
            E_True_bin[j] += E_True[i] # cumulative true energy per bin
            n[j] += 1 # number of observations in each bin
            X[j] += CoM[i]
    E_True_bin = np.divide(E_True_bin, n) # average value
    X = np.divide(X, n) # average value
    Error_bin = np.zeros(shape=(nFunctions, nIntervals), dtype=float) # Y axis for error plot
    E_Predicted_bin = np.zeros(shape=(nFunctions, nIntervals), dtype=float) # Y axis for energy plot
    for k in range(0, nFunctions, 1): # for each function
        error = np.zeros(shape=(Size), dtype=float)
        for i in range(0, Size, 1):
            error[i] = abs(E_True[i] - E_Predicted[k, i])
        error_bin = np.zeros(shape=(nIntervals), dtype=float)
        e_Predicted_bin = np.zeros(shape=(nIntervals), dtype=float)  
        for i in range(0, Size, 1):
            j = library.InInterval(CoM[i], Grid)
            if j != -10: # not in any of intervals
                error_bin[j] += error[i] # cumulative error per bin for each function
                e_Predicted_bin[j] += E_Predicted[k, i] # cumulative energy per bin for each functione
        error_bin = np.divide(error_bin, n) # average error per bin
        e_Predicted_bin = np.divide(e_Predicted_bin, n) # averare predicted energy per bin
        Error_bin[k, :] = error_bin[:]
        E_Predicted_bin[k, :] = e_Predicted_bin[:]        
    for i in range(0, nIntervals, 1): # separate trained region from non-trained
        if library.InInterval(X[i], GridTrained) != -10:
            Trained[i] = True
        else:
            Trained[i] = False
    nTrained_bins = np.count_nonzero(Trained) # cannot be 0
    nNontrained_bins = nIntervals - nTrained_bins # can be 0
    X_Trained = np.zeros(shape=(nTrained_bins), dtype=float)
    E_True_bin_trained = np.zeros(shape=(nTrained_bins), dtype=float)
    Error_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)    
    E_Predicted_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)    
    if nNontrained_bins != 0:
        X_Nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
        E_True_bin_nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
        Error_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
        E_Predicted_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
    else:
        X_Nontrained, E_True_bin_nontrained, Error_bin_nontrained,\
            E_Predicted_bin_nontrained = None, None, None, None
    j = 0 # trained index
    k = 0 # nontrained index
# separate trained and nontrained points    
    for i in range(0, nIntervals, 1):
        if Trained[i]:
            X_Trained[j] = X[i]
            E_True_bin_trained[j] = E_True_bin[i]
            for l in range(0, nFunctions, 1):
                Error_bin_trained[l, j] = Error_bin[l, i]
                E_Predicted_bin_trained[l, j] = E_Predicted_bin[l, i]
            j += 1
        else:
            X_Nontrained[k] = X[i]
            E_True_bin_nontrained[k] = E_True_bin[i]
            for l in range(0, nFunctions, 1):
                Error_bin_nontrained[l, k] = Error_bin[l, i]
                E_Predicted_bin_nontrained[l, k] = E_Predicted_bin[l, i]
            k += 1    
    x_trained = []
    e_True_bin_trained = []
    for i in range(0, len(TrainedIntervals), 1):
        x_trained.append([])
        e_True_bin_trained.append([])
    if nNontrained_bins != 0:
        x_nontrained = []
        e_True_bin_nontrained = []
        for i in range(0, len(NontrainedIntervals), 1):
            x_nontrained.append([])
            e_True_bin_nontrained.append([])
    for i in range(0, nTrained_bins, 1):
        j = library.InInterval(X_Trained[i], TrainedIntervals)
        if j != -10:
            x_trained[j].append(X_Trained[i])
            e_True_bin_trained[j].append(E_True_bin_trained[i])
    if nNontrained_bins != 0:
        for i in range(0, nNontrained_bins, 1):
            j = library.InInterval(X_Nontrained[i], NontrainedIntervals)
            if j != -10:
                x_nontrained[j].append(X_Nontrained[i])
                e_True_bin_nontrained[j].append(E_True_bin_nontrained[i])
        

# plot Error
    fig_error = plt.figure(1, figsize=figsize)
    xMin, xMax = get_bounds(Grid, adj=0.02)
    yMin, yMax = get_bounds(Error_bin_trained, Error_bin_nontrained, adj=0.02)    
    plt.xlim((xMin, xMax))
    plt.ylim((yMin, yMax))
    if Legend is None: # assigne some text for empty Legend list
        Legend = []
        Legend.append('{}'.format('Reference energy'))
        for i in range(0, nFunctions, 1):
            Legend.append('{} {}'.format('Function', i+1))
    for i in range(0, nFunctions, 1):
        plt.scatter(X_Trained, Error_bin_trained[i], s=marker_size,\
            c=color_train_fun[i], marker=marker_fun[i], label=Legend[i+1])
        if nNontrained_bins != 0:
            plt.scatter(X_Nontrained, Error_bin_nontrained[i], s=marker_size,\
                c=color_nontrain_fun[i], marker=marker_fun[i], label=None)
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yErrorLabel)
    plt.show(fig_error)
    plt.savefig(F_Error, bbox_inches='tight', format=fig_format, dpi=1000)
    plt.close(fig_error)
# plot Energy. x bounds are the same as prev. plot
    fig_energy = plt.figure(2, figsize=figsize)
    yMin, yMax = get_bounds(E_True_bin_trained, E_True_bin_nontrained,\
        E_Predicted_bin_trained, E_Predicted_bin_nontrained, adj=0.02)
    plt.xlim((xMin, xMax))
    plt.ylim((yMin, yMax))
    for i in range(0, len(TrainedIntervals), 1): # plot true energy on trained region
        if i == 0: # plot legend only once
            legend = Legend[0]
        else:
            legend = None
        plt.plot(x_trained[i], e_True_bin_trained[i], c=color_train_energy,\
            markersize=3, marker=marker_energy, label=legend, lw=line_width)
    if nNontrained_bins != 0: # plot true energy on non-trained region without legend
        for i in range(0, len(NontrainedIntervals), 1):
            plt.plot(x_nontrained[i], e_True_bin_nontrained[i], markersize=3,\
                c=color_nontrain_enegry, marker=marker_energy, label=None, lw=line_width)
    for i in range(0, nFunctions, 1): # plot functions on trained region
        plt.scatter(X_Trained, E_Predicted_bin_trained[i], s=marker_size,\
            c=color_train_fun[i], marker=marker_fun[i], label=Legend[i+1])
        if nNontrained_bins != 0: # plot functions on non-trained region
            plt.scatter(X_Nontrained, E_Predicted_bin_nontrained[i], s=marker_size,\
                c=color_nontrain_fun[i], marker=marker_fun[i], label=None)
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yEnergyLabel)
    plt.show(fig_energy)
    plt.savefig(F_Energy, bbox_inches='tight', format=fig_format, dpi=1000)
    plt.close(fig_energy)
    return

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

Data = {'Proceed fractions': True,\
        'Train intervals': [(2.8, 5.6)],\
        'Grid start': 2.8,\
        'Grid end': 5.6,\
        'Grid spacing': 0.2,\
        'Confidence interval': 0.95,\
        'Test fraction': 0.2,\
        'Train fraction': 1,\
        'Train fractions': np.linspace(0.2, 1, 9, endpoint=True, dtype=float),\
        'Random state': 1001,\
        'Figure file format': 'eps',\
        'Figure size': (4, 3),\
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
        'ENet criterion': 'Mallow',\
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
        'GA mutation probability': 0.3,\
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
        'GP min length scale increment': 0.01,\
        'GP min noise level increment': 0.01,\
# None of number of random climbing hill simulations
        'GP hill simulations': 3}

def Proceed(Fiels, Data):
    # for set 6
    # seed 10; ConfidenceInterval=0.95; GridSpacing=0.2; TrainIntervals=[(2.8, 5), (8, 10)]
    # Best Gaussian R2: 0.448418157908 length_scale: 1.4697829746212416 noise_level: 0.06325484382297103
    
    # set 4
    # ConfidenceInterval=0.85
    # set 2
    # ConfidenceInterval=0.95 MinCorrBestFit=0.9 model='Parent'
    # ChromosomeSize=15, StopTime=600, BestFitStopTime=100, nIter=200
    
    FilterDataDict = library.FilterData(Files, Data)
            
    f, f_old = library.RedirectPrintToFile(Files['Set params'])
    for i, j in FilterDataDict.items():
        print(i, ':', j)   
    library.RedirectPrintToConsole(f, f_old)
    
    FeaturesDict = library.GenerateFeatures(FilterDataDict, Files)
    
    #IOfunctions.SaveObject(Files['Filter data'], FilterDataDict)
    #IOfunctions.SaveObject(Files['Generate features'], FeaturesDict)
    
    #FilterDataDict = IOfunctions.LoadObject(Files['Filter data'])        
    #FeaturesDict = IOfunctions.LoadObject(Files['Generate features']) 
    
    ga = library.GetFitGA(FilterDataDict, Files, Data, FeaturesDict)
    
    gp, Path = library.GetFitGP5(Files, Data)
    
    plot_contour(Path['length_scale'], Path['noise_level'], Path['R2'], 100, 100,\
        FileName=Files['GP path'], FileFormat=Data['Figure file format'],\
        FigSize=Data['Figure size'], xTitle='Length scale',\
        yTitle='Noise level', barTitle='Gaussian R2')
      
    COM_train = IOfunctions.ReadCSV(Files['COM train'])
    COM_test = IOfunctions.ReadCSV(Files['COM test']) 
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])   
    X_LinearSingle_train = IOfunctions.ReadCSV(Files['Linear Single Train'])
    X_LinearSingle_test = IOfunctions.ReadCSV(Files['Linear Single Test'])
    X_LinearDouble_train = IOfunctions.ReadCSV(Files['Linear Double Train'])
    X_LinearDouble_test = IOfunctions.ReadCSV(Files['Linear Double Test'])
    X_LinearTriple_train = IOfunctions.ReadCSV(Files['Linear Triple Train'])
    X_LinearTriple_test = IOfunctions.ReadCSV(Files['Linear Triple Test'])
    X_ExpSingleD_train = IOfunctions.ReadCSV(Files['Exp Single Train D'])
    X_ExpSingleDn_train = IOfunctions.ReadCSV(Files['Exp Single Train D^n'])
    X_ExpSingleD_test = IOfunctions.ReadCSV(Files['Exp Single Test D'])
    X_ExpSingleDn_test = IOfunctions.ReadCSV(Files['Exp Single Test D^n'])    
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Test'])
        
    y_pred = gp.predict(X_Gaussian_test)
    ga.gp_MSE = skm.mean_squared_error(Y_test, y_pred)
    ga.gp_R2 = gp.score(X_Gaussian_test, Y_test)
    IOfunctions.SaveObject(Files['GA object'], ga)    
    #ga = IOfunctions.LoadObject(Files['GA object'])
    IOfunctions.SaveObject(Files['GP object'], gp) 
    #gp = IOfunctions.LoadObject(Files['GP object']) 

    
    if (X_LinearSingle_train is not None) and (X_LinearDouble_train is not None) and (X_LinearTriple_train is not None): # all three exist
        X_Linear_train = np.concatenate((X_LinearSingle_train, X_LinearDouble_train, X_LinearTriple_train),axis=1)
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearSingleAll'])
        FeaturesLinearAll.extend(FeaturesDict['FeaturesLinearDoubleAll'])
        FeaturesLinearAll.extend(FeaturesDict['FeaturesLinearTripleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearSingleReduced'])
        FeaturesLinearReduced.extend(FeaturesDict['FeaturesLinearDoubleReduced'])
        FeaturesLinearReduced.extend(FeaturesDict['FeaturesLinearTripleReduced'])
    elif X_LinearSingle_train is not None and X_LinearDouble_train is not None: # single + double exist
        X_Linear_train = np.concatenate((X_LinearSingle_train,X_LinearDouble_train),axis=1)
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearSingleAll'])
        FeaturesLinearAll.extend(FeaturesDict['FeaturesLinearDoubleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearSingleReduced'])
        FeaturesLinearReduced.extend(FeaturesDict['FeaturesLinearDoubleReduced'])
    elif X_LinearSingle_train is not None and X_LinearDouble_train is None: # only single
        X_Linear_train = X_LinearSingle_train
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearSingleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearSingleReduced'])
    elif X_LinearSingle_train is None and X_LinearDouble_train is not None: # only double
        X_Linear_train = X_LinearDouble_train
        FeaturesLinearAll = copy.deepcopy(FeaturesDict['FeaturesLinearDoubleAll'])
        FeaturesLinearReduced = copy.deepcopy(FeaturesDict['FeaturesLinearDoubleReduced'])
    else: # no linear features
        X_Linear_train = None
        FeaturesLinearAll = None
        FeaturesLinearReduced = None
    if (X_LinearSingle_test is not None) and (X_LinearDouble_test is not None) and (X_LinearTriple_test is not None): # all exist
        X_Linear_test = np.concatenate((X_LinearSingle_test,X_LinearDouble_test,X_LinearTriple_test),axis=1)
    elif X_LinearSingle_test is not None and X_LinearDouble_test is not None: # single + double exist
        X_Linear_test = np.concatenate((X_LinearSingle_test,X_LinearDouble_test),axis=1)
    elif X_LinearSingle_test is not None and X_LinearDouble_test is None: # only single
        X_Linear_test = X_LinearSingle_test
    elif X_LinearSingle_test is None and X_LinearDouble_test is not None: # only double
        X_Linear_test = X_LinearDouble_test
    else: # no linear features
        X_Linear_test = None   
    
    y_pred_gp = gp.predict(X_Gaussian_test)
    # converte energy to kJ / mol
    y_test_kj = library.HARTREE_TO_KJMOL * Y_test
    y_pred_gp_kj = library.HARTREE_TO_KJMOL * y_pred_gp
    
    PlotHistogram(FileName='{} {} {}'.format(Files['GP energy error histogram'],\
        X_Gaussian_test.shape[1], 'predictors'), y_true=y_test_kj, y_pred=y_pred_gp_kj,\
        FigNumber=1, FigSize=(4,3), Bins='auto', FileFormat='eps',\
        xLabel='GP energy error, kJ/mol', yLabel='Frequency')
            
    for chromosome in ga.DecreasingChromosomes:
        y_pred_ga = chromosome.predict(x_expD=X_ExpSingleD_test,\
            x_expDn=X_ExpSingleDn_test, x_lin=X_Linear_test)
        y_pred_ga_kj = library.HARTREE_TO_KJMOL * y_pred_ga    
        PlotHistogram(FileName='{} {} {}'.format(Files['GA energy error histogram'],\
            chromosome.Size, 'predictors'), y_true=y_test_kj, y_pred=y_pred_ga_kj,\
            FigNumber=2, FigSize=(4, 3), Bins='auto', FileFormat='eps',\
            xLabel='GA energy error, kJ/mol', yLabel='Frequency')
        Plot(CoM=COM_test, E_True=y_test_kj, nFunctions=2, xLabel='R, CoM (Å)', \
            yErrorLabel='Average Error (kJ/mol)', yEnergyLabel='Average Energy (kJ/mol)',\
            E_Predicted=[y_pred_ga_kj, y_pred_gp_kj],\
            Legend=['Reference', 'GA', 'GP'], Grid=FilterDataDict['Test Grid'],\
            GridTrained=FilterDataDict['Train Grid'],\
            TrainedIntervals=FilterDataDict['Train Intervals'],\
            NontrainedIntervals=FilterDataDict['Test Intervals'],\
            F_Error='{} {} {}'.format(Files['Plot error'], chromosome.Size, 'predictors'),\
            F_Energy='{} {} {}'.format(Files['Plot energy'], chromosome.Size, 'predictors'),\
            figsize=(4, 3), fig_format='eps', marker_size=5,\
            line_width = 1, bounds=(0 ,20))

    return FilterDataDict, FeaturesDict

if not Data['Proceed fractions']:
    fractions = [Data['Train fraction']] # one fit
else:
    fractions = Data['Train fractions']# full analysis
for fraction in fractions:
    Data['Train fraction'] = fraction
    FilterDataDict, FeaturesDict = Proceed(Files, Data)
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
    for file in glob.glob('*.eps'):
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
        library.Print('Nunber of catalogs not equal to number of fractions', library.RED)
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
        yMin, yMax = get_bounds(y_rmse_ga[:, j], y_rmse_gp[:, j], adj=0.02)
        xMin, xMax = get_bounds(x, adj=0.02)        
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
        yMin, yMax = get_bounds(y_R2_ga[:, j], y_R2_gp[:, j], adj=0.02)
        xMin, xMax = get_bounds(x, adj=0.02)        
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
    for file in glob.glob('*.eps'):
        files.append('{}{}{}'.format(parent_dir, '\\', file)) # all plots         

    for file in files: # move all plots 
        shutil.move(file, main_dir)