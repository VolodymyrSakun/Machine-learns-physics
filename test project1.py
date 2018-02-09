from project1 import library
from project1 import IOfunctions
from project1 import regression
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd

def PlotHistogram(FileName=None, y_true=None, y_pred=None, FigNumber=1,\
        FigSize=(4,3), Bins='auto', xLabel=None, yLabel='Frequency'):
    
    if y_true.size != y_pred.size:
        return False
    error = np.zeros(shape=(y_true.size), dtype=float)
    error[:] = np.abs(y_pred[:] - y_true[:])        
    fig = plt.figure(FigNumber, figsize=FigSize)
    plt.hist(error, bins=Bins)
    if xLabel is not None:
        plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show(fig)
    if FileName is not None:
        F = '{}{}'.format(FileName, '.eps')
        plt.savefig(F, bbox_inches='tight', format='eps', dpi=1000)
        plt.close(fig)         
    return 

# general form
def Plot(CoM=None, E_True=None, nFunctions=None, nPredictors=None, E_Predicted=None,\
         Legend=None, Grid=None, GridTrained=None, TrainedIntervals=None, NontrainedIntervals=None):
    if nFunctions > 3:
        return # now only for 3 functions. if necessary add colors
    marker_energy = '.'
 # point, pixel, circle, star, thin diamond, hexagon, square    
    marker_fun = ['.', ',', 'o', '*', 'd', 'h', 's']
    color_train_energy = 'red'
    color_nontrain_enegry = 'blue'
    color_train_fun = ['magenta', 'violet', 'yellow']
    color_nontrain_fun = ['green', 'black', 'brown']
    if CoM is None:
        return    
    if len(nPredictors) != nFunctions:
        print('Incorrect input, len(nPredictors) != nFunctions')
        return
    if Legend is not None and nPredictors is not None:
        for i in range(0, nFunctions, 1):
            Legend[i+1] = '{}{}{}'.format(Legend[i+1],'. Number of predictors=', nPredictors[i])
    if Legend is None:
        Legend = []
        Legend.append('True Enegry')
        for i in range(0, nFunctions, 1):
            Legend.append('{} {}'.format('Function', i+1))
    if nFunctions == 0:
        E_Predicted = None
    E_Predicted = np.asarray(E_Predicted)
    Size = CoM.size # number of observations
    RMax = max(CoM)
    RMin = min(CoM)
    if Grid is None:
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
#    print('nTrained_bins=', nTrained_bins)
#    print('nNontrained_bins=', nNontrained_bins)
    X_Trained = np.zeros(shape=(nTrained_bins), dtype=float)
    X_Nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
    E_True_bin_trained = np.zeros(shape=(nTrained_bins), dtype=float)
    E_True_bin_nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
    Error_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)
    Error_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
    E_Predicted_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)
    E_Predicted_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
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
        
    XMin = RMin - (RMax-RMin) * 0.02
    XMax = RMax + (RMax-RMin) * 0.02  
    if nNontrained_bins != 0:
        YMax = max(np.max(Error_bin_trained), np.max(Error_bin_nontrained))
        YMin = min(np.min(Error_bin_trained), np.min(Error_bin_nontrained))
    else:        
        YMax = np.max(Error_bin_trained)
        YMin = np.min(Error_bin_trained)
    YMinAdj = YMin - (YMax-YMin) * 0.02
    YMaxAdj = YMax + (YMax-YMin) * 0.02
# plot RMSE
    fig_error = plt.figure(1, figsize=(4,3))
    plt.xlim((XMin, XMax))
    plt.ylim((YMinAdj, YMaxAdj))
    for i in range(0, nFunctions, 1):
        label_trained = '{} {}'.format(Legend[i+1], 'Trained region')
        plt.scatter(X_Trained, Error_bin_trained[i], s=1, c=color_train_fun[i], marker=marker_fun[i], label=label_trained)
        if nNontrained_bins != 0:
            label_nontrained = '{} {}'.format(Legend[i+1], 'non-Trained region')
            plt.scatter(X_Nontrained, Error_bin_nontrained[i], s=1, c=color_nontrain_fun[i], marker=marker_fun[i], label=label_nontrained)
    plt.legend()
    plt.xlabel('Average distance between centers of masses of molecules')
    plt.ylabel('Average bin error')
    Title = 'Average bin error vs. average distance'
    if nPredictors is not None:
        Title = '{}{}{}'.format(Title,'. Number of predictors=', nPredictors)
    F = '{}{}'.format(Title, '.eps')    
    plt.title(Title)
    plt.show(fig_error)
    plt.savefig(F, bbox_inches='tight', format='eps', dpi=1000)
    plt.close(fig_error)
# plot energy
    if nNontrained_bins == 0:
        YMax = max(np.max(E_True_bin_trained), np.max(E_Predicted_bin_trained))
        YMin = min(np.min(E_True_bin_trained), np.min(E_Predicted_bin_trained))
    else:
        YMax = max(np.max(E_True_bin_trained), np.max(E_True_bin_nontrained), np.max(E_Predicted_bin_trained), np.max(E_Predicted_bin_nontrained))
        YMin = min(np.min(E_True_bin_trained), np.min(E_True_bin_nontrained), np.min(E_Predicted_bin_trained), np.min(E_Predicted_bin_nontrained))       
    YMinAdj = YMin - (YMax-YMin) * 0.02
    YMaxAdj = YMax + (YMax-YMin) * 0.02
    fig_energy = plt.figure(2, figsize=(4,3))
    plt.xlim((XMin, XMax))
    plt.ylim((YMinAdj, YMaxAdj))
    label_trained = '{} {}'.format(Legend[0], 'Trained region')
    for i in range(0, len(TrainedIntervals), 1):
        plt.plot(x_trained[i], e_True_bin_trained[i], c=color_train_energy, ms=1, marker=marker_energy, label=label_trained)
    if nNontrained_bins != 0:
        label_nontrained = '{} {}'.format(Legend[0], 'non-Trained region')
        for i in range(0, len(NontrainedIntervals), 1):
            plt.plot(x_nontrained[i], e_True_bin_nontrained[i], c=color_nontrain_enegry, marker=marker_energy, label=label_nontrained)
    for i in range(0, nFunctions, 1):
        label_trained = '{} {}'.format(Legend[i+1], 'Trained region')
        plt.scatter(X_Trained, E_Predicted_bin_trained[i], s=1, c=color_train_fun[i], marker=marker_fun[i], label=label_trained)
        if nNontrained_bins != 0:
            label_nontrained = '{} {}'.format(Legend[i+1], 'non-Trained region')
            plt.scatter(X_Nontrained, E_Predicted_bin_nontrained[i], s=1, c=color_nontrain_fun[i], marker=marker_fun[i], label=label_nontrained)
    plt.legend()
    plt.xlabel('Average distance between centers of masses of molecules')
    plt.ylabel('Average energy per bin')
    Title = 'Average enegry per bin vs. average distance'
    if nPredictors is not None:
        Title = '{}{}{}'.format(Title,'. Number of predictors=', nPredictors)
    F = '{}{}'.format(Title, '.eps')
    plt.title(Title)
    plt.show(fig_energy)
    plt.savefig(F, bbox_inches='tight', format='eps', dpi=1000)
    plt.close(fig_energy)
    return

# general form
def Plot2(CoM=None, E_True=None, nFunctions=None, nPredictors=None, E_Predicted=None,\
         Legend=None, Grid=None, GridTrained=None, TrainedIntervals=None, NontrainedIntervals=None):
    if nFunctions > 3:
        return # now only for 3 functions. if necessary add colors
    marker_energy = '.'
 # point, pixel, circle, star, thin diamond, hexagon, square    
    marker_fun = ['.', ',', 'o', '*', 'd', 'h', 's']
    color_train_energy = 'red'
    color_nontrain_enegry = 'blue'
    color_train_fun = ['magenta', 'violet', 'yellow']
    color_nontrain_fun = ['green', 'black', 'brown']
    if CoM is None:
        return    
    if len(nPredictors) != nFunctions:
        print('Incorrect input, len(nPredictors) != nFunctions')
        return
    if Legend is not None and nPredictors is not None:
        for i in range(0, nFunctions, 1):
            Legend[i+1] = '{}{}{}'.format(Legend[i+1],'. Number of predictors=', nPredictors[i])
    if Legend is None:
        Legend = []
        Legend.append('True Enegry')
        for i in range(0, nFunctions, 1):
            Legend.append('{} {}'.format('Function', i+1))
    if nFunctions == 0:
        E_Predicted = None
    E_Predicted = np.asarray(E_Predicted)
    Size = CoM.size # number of observations
    RMax = max(CoM)
    RMin = min(CoM)
    if Grid is None:
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
#    print('nTrained_bins=', nTrained_bins)
#    print('nNontrained_bins=', nNontrained_bins)
    X_Trained = np.zeros(shape=(nTrained_bins), dtype=float)
    X_Nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
    E_True_bin_trained = np.zeros(shape=(nTrained_bins), dtype=float)
    E_True_bin_nontrained = np.zeros(shape=(nNontrained_bins), dtype=float)
    Error_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)
    Error_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
    E_Predicted_bin_trained = np.zeros(shape=(nFunctions, nTrained_bins), dtype=float)
    E_Predicted_bin_nontrained = np.zeros(shape=(nFunctions, nNontrained_bins), dtype=float)
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
        
    XMin = RMin - (RMax-RMin) * 0.02
    XMax = RMax + (RMax-RMin) * 0.02  
    if nNontrained_bins != 0:
        YMax = max(np.max(Error_bin_trained), np.max(Error_bin_nontrained))
        YMin = min(np.min(Error_bin_trained), np.min(Error_bin_nontrained))
    else:        
        YMax = np.max(Error_bin_trained)
        YMin = np.min(Error_bin_trained)
    YMinAdj = YMin - (YMax-YMin) * 0.02
    YMaxAdj = YMax + (YMax-YMin) * 0.02
# plot RMSE
    fig_error = plt.figure(1, figsize=(4,3))
    plt.xlim((XMin, XMax))
    plt.ylim((YMinAdj, YMaxAdj))
    for i in range(0, nFunctions, 1):
        label_trained = '{} {}'.format(Legend[i+1], 'Trained region')
        plt.scatter(X_Trained, Error_bin_trained[i], s=1, c=color_train_fun[i], marker=marker_fun[i], label=label_trained)
        if nNontrained_bins != 0:
            label_nontrained = '{} {}'.format(Legend[i+1], 'non-Trained region')
            plt.scatter(X_Nontrained, Error_bin_nontrained[i], s=1, c=color_nontrain_fun[i], marker=marker_fun[i], label=label_nontrained)
    plt.legend()
    plt.xlabel('Average distance between centers of masses of molecules')
    plt.ylabel('Average bin error')
    Title = 'Average bin error vs. average distance'
    if nPredictors is not None:
        Title = '{}{}{}'.format(Title,'. Number of predictors=', nPredictors)
    F = '{}{}'.format(Title, '.eps')    
    plt.title(Title)
    plt.show(fig_error)
    plt.savefig(F, bbox_inches='tight', format='eps', dpi=1000)
    plt.close(fig_error)
# plot energy
    if nNontrained_bins == 0:
        YMax = max(np.max(E_True_bin_trained), np.max(E_Predicted_bin_trained))
        YMin = min(np.min(E_True_bin_trained), np.min(E_Predicted_bin_trained))
    else:
        YMax = max(np.max(E_True_bin_trained), np.max(E_True_bin_nontrained), np.max(E_Predicted_bin_trained), np.max(E_Predicted_bin_nontrained))
        YMin = min(np.min(E_True_bin_trained), np.min(E_True_bin_nontrained), np.min(E_Predicted_bin_trained), np.min(E_Predicted_bin_nontrained))       
    YMinAdj = YMin - (YMax-YMin) * 0.02
    YMaxAdj = YMax + (YMax-YMin) * 0.02
    fig_energy = plt.figure(2, figsize=(4,3))
    plt.xlim((XMin, XMax))
    plt.ylim((YMinAdj, YMaxAdj))
    label_trained = '{} {}'.format(Legend[0], 'Trained region')
    for i in range(0, len(TrainedIntervals), 1):
        plt.plot(x_trained[i], e_True_bin_trained[i], c=color_train_energy, ms=1, marker=marker_energy, label=label_trained)
    if nNontrained_bins != 0:
        label_nontrained = '{} {}'.format(Legend[0], 'non-Trained region')
        for i in range(0, len(NontrainedIntervals), 1):
            plt.plot(x_nontrained[i], e_True_bin_nontrained[i], c=color_nontrain_enegry, marker=marker_energy, label=label_nontrained)
    for i in range(0, nFunctions, 1):
        label_trained = '{} {}'.format(Legend[i+1], 'Trained region')
        plt.scatter(X_Trained, E_Predicted_bin_trained[i], s=1, c=color_train_fun[i], marker=marker_fun[i], label=label_trained)
        if nNontrained_bins != 0:
            label_nontrained = '{} {}'.format(Legend[i+1], 'non-Trained region')
            plt.scatter(X_Nontrained, E_Predicted_bin_nontrained[i], s=1, c=color_nontrain_fun[i], marker=marker_fun[i], label=label_nontrained)
    plt.legend()
    plt.xlabel('Average distance between centers of masses of molecules')
    plt.ylabel('Average energy per bin')
    Title = 'Average enegry per bin vs. average distance'
    if nPredictors is not None:
        Title = '{}{}{}'.format(Title,'. Number of predictors=', nPredictors)
    F = '{}{}'.format(Title, '.eps')
    plt.title(Title)
    plt.show(fig_energy)
    plt.savefig(F, bbox_inches='tight', format='eps', dpi=1000)
    plt.close(fig_energy)
    return

Files = {'Response Train': 'ResponseTrain.csv', 'Response Test': 'ResponseTest.csv',\
        'Linear Single Train': 'LinearSingleTrain.csv','Linear Single Test': 'LinearSingleTest.csv',\
        'Linear Double Train': 'LinearDoubleTrain.csv','Linear Double Test': 'LinearDoubleTest.csv',\
        'Linear Triple Train': 'LinearTripleTrain.csv','Linear Triple Test': 'LinearTripleTest.csv',\
        'Exp Single Train D': 'ExpSingleTrainD.csv','Exp Single Test D': 'ExpSingleTestD.csv',\
        'Exp Single Train D^n': 'ExpSingleTrainD^n.csv','Exp Single Test D^n': 'ExpSingleTestD^n.csv',\
        'Exp Double Train D1': 'ExpDoubleTrainD1.csv','Exp Double Test D1': 'ExpDoubleTestD1.csv',\
        'Exp Double Train D2': 'ExpDoubleTrainD2.csv','Exp Double Test D2': 'ExpDoubleTestD2.csv',\
        'Exp Double Train D1^mD2^n': 'ExpDoubleTrainD1^mD2^n.csv','Exp Double Test D1^mD2^n': 'ExpDoubleTestD1^mD2^n.csv',\
        'Gaussian Single Train': 'GaussianSingleTrain.csv', 'Gaussian Single Test': 'GaussianSingleTest.csv',\
        'Features List': 'Features List.xlsx','Structure': 'Structure.xlsx'}

# for set 6
# seed 10 
# ConfidenceInterval=0.95

random_seed = 10

#FilterDataResults = library.FilterData(F_Records='SET 6.x', F_MoleculesDescriptor = 'MoleculesDescriptor.',\
#    TrainIntervals=[(0, 20)], F_Train = 'Training Set.x', F_Test = 'Test Set.x',\
#    D_Train = 'D Train.csv', D_Test = 'D Test.csv',\
#    GridStart = 0, GridEnd = 20, GridSpacing=0.2, ConfidenceInterval=0.95,\
#    TestFraction=0.2, TrainFraction=1, RandomSeed=random_seed)
#
#f, f_old = library.RedirectPrintToFile('FilterDataResults.txt')
#for i, j in FilterDataResults.items():
#    print(i, ':', j)   
#library.RedirectPrintToConsole(f, f_old)
#
#GenerateFeaturesResults = library.GenerateFeatures(FilterDataResults, Files, F_SystemDescriptor='SystemDescriptor.')
#
#IOfunctions.SaveObject('FilterDataResults.dat', FilterDataResults)
#IOfunctions.SaveObject('GenerateFeaturesResults.dat', GenerateFeaturesResults)

FilterDataResults = IOfunctions.LoadObject('FilterDataResults.dat')        
GenerateFeaturesResults = IOfunctions.LoadObject('GenerateFeaturesResults.dat') 

#ga = library.GetFitGA(FilterDataResults, Files, GenerateFeaturesResults, F_xlsx='Fit', F_ENet='ENet path',\
#        F_GA='GA path', UseVIP=False, nVIP=None, FirstAlgorithm='GA',\
#        L1_Single=0.7, eps_single=1e-3, n_alphas_single=100, L1=0.7, eps=1e-3,\
#        n_alphas=100, alpha_grid_start=-7, alpha_grid_end=-3, cv=30, MinChromosomeSize=2,\
#        ChromosomeSize=15, StopTime=600, BestFitStopTime=30, nIter=20, PopulationSize=100,\
#        MutationProbability=0.3, MutationInterval=[1, 4], BestFitMaxQueue=100,\
#        EliteFraction=0.5, MutationCrossoverFraction=0.5, CrossoverFractionInterval=[0.6, 0.4],\
#        UseCorrelationMutation=False, MinCorrMutation=0.8, CrossoverMethod='Random',\
#        MutationMethod='Correlated', LinearSolver='sklearn', cond=1e-03,\
#        lapack_driver='gelsy', UseCorrelationBestFit=False, MinCorrBestFit=0.99,\
#        PrintInterval=10, RandomSeed=random_seed, BestFitPathLen=100, verbose=True)


#gp = library.GetFitGP(Files, GaussianPrecision=5, GaussianStart=0.01, GaussianEnd=20, GaussianLen=5)
# set 6 R2= 0.640574623205 length_scale= 1.5367666815875023 noise_level= 0.0314196466815393
# set 6 R2= 0.642671463024 length_scale= 2.39927628617527 noise_level= 0.01194493099928471

#gp = library.GetFitGP5(Files, length_scale_start=2.399, noise_level_start=0.01194,\
#    length_scale_bounds=(1e-3, 20), noise_level_bounds=(1e-20, 1),\
#    length_scale_inc=0.1, noise_level_inc=0.1, length_scale_inc_min=0.02,\
#    noise_level_inc_min=0.02, simulation=None, random_state=random_seed)
    
#gp = library.GetFitGP2(Files)

#IOfunctions.SaveObject('ga.dat', ga) 
#IOfunctions.SaveObject('gp.dat', gp)   

ga = IOfunctions.LoadObject('ga.dat') 
gp = IOfunctions.LoadObject('gp.dat')   

COM_train = IOfunctions.ReadCSV(FilterDataResults['COM Train'])
COM_test = IOfunctions.ReadCSV(FilterDataResults['COM Test']) 
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
X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    
if (X_LinearSingle_train is not None) and (X_LinearDouble_train is not None) and (X_LinearTriple_train is not None): # all three exist
    X_Linear_train = np.concatenate((X_LinearSingle_train, X_LinearDouble_train, X_LinearTriple_train),axis=1)
    FeaturesLinearAll = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearSingleAll'])
    FeaturesLinearAll.extend(GenerateFeaturesResults['FeaturesLinearDoubleAll'])
    FeaturesLinearAll.extend(GenerateFeaturesResults['FeaturesLinearTripleAll'])
    FeaturesLinearReduced = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearSingleReduced'])
    FeaturesLinearReduced.extend(GenerateFeaturesResults['FeaturesLinearDoubleReduced'])
    FeaturesLinearReduced.extend(GenerateFeaturesResults['FeaturesLinearTripleReduced'])
elif X_LinearSingle_train is not None and X_LinearDouble_train is not None: # single + double exist
    X_Linear_train = np.concatenate((X_LinearSingle_train,X_LinearDouble_train),axis=1)
    FeaturesLinearAll = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearSingleAll'])
    FeaturesLinearAll.extend(GenerateFeaturesResults['FeaturesLinearDoubleAll'])
    FeaturesLinearReduced = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearSingleReduced'])
    FeaturesLinearReduced.extend(GenerateFeaturesResults['FeaturesLinearDoubleReduced'])
elif X_LinearSingle_train is not None and X_LinearDouble_train is None: # only single
    X_Linear_train = X_LinearSingle_train
    FeaturesLinearAll = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearSingleAll'])
    FeaturesLinearReduced = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearSingleReduced'])
elif X_LinearSingle_train is None and X_LinearDouble_train is not None: # only double
    X_Linear_train = X_LinearDouble_train
    FeaturesLinearAll = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearDoubleAll'])
    FeaturesLinearReduced = copy.deepcopy(GenerateFeaturesResults['FeaturesLinearDoubleReduced'])
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

PlotHistogram(FileName='{} {} {}'.format('Gaussian process energy prediction error histogram.',\
    X_Gaussian_test.shape[1], 'predictors'), y_true=y_test_kj, y_pred=y_pred_gp_kj,\
    FigNumber=1, FigSize=(4,3), Bins='auto',\
    xLabel='Gaussian process energy prediction error, kJ/mol', yLabel='Frequency')
        
k = 0
for chromosome in ga.DecreasingChromosomes:
    if k == 1:
        break
    y_pred_ga = chromosome.predict(x_expD=X_ExpSingleD_test,\
        x_expDn=X_ExpSingleDn_test, x_lin=X_Linear_test)
    y_pred_ga_kj = library.HARTREE_TO_KJMOL * y_pred_ga    
    PlotHistogram(FileName='{} {} {}'.format('Linear regression energy prediction error histogram.',\
        chromosome.Size, 'predictors'), y_true=y_test_kj, y_pred=y_pred_ga_kj,\
        FigNumber=1, FigSize=(4,3), Bins='auto',\
        xLabel='Linear regression energy prediction error, kJ/mol', yLabel='Frequency')
    Plot2(CoM=COM_test, E_True=y_test_kj, nFunctions=2, nPredictors=[chromosome.Size, X_Gaussian_test.shape[1]],\
         E_Predicted=[y_pred_ga, y_pred_gp], Legend=None,\
         Grid=FilterDataResults['Test Grid'], GridTrained=FilterDataResults['Train Grid'],\
         TrainedIntervals=FilterDataResults['Train Intervals'], NontrainedIntervals=FilterDataResults['Test Intervals'])
    k += 1

