from project1 import library
from project1 import IOfunctions
from project1 import regression
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
    
# return YMin, YMax for boundaries. *argv - lists of y values
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
            ms=marker_size, marker=marker_energy, label=legend, lw=line_width)
    if nNontrained_bins != 0: # plot true energy on non-trained region without legend
        for i in range(0, len(NontrainedIntervals), 1):
            plt.plot(x_nontrained[i], e_True_bin_nontrained[i], ms=marker_size,\
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
# set 4
# ConfidenceInterval=0.85
# set 2
# ConfidenceInterval=0.95 MinCorrBestFit=0.9 model='Parent'
# ChromosomeSize=15, StopTime=600, BestFitStopTime=100, nIter=200

random_seed = 10

FilterDataResults = library.FilterData(F_Records='SET 2.x', F_MoleculesDescriptor = 'MoleculesDescriptor.',\
    TrainIntervals=[(0, 20)], F_Train = 'Training Set.x', F_Test = 'Test Set.x',\
    D_Train = 'D Train.csv', D_Test = 'D Test.csv',\
    GridStart = 0, GridEnd = 20, GridSpacing=0.2, ConfidenceInterval=0.95,\
    TestFraction=0.2, TrainFraction=1, RandomSeed=random_seed)

f, f_old = library.RedirectPrintToFile('FilterDataResults.txt')
for i, j in FilterDataResults.items():
    print(i, ':', j)   
library.RedirectPrintToConsole(f, f_old)

GenerateFeaturesResults = library.GenerateFeatures(FilterDataResults, Files, F_SystemDescriptor='SystemDescriptor.')

IOfunctions.SaveObject('FilterDataResults.dat', FilterDataResults)
IOfunctions.SaveObject('GenerateFeaturesResults.dat', GenerateFeaturesResults)

FilterDataResults = IOfunctions.LoadObject('FilterDataResults.dat')        
GenerateFeaturesResults = IOfunctions.LoadObject('GenerateFeaturesResults.dat') 

ga = library.GetFitGA(FilterDataResults, Files, GenerateFeaturesResults, F_xlsx='Fit', F_ENet='ENet path',\
        F_GA='GA path', UseVIP=False, nVIP=6, FirstAlgorithm='GA', goal=1e-18,\
        L1_Single=0.7, eps_single=1e-3, n_alphas_single=100, L1=0.7, eps=1e-3,\
        n_alphas=100, alpha_grid_start=-7, alpha_grid_end=-3, cv=30, MinChromosomeSize=2,\
        ChromosomeSize=15, StopTime=600, BestFitStopTime=200, nIter=200, PopulationSize=100,\
        MutationProbability=0.3, MutationInterval=[1, 4], BestFitMaxQueue=100,\
        EliteFraction=0.5, MutationCrossoverFraction=0.5, CrossoverFractionInterval=[0.6, 0.4],\
        UseCorrelationMutation=False, MinCorrMutation=0.8, CrossoverMethod='Random',\
        MutationMethod='Random', LinearSolver='sklearn', cond=1e-03,\
        lapack_driver='gelsy', UseCorrelationBestFit=True, MinCorrBestFit=0.9,\
        PrintInterval=10, RandomSeed=random_seed, model='Parent', verbose=True)

# set 6 R2= 0.640574623205 length_scale= 1.5367666815875023 noise_level= 0.0314196466815393
# set 6 R2= 0.642671463024 length_scale= 2.39927628617527 noise_level= 0.01194493099928471

gp = library.GetFitGP5(Files, length_scale_start=2.399, noise_level_start=0.01194,\
    length_scale_bounds=(1e-3, 20), noise_level_bounds=(1e-20, 1),\
    length_scale_inc=0.1, noise_level_inc=0.1, length_scale_inc_min=0.02,\
    noise_level_inc_min=0.02, simulation=None, random_state=random_seed)

IOfunctions.SaveObject('ga.dat', ga) 
IOfunctions.SaveObject('gp.dat', gp)   

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

PlotHistogram(FileName='{} {} {}'.format('GP energy error histogram.',\
    X_Gaussian_test.shape[1], 'predictors'), y_true=y_test_kj, y_pred=y_pred_gp_kj,\
    FigNumber=1, FigSize=(4,3), Bins='auto',\
    xLabel='GP energy error, kJ/mol', yLabel='Frequency')
        
k = 0
for chromosome in ga.DecreasingChromosomes:
#    if k == 1:
#        break
    y_pred_ga = chromosome.predict(x_expD=X_ExpSingleD_test,\
        x_expDn=X_ExpSingleDn_test, x_lin=X_Linear_test)
    y_pred_ga_kj = library.HARTREE_TO_KJMOL * y_pred_ga    
    PlotHistogram(FileName='{} {} {}'.format('GA energy error histogram.',\
        chromosome.Size, 'predictors'), y_true=y_test_kj, y_pred=y_pred_ga_kj,\
        FigNumber=1, FigSize=(4, 3), Bins='auto',\
        xLabel='GA energy error, kJ/mol', yLabel='Frequency')
    Plot(CoM=COM_test, E_True=y_test_kj, nFunctions=2, xLabel='R, CoM (Å)', \
        yErrorLabel='Average Energy (kJ/mol)', E_Predicted=[y_pred_ga_kj, y_pred_gp_kj],\
        Legend=['Reference', 'GA', 'GP'], Grid=FilterDataResults['Test Grid'],\
        GridTrained=FilterDataResults['Train Grid'],\
        TrainedIntervals=FilterDataResults['Train Intervals'],\
        NontrainedIntervals=FilterDataResults['Test Intervals'],\
        F_Error='{} {} {}'.format('Error.',chromosome.Size, 'predictors'),\
        F_Energy='{} {} {}'.format('Energy.',chromosome.Size, 'predictors'),\
        figsize=(4, 3), fig_format='eps', marker_size=1,\
        line_width = 0.3, bounds=(0 ,20))
    k += 1


        