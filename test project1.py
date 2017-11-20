from project1 import library
from project1 import IOfunctions
import numpy as np
import matplotlib.pyplot as plt

def PlotHistogram(Legend=None, y_true=None, y_pred=None, PlotType='Absolute',\
        nPredictors=None, FigNumber=1, FigSize=(19,10), Bins='auto'):
    if y_true.size != y_pred.size:
        return False
    error = np.zeros(shape=(y_true.size), dtype=float)
    error[:] = np.abs(y_pred[:] - y_true[:])        
    if PlotType == 'Relative':
        error = np.abs(np.divide(error, y_true))
    fig = plt.figure(FigNumber, figsize=FigSize)
    plt.hist(error, bins=Bins)
    plt.ylabel('Frequency')
    XLabel = '{} {}'.format(PlotType, 'Error')
    Title = '{} {}'.format(XLabel, 'Histogram')
    if nPredictors is not None:
        Title = '{}{}{}'.format(Title, '. Number of predictors=', nPredictors)
    if Legend is not None:
        Title = '{}{} {}'.format(Legend, '.', Title)
    plt.xlabel(XLabel)
    plt.title(Title)
    plt.show(fig)
    F_png = '{}{}'.format(Title, '.png')
    plt.savefig(F_png, bbox_inches='tight')
    plt.close(fig)         
    return True

def Plot(CoM=None, E_True=None, nFunctions=None, nPredictors=None, E_Predicted=None,\
         Legend=None, Grid=None, GridTrained=None, TrainedIntervals=None, NontrainedIntervals=None):
    if nFunctions > 3:
        return # now only for 3 functions. if necessary add colors
    marker_energy = '.'
    marker_fun = ['*', 'd', 'h']
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
# plot error
    fig_error = plt.figure(1, figsize=(19,10))
    plt.xlim((XMin, XMax))
    plt.ylim((YMinAdj, YMaxAdj))
    for i in range(0, nFunctions, 1):
        label_trained = '{} {}'.format(Legend[i+1], 'Trained region')
        plt.scatter(X_Trained, Error_bin_trained[i], c=color_train_fun[i], marker=marker_fun[i], label=label_trained)
        if nNontrained_bins != 0:
            label_nontrained = '{} {}'.format(Legend[i+1], 'non-Trained region')
            plt.scatter(X_Nontrained, Error_bin_nontrained[i], c=color_nontrain_fun[i], marker=marker_fun[i], label=label_nontrained)
    plt.legend()
    plt.xlabel('Average distance between centers of masses of molecules')
    plt.ylabel('Average bin error')
    Title = 'Average bin error vs. average distance'
    if nPredictors is not None:
        Title = '{}{}{}'.format(Title,'. Number of predictors=', nPredictors)
    F = '{}{}'.format(Title, '.png')    
    plt.title(Title)
    plt.show(fig_error)
    plt.savefig(F, bbox_inches='tight')
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
    fig_energy = plt.figure(2, figsize=(19,10))
    plt.xlim((XMin, XMax))
    plt.ylim((YMinAdj, YMaxAdj))
    label_trained = '{} {}'.format(Legend[0], 'Trained region')
    for i in range(0, len(TrainedIntervals), 1):
        plt.plot(x_trained[i], e_True_bin_trained[i], c=color_train_energy, marker=marker_energy, label=label_trained)
    if nNontrained_bins != 0:
        label_nontrained = '{} {}'.format(Legend[0], 'non-Trained region')
        for i in range(0, len(NontrainedIntervals), 1):
            plt.plot(x_nontrained[i], e_True_bin_nontrained[i], c=color_nontrain_enegry, marker=marker_energy, label=label_nontrained)
    for i in range(0, nFunctions, 1):
        label_trained = '{} {}'.format(Legend[i+1], 'Trained region')
        plt.scatter(X_Trained, E_Predicted_bin_trained[i], c=color_train_fun[i], marker=marker_fun[i], label=label_trained)
        if nNontrained_bins != 0:
            label_nontrained = '{} {}'.format(Legend[i+1], 'non-Trained region')
            plt.scatter(X_Nontrained, E_Predicted_bin_nontrained[i], c=color_nontrain_fun[i], marker=marker_fun[i], label=label_nontrained)
    plt.legend()
    plt.xlabel('Average distance between centers of masses of molecules')
    plt.ylabel('Average energy per bin')
    Title = 'Average enegry per bin vs. average distance'
    if nPredictors is not None:
        Title = '{}{}{}'.format(Title,'. Number of predictors=', nPredictors)
    F = '{}{}'.format(Title, '.png')
    plt.title(Title)
    plt.show(fig_energy)
    plt.savefig(F, bbox_inches='tight')
    plt.close(fig_energy)
    return

Files = {'Response Train': 'ResponseTrain.csv', 'Response Test': 'ResponseTest.csv',\
        'Linear Single Train': 'LinearSingleTrain.csv','Linear Single Test': 'LinearSingleTest.csv',\
        'Linear Double Train': 'LinearDoubleTrain.csv','Linear Double Test': 'LinearDoubleTest.csv',\
        'Exp Single Train D': 'ExpSingleTrainD.csv','Exp Single Test D': 'ExpSingleTestD.csv',\
        'Exp Single Train D^n': 'ExpSingleTrainD^n.csv','Exp Single Test D^n': 'ExpSingleTestD^n.csv',\
        'Exp Double Train D1': 'ExpDoubleTrainD1.csv','Exp Double Test D1': 'ExpDoubleTestD1.csv',\
        'Exp Double Train D2': 'ExpDoubleTrainD2.csv','Exp Double Test D2': 'ExpDoubleTestD2.csv',\
        'Exp Double Train D1^mD2^n': 'ExpDoubleTrainD1^mD2^n.csv','Exp Double Test D1^mD2^n': 'ExpDoubleTestD1^mD2^n.csv',\
        'Gaussian Single Train': 'GaussianSingleTrain.csv', 'Gaussian Single Test': 'GaussianSingleTest.csv',\
        'Features List': 'Features List.xlsx','Structure': 'Structure.xlsx'}

FilterDataResults = library.FilterData(F_Records='SET 1.x', F_MoleculesDescriptor = 'MoleculesDescriptor.',\
    TrainIntervals=[(2.6, 20)], F_Train = 'Training Set.x', F_Test = 'Test Set.x',\
    D_Train = 'D Train.csv', D_Test = 'D Test.csv',\
    GridStart = 2.6, GridEnd = 20.0, GridSpacing=0.2, ConfidenceInterval=0.85,\
    TestFraction=0.2, TrainFraction=1, RandomSeed=None)

GenerateFeaturesResults = library.GenerateFeatures(FilterDataResults, Files, F_SystemDescriptor='SystemDescriptor.')

FitResults = library.GetFit(FilterDataResults, Files, GenerateFeaturesResults, F_xlsx='Fit', F_ENet='ENet path',\
        F_GA='GA path', UseVIP=False, nVIP=None, FirstAlgorithm='GA',\
        L1_Single=0.7, eps_single=1e-3, n_alphas_single=100, L1=0.7, eps=1e-3,\
        n_alphas=100, alpha_grid_start=-7, alpha_grid_end=-3, cv=30, MinChromosomeSize=2,\
        ChromosomeSize=7, StopTime=1000, nIter=50, PopulationSize=10,\
        MutationProbability=0.3, MutationInterval=[1, 3],\
        EliteFraction=0.4, MutationCrossoverFraction=0.3, CrossoverFractionInterval=[0.6, 0.4],\
        UseCorrelationMutation=True, MinCorrMutation=0.8, CrossoverMethod='Random',\
        MutationMethod='Correlated', LinearSolver='sklearn', cond=1e-03,\
        lapack_driver='gelsy', UseCorrelationBestFit=False, MinCorrBestFit=0.9,\
        GaussianPrecision=5, GaussianStart=0.01, GaussianEnd=20, GaussianLen=5,\
        PrintInterval=1, RandomSeed=None, verbose=True)

IOfunctions.SaveObject('FilterDataResults.dat', FilterDataResults)
IOfunctions.SaveObject('GenerateFeaturesResults.dat', GenerateFeaturesResults)     
IOfunctions.SaveObject('FitResults.dat', FitResults)   
FilterDataResults = IOfunctions.LoadObject('FilterDataResults.dat')        
GenerateFeaturesResults = IOfunctions.LoadObject('GenerateFeaturesResults.dat')  
FitResults = IOfunctions.LoadObject('FitResults.dat') 

ga = FitResults['GA Object'] # genetic algorithm object
gp = FitResults['GP Object'] # gaussian process regressor object

COM_train = IOfunctions.ReadCSV(FilterDataResults['COM Train'])
COM_test = IOfunctions.ReadCSV(FilterDataResults['COM Test']) 
Y_train = IOfunctions.ReadCSV(Files['Response Train'])
Y_test = IOfunctions.ReadCSV(Files['Response Test'])   
X_LinearSingle_train = IOfunctions.ReadCSV(Files['Linear Single Train'])
X_LinearSingle_test = IOfunctions.ReadCSV(Files['Linear Single Test'])
X_LinearDouble_train = IOfunctions.ReadCSV(Files['Linear Double Train'])
X_LinearDouble_test = IOfunctions.ReadCSV(Files['Linear Double Test'])
X_ExpSingleD_train = IOfunctions.ReadCSV(Files['Exp Single Train D'])
X_ExpSingleDn_train = IOfunctions.ReadCSV(Files['Exp Single Train D^n'])
X_ExpSingleD_test = IOfunctions.ReadCSV(Files['Exp Single Test D'])
X_ExpSingleDn_test = IOfunctions.ReadCSV(Files['Exp Single Test D^n'])    
X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    
y_pred_gp = gp.predict(X_Gaussian_test)
PlotHistogram(Legend='Gaussian process', y_true=Y_test, y_pred=y_pred_gp, PlotType='Absolute',\
    nPredictors=X_Gaussian_test.shape[1], FigNumber=1, FigSize=(19,10), Bins='auto')
#PlotHistogram(Legend='Gaussian process', y_true=Y_test, y_pred=y_pred_gp, PlotType='Relative',\
#    nPredictors=X_Gaussian_test.shape[1], FigNumber=1, FigSize=(19,10), Bins='auto')

for chromosome in ga.DecreasingChromosomes:
    chromosome.print_score()
    y_pred_ga = chromosome.predict(x_expD=X_ExpSingleD_test,\
        x_expDn=X_ExpSingleDn_test, x_lin=X_LinearSingle_test)
    PlotHistogram(Legend='Genetic algorithm', y_true=Y_test, y_pred=y_pred_ga, PlotType='Absolute',\
        nPredictors=chromosome.Size, FigNumber=10, FigSize=(19,10), Bins='auto')
#    PlotHistogram(Legend='Genetic algorithm', y_true=Y_test, y_pred=y_pred_ga, PlotType='Relative',\
#        nPredictors=chromosome.Size, FigNumber=20, FigSize=(19,10), Bins='auto')   
    Plot(CoM=COM_test, E_True=Y_test, nFunctions=2, nPredictors=[chromosome.Size, X_Gaussian_test.shape[1]],\
         E_Predicted=[y_pred_ga, y_pred_gp], Legend=['True Enegry', 'Genetic', 'Gaussian'],\
         Grid=FilterDataResults['Test Grid'], GridTrained=FilterDataResults['Train Grid'],\
         TrainedIntervals=FilterDataResults['Train Intervals'], NontrainedIntervals=FilterDataResults['Test Intervals'])

