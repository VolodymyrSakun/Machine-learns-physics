from project1 import library
import pickle 
from project1 import IOfunctions
import matplotlib.pyplot as plt
import os
import numpy as np
import sklearn.metrics as skm


if __name__ == '__main__':
    initial_directory = os.getcwd()
    F = 'SET 3.x'
    TrainIntervals = [(0, 5), (7, 9)] # (Low, High)   
    GridStart = 0.0
    GridEnd = 14.0
    GridSpacing = 0.2
    ConfidenceInterval = 0.9
    TestFraction = 0.2
    TrainFraction = 0.1
    F_MoleculesDescriptor = 'MoleculesDescriptor.'
    F_Test = 'Test Set.x'
    F_ga_structure = 'GA_structure.dat'
    F_gp_structure = 'GP_structure.dat'
    F_Filter = 'Filter.dat' 
# Get fit for all data    
    while TrainFraction <= 1: 
        results = library.FilterData(F, TrainIntervals, GridStart, GridEnd, GridSpacing,\
            ConfidenceInterval, TestFraction, TrainFraction)
        library.GenerateFeatures()
        library.GetFit()
        TrainFraction += 0.1  
# Prepare for plot
    SetName = F
    SetName = SetName.split(".")
    directory = SetName[0] # directory name
    subdirectory_list = os.listdir(directory) 
    os.chdir(directory) # make current directory 
    parent_directory = os.getcwd() # store it
    f = open(subdirectory_list[0] + '\\' + F_ga_structure, "rb")
    ga = pickle.load(f)
    f.close() 
    nChromosomes = len(ga.DecreasingChromosomes)
    X3Plot = []
    Y3PlotGP_RMSE = []
    Y3PlotGP_R2 = []
    Y3PlotLP_R2 = np.zeros(shape=(nChromosomes, len(subdirectory_list)),\
        dtype=float) # rows - number of LP predictors, columns - %
    Y3PlotLP_RMSE = np.zeros(shape=(nChromosomes, len(subdirectory_list)),\
        dtype=float) # rows - number of LP predictors, columns - %
    chromosome_size_list = []
    o = 0
    for subdirectory in subdirectory_list:     
        if not os.path.exists(subdirectory):
            continue
        x = subdirectory.split('%')
        x = float(x[0])
        X3Plot.append(x) # for 3 graph
        os.chdir(parent_directory) # start from parent dir
        os.chdir(subdirectory) # move to next nested dir
# load intervals 
        f = open(F_Filter, "rb")
        Filter = pickle.load(f)
        f.close()
        SetName = Filter['Initial dataset']
        TrainIntervals = Filter['Train Intervals'] 
        TestIntervals = Filter['Test Intervals'] 
        nTrainPoints = Filter['Train records number'] 
        nTestPoints = Filter['Test records number']
        GridTrain = Filter['Train Grid']
        GridTest = Filter['Test Grid']
        TrainFraction = Filter['Train Fraction Used']
        nTrainPointsGrid = Filter['Train points per grid']
        nTestPointsGrid = ['Test points per grid']  
# load GA structure
        f = open(F_ga_structure, "rb")
        ga = pickle.load(f)
        f.close()            
# load GP structure
        f = open(F_gp_structure, "rb")
        gp = pickle.load(f)
        f.close() 
        results = IOfunctions.ReadFeatures(F_Nonlinear_Train='Distances Train.csv', F_Nonlinear_Test='Distances Test.csv', F_linear_Train='LinearFeaturesTrain.csv',\
            F_Response_Train='ResponseTrain.csv', F_linear_Test='LinearFeaturesTest.csv',\
            F_Response_Test='ResponseTest.csv', F_NonlinearFeatures='NonlinearFeatures.dat',\
            F_FeaturesAll='LinearFeaturesAll.dat', F_FeaturesReduced='LinearFeaturesReduced.dat',\
            F_System='system.dat', F_Records=None, verbose=False)
        FeaturesAll = results['Linear Features All']
        FeaturesReduced = results['Linear Features Reduced']
        Y_train = results['Response Train'] # for Gaussian fit
        X_test = results['X Linear Test']
        Y_test = results['Response Test']
        Y_test = Y_test.reshape(-1)
        X_dist_train = results['X Nonlinear Train']
        X_dist_test = results['X Nonlinear Test']
        MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=F_MoleculesDescriptor)
        RecordsTest = IOfunctions.ReadRecords(F_Test, MoleculePrototypes)
        if RecordsTest[0].nMolecules == 2:
            R = 'R_Average'
        else:
            R = 'R_CenterOfMass_Average'
# gaussian prediction
        y_pred = gp.predict(X_dist_test)
        gpR2 = gp.score(X_dist_test, Y_test)
        mse_gp = skm.mean_squared_error(Y_test, y_pred)
        Y3PlotGP_RMSE.append(np.sqrt(mse_gp)) # for 3 graph
        Y3PlotGP_R2.append(gpR2) # for 3 graph
        print('Gaussian R2 = ', gpR2)
        RMax = GridTest[-1][1]
        RMin = GridTest[0][0] 
        nIntervals = len(GridTest)

# plot graphs for all possible number of predictors
        for m in range(0, len(ga.DecreasingChromosomes), 1):
#        for m in range(0, 1, 1):            
            chromosome = ga.DecreasingChromosomes[m] # current LP predictor
            n = np.zeros(shape=(nIntervals), dtype=float) # n[i] = number of observations in interval    
            mSE = np.zeros(shape=(nIntervals), dtype=float) # MSE[i] = cumulative MSE in interval
            rMSE = np.zeros(shape=(nIntervals), dtype=float) # error in interval = sqrt(MSE[i] / n[i])
            mSE_gp = np.zeros(shape=(nIntervals), dtype=float) # MSE[i] = cumulative MSE in interval
            rMSE_gp = np.zeros(shape=(nIntervals), dtype=float) # error in interval = sqrt(MSE[i] / n[i])
            E_True = np.zeros(shape=(nIntervals), dtype=float) # n[i] = number of observations in interval    
            E_Predicted = np.zeros(shape=(nIntervals), dtype=float) # n[i] = number of observations in interval    
            E_Predicted_gp = np.zeros(shape=(nIntervals), dtype=float) # n[i] = number of observations in interval    
            XPlot = np.zeros(shape=(nIntervals), dtype=float)
            energy = np.zeros(shape=(len(RecordsTest)), dtype=float)
            if o == 0:
                chromosome_size_list.append(chromosome.Size)
            for i in range(0, nIntervals, 1):
                XPlot[i] = (GridTest[i][1] + GridTest[i][0])/2 # average in grid
            for i in range(0, len(RecordsTest), 1): # record index
                if R == 'R_Average': 
                    Criterion = RecordsTest[i].R_Average
                elif R == 'R_CenterOfMass_Average':
                    Criterion = RecordsTest[i].R_CenterOfMass_Average
                else:
                    print('Wrong argument')
                    break
                RecordsTest[i].get_energy(chromosome, FeaturesAll, FeaturesReduced)# predict LP energy
                energy[i] = RecordsTest[i].E_Predicted
                j = library.InInterval(Criterion, GridTest) # grid index
                mSE[j] += RecordsTest[i].MSE
                E_True[j] += Y_test[i]
                E_Predicted[j] += RecordsTest[i].E_Predicted
                mSE_gp[j] += (y_pred[i] - Y_test[i])**2
                E_Predicted_gp[j] += y_pred[i]
                n[j] += 1
                
            Y3PlotLP_RMSE[m, o] = np.sqrt(skm.mean_squared_error(energy, Y_test))
            Y3PlotLP_R2[m, o] = skm.r2_score(Y_test, energy)
            for j in range(0, len(n), 1):
                rMSE[j] = np.sqrt(mSE[j] / n[j])
                rMSE_gp[j] = np.sqrt(mSE_gp[j] / n[j])
                E_True[j] = E_True[j] / n[j]
                E_Predicted[j] = E_Predicted[j] / n[j]
                E_Predicted_gp[j] = E_Predicted_gp[j] / n[j]
        
            X_Plot_Train = [] # X axis
            X_Plot_Test = []
            rMSE_Plot_Train = [] # Y axis
            rMSE_Plot_Test = []
            E_True_Plot_Train = []
            E_True_Plot_Test = []
            E_Predicted_Plot_Train = []
            E_Predicted_Plot_Test = []
            rMSE_gp_Plot_Train = []
            E_Predicted_gp_Plot_Train = []
            rMSE_gp_Plot_Test = []
            E_Predicted_gp_Plot_Test = []
            for i in range(0, len(XPlot), 1):
                if library.InInterval(XPlot[i], TrainIntervals) != -1: # plot as trained points
                    X_Plot_Train.append(XPlot[i])
                    rMSE_Plot_Train.append(rMSE[i])
                    rMSE_gp_Plot_Train.append(rMSE_gp[i])
                    E_True_Plot_Train.append(E_True[i])
                    E_Predicted_Plot_Train.append(E_Predicted[i])
                    E_Predicted_gp_Plot_Train.append(E_Predicted_gp[i])
                else:
                    X_Plot_Test.append(XPlot[i])
                    rMSE_Plot_Test.append(rMSE[i])
                    rMSE_gp_Plot_Test.append(rMSE_gp[i])
                    E_True_Plot_Test.append(E_True[i])
                    E_Predicted_Plot_Test.append(E_Predicted[i])
                    E_Predicted_gp_Plot_Test.append(E_Predicted_gp[i])
            
            X_plot_train = []
            Y_plot_train = []
            for i in range(0, len(TrainIntervals), 1):
                X_plot_train.append([])
                Y_plot_train.append([])
                for j in range(0, len(X_Plot_Train), 1):
                    if X_Plot_Train[j] >= TrainIntervals[i][0] and X_Plot_Train[j] <= TrainIntervals[i][1]:
                        X_plot_train[i].append(X_Plot_Train[j])    
                        Y_plot_train[i].append(E_True_Plot_Train[j])
            X_plot_test = []
            Y_plot_test = []
            for i in range(0, len(TestIntervals), 1):
                X_plot_test.append([])
                Y_plot_test.append([])
                for j in range(0, len(X_Plot_Test), 1):
                    if X_Plot_Test[j] >= TestIntervals[i][0] and X_Plot_Test[j] <= TestIntervals[i][1]:
                        X_plot_test[i].append(X_Plot_Test[j])    
                        Y_plot_test[i].append(E_True_Plot_Test[j])                    
    # plot RMSE          
            XMin = RMin - (RMax-RMin) * 0.02
            XMax = RMax + (RMax-RMin) * 0.02
            if len(rMSE_Plot_Test) != 0:
                rMSE_Min = min(min(rMSE_Plot_Train), min(rMSE_Plot_Test),\
                    min(rMSE_gp_Plot_Train), min(rMSE_gp_Plot_Test))
                rMSE_Max = max(max(rMSE_Plot_Train), max(rMSE_Plot_Test),\
                    max(rMSE_gp_Plot_Train), max(rMSE_gp_Plot_Test))
            else:
                rMSE_Min = min(min(rMSE_Plot_Train), min(rMSE_gp_Plot_Train))
                rMSE_Max = max(max(rMSE_Plot_Train), max(rMSE_gp_Plot_Train))          
            YMin = rMSE_Min - (rMSE_Max-rMSE_Min) * 0.02
            YMax = rMSE_Max + (rMSE_Max-rMSE_Min) * 0.02
            fig1 = plt.figure(1, figsize=(19,10))
            plt.xlim((XMin, XMax))
            plt.ylim((YMin, YMax))
            plt.scatter(X_Plot_Train, rMSE_Plot_Train, c='red', marker='d', label='LM Trained region')
            plt.scatter(X_Plot_Test, rMSE_Plot_Test, c='blue', marker='d', label='LM Not trained region')
            plt.scatter(X_Plot_Train, rMSE_gp_Plot_Train, c='magenta', marker="*", label='GP Trained region')
            plt.scatter(X_Plot_Test, rMSE_gp_Plot_Test, c='green', marker="*", label='GP Not trained region')
            plt.legend()
            plt.ylabel('RMSE')
            plt.xlabel('Average distance between centers of masses of molecules')
            plt.title('RMSE vs. average distance')
            plt.show(fig1)
            F = "RMSE. Number of predictors = " + str(chromosome.Size) + ".png"
            plt.savefig(F, bbox_inches='tight')
            plt.close(fig1)
    # plot Energy
            if len(E_True_Plot_Test) != 0:
                EMin = min(min(E_True_Plot_Train), min(E_True_Plot_Test),\
                    min(E_Predicted_Plot_Train), min(E_Predicted_Plot_Test),\
                    min(E_Predicted_gp_Plot_Train), min(E_Predicted_gp_Plot_Test))
                EMax = max(max(E_True_Plot_Train), max(E_True_Plot_Test),\
                    max(E_Predicted_Plot_Train), max(E_Predicted_Plot_Test),\
                    max(E_Predicted_gp_Plot_Train), max(E_Predicted_gp_Plot_Test))  
            else:
                EMin = min(min(E_True_Plot_Train), min(E_Predicted_Plot_Train),\
                    min(E_Predicted_gp_Plot_Train))
                EMax = max(max(E_True_Plot_Train), max(E_Predicted_Plot_Train),\
                    max(E_Predicted_gp_Plot_Train))
            YMin = EMin - (EMax-EMin) * 0.02
            YMax = EMax + (EMax-EMin) * 0.02
            fig2 = plt.figure(2, figsize=(19,10))
            plt.xlim((XMin, XMax))
            plt.ylim((YMin, YMax))
            for i in range(0, len(X_plot_train), 1):
                plt.plot(X_plot_train[i], Y_plot_train[i], c='red', marker='.', label='True Energy. Trained region')
            for i in range(0, len(X_plot_test), 1):
                plt.plot(X_plot_test[i], Y_plot_test[i], c='blue', marker='.', label='True Energy. Not trained region')        
            plt.scatter(X_Plot_Train, E_Predicted_Plot_Train, c='green', marker='d', label='LM Predicted Energy. Trained region')
            plt.scatter(X_Plot_Test, E_Predicted_Plot_Test, c='violet', marker='d', label='LM Predicted Energy. Not trained region')
            plt.scatter(X_Plot_Train, E_Predicted_gp_Plot_Train, c='magenta', marker="*", label='GP Predicted Energy. Trained region')
            plt.scatter(X_Plot_Test, E_Predicted_gp_Plot_Test, c='black', marker="*", label='GP Predicted Energy. Not trained region')
            plt.legend()
            plt.ylabel('Energy')
            plt.xlabel('Average distance between centers of masses of molecules')
            plt.title('Energy vs. average distance')
            plt.show(fig2)
            F = "Energy. Number of predictors = " + str(chromosome.Size) + ".png"
            plt.savefig(F, bbox_inches='tight')
            plt.close(fig2)
        o += 1
        os.chdir(parent_directory) # set parent dir

    os.chdir(parent_directory) # back to parent dir
    fig3 = plt.figure(3, figsize=(19,10))
#    plt.xlim((XMin, XMax))
#    plt.ylim((YMin, YMax))
    plt.plot(X3Plot, Y3PlotGP_RMSE, c='red', marker='*', label='Gaussian')
    for i in range(0, Y3PlotLP_RMSE.shape[0], 1):
        Label = 'LM with ' + str(chromosome_size_list[i]) + ' predictors'
        plt.plot(X3Plot, Y3PlotLP_RMSE[i, :], marker='.', label=Label)
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('% of usage of trained points')
    plt.title('RMSE vs. % of used training points')
    plt.show(fig3)
    F = "Graph 3" + ".png"
    plt.savefig(F, bbox_inches='tight')
    plt.close(fig3)    
            
    fig4 = plt.figure(4, figsize=(19,10))
#    plt.xlim((XMin, XMax))
#    plt.ylim((YMin, YMax))
    plt.plot(X3Plot, Y3PlotGP_R2, c='red', marker='*', label='Gaussian')
    for i in range(0, Y3PlotLP_R2.shape[0], 1):
        Label = 'LM with ' + str(chromosome_size_list[i]) + ' predictors'
        plt.plot(X3Plot, Y3PlotLP_R2[i, :], marker='.', label=Label)
    plt.legend()
    plt.ylabel('R2')
    plt.xlabel('% of usage of trained points')
    plt.title('R2 vs. % of used training points')
    plt.show(fig4)
    F = "Graph 4" + ".png"
    plt.savefig(F, bbox_inches='tight')
    plt.close(fig4) 
    os.chdir(initial_directory)
    
    # erase *.csv, *.dat, *.x at the end
    