# Elastic Net + Backward Sequential + Best Fit
# Proceeds single distances, selects VIP features, than proceeds double features
from structure import library3
import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import shutil

if __name__ == '__main__':
    # Global variables
    L1_Single = 0.7
    eps_single = 1e-3
    n_alphas_single = 100
    LastIterationsToStoreSingle = 15
    MinCorr_Single = 1e-7
    F_Out_Single = 'Single ENet BS BF.xlsx'
    F_ENet_Single = 'Single ENet path.png'
    F_Plot_Single = 'Single Plot'
    Slope = 0.001
    VIP_number = 5

    L1_Double = 0.7
    eps_double = 1e-3
    n_alphas_double = 100
    LastIterationsToStoreDouble = 15
    MinCorr_Double = 1e-7
    F_Out_Double = 'Double ENet BS BF.xlsx'
    F_ENet_Double = 'Double ENet path.png'
    F_Plot_Double = 'Double Plot'
    
    # Read features and structures from files stored by "Generate combined features"
    X, Y, FeaturesAll, FeaturesReduced, system = library3.ReadFeatures('Harmonic Features.csv', \
        'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', 'system.dat', verbose=False)
    if system.nAtoms == 6:
        VIP_number = 5
    if system.nAtoms == 9:
        VIP_number = 3
    # split data in order to separate training set and test set
    # all response variables Y must be 1D arrays
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)
    Y_train = Y_train.reshape(-1, 1)
    x_std = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
    y_std = scale(Y_train, axis=0, with_mean=True, with_std=False, copy=True)
    Y_train = Y_train.reshape(-1)
    y_std = y_std.reshape(-1)
# single distances
    SingleFeaturesAll = []
    for i in range(0, len(FeaturesAll), 1):
        if FeaturesAll[i].nDistances == 1:
            SingleFeaturesAll.append(FeaturesAll[i])
    count_singles = 0
    SingleFeaturesReduced = []
    for i in range(0, len(FeaturesReduced), 1):
        if FeaturesReduced[i].nDistances == 1:
            count_singles += 1
            SingleFeaturesReduced.append(FeaturesReduced[i])
    Single_X_train = np.zeros(shape=(X_train.shape[0], len(SingleFeaturesReduced)), dtype=float)
    Single_X_test = np.zeros(shape=(X_test.shape[0], len(SingleFeaturesReduced)), dtype=float)
    single_x_std = np.zeros(shape=(x_std.shape[0], len(SingleFeaturesReduced)), dtype=float)
    j = 0
    for i in range(0, len(FeaturesReduced), 1):
        if FeaturesReduced[i].nDistances == 1:
            Single_X_train[:, j] = X_train[:, i]
            Single_X_test[:, j] = X_test[:, i]
            single_x_std[:, j] = x_std[:, i]
            j += 1
    print('Elastic Net Fit for single features only')
    print('L1 portion = ', L1_Single)
    print('Epsilon = ', eps_single)
    print('Number of alphas =', n_alphas_single)
    print('Number of features go to elastic net regularisation = ', len(SingleFeaturesReduced))
    alphas = np.logspace(-3, -6, num=100, endpoint=True, base=10.0)
    t = time.time()
    idx, alpha, mse, nonz = library3.SelectBestSubsetFromElasticNetPath(single_x_std, y_std,\
        Method='grid', MSE_threshold=None, R2_threshold=None, L1_ratio=L1_Single, Eps=eps_single,\
        N_alphas=n_alphas_single, Alphas=None, max_iter=10000, tol=0.0001, cv=None, n_jobs=1, \
        selection='random', PlotPath=True, FileName=F_ENet_Single, verbose=False)
    t_sec = time.time() - t
    print("\n", 'Elastic Net worked ', t_sec, 'sec')  
    print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    print('MinCorrelation = ', MinCorr_Single)
    print('Calculating correlation matrix')
    C = np.cov(single_x_std, rowvar=False, bias=True)
    writeResults = pd.ExcelWriter(F_Out_Single, engine='openpyxl')
    mse_list = []
    rmse_list = []
    r2_list = []
    aIC_list = []
    nonzero_list = []
    features_list = []
    t = time.time()        
    print('Start Backward sequential elimination')
    while len(idx) > 2:
        FeatureSize = len(idx)
        if FeatureSize > LastIterationsToStoreSingle:
            print('Features left: ', FeatureSize)
        idx_backward = library3.BackwardElimination(Single_X_train, Y_train, Single_X_test, Y_test,\
            SingleFeaturesAll, SingleFeaturesReduced, Method='sequential', Criterion='MSE', \
            N_of_features_to_select=FeatureSize-1, idx=idx, PlotPath=False, StorePath=False, \
            FileName='Backward sequential', N_of_last_iterations_to_store=None, verbose=False)
        if len(idx_backward) > LastIterationsToStoreSingle:
            idx = idx_backward
            continue
        idx_corr = library3.get_full_list(idx, single_x_std.shape[1])
        print('Features left: ', len(idx_backward))
        idx = library3.rank_features(single_x_std, y_std, idx_backward)
        idx_alternative = library3.FindBestSet(single_x_std, y_std, idx, idx_corr, Method='MSE', verbose=True)
        library3.Results_to_xls3(writeResults, str(len(idx_alternative)),\
            idx_alternative, Single_X_train, Y_train, Single_X_test, Y_test, SingleFeaturesAll, SingleFeaturesReduced)
        idx = idx_alternative
        nonzero, mse, rmse, r2, aIC = library3.get_fit_score(Single_X_train, Single_X_test, Y_train, Y_test, idx=idx)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        aIC_list.append(aIC)
        nonzero_list.append(nonzero)
        features_list.append(idx)
    t_sec = time.time() - t
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    writeResults.save()
    library3.plot_rmse(F_Plot_Single, nonzero_list, rmse_list, r2_list)
    size_list = len(nonzero_list)
    for i in range(0, size_list-1, 1):
        mse = mse_list[size_list-i-1]
        mse_next =  mse_list[size_list-i-2]
        delta = mse - mse_next
        fraction = delta / mse_next
        if (fraction < 0) or (abs(fraction) < Slope) or (nonzero_list[size_list-i-2] > VIP_number):
            VIP_idx = features_list[size_list-i-1]
            if fraction < 0:
                print('VIP selection criteria: Slope became positive')
            if abs(fraction) < Slope:
                print('VIP selection criteria: Slope less than ', Slope)
            if nonzero_list[size_list-i-2] > VIP_number:
                print('VIP selection criteria: desired number of VIP features = ', VIP_number, 'has been reached')
            print('List of VIP features: ', VIP_idx)
            break
# double distances
    print('Elastic Net Fit for all features')
    print('L1 portion = ', L1_Double)
    print('Epsilon = ', eps_double)
    print('Number of alphas =', n_alphas_double)
    print('Number of features go to elastic net regularisation = ', len(FeaturesReduced))
    print('VIP features to be conserved: ', VIP_idx)
    alphas = np.logspace(-3, -6, num=100, endpoint=True, base=10.0)
    t = time.time()
    idx, alpha, mse, nonz = library3.SelectBestSubsetFromElasticNetPath(x_std, y_std,\
        Method='grid', MSE_threshold=None, R2_threshold=None, L1_ratio=L1_Double, Eps=eps_double,\
        N_alphas=n_alphas_double, Alphas=None, max_iter=10000, tol=0.0001, cv=None, n_jobs=1, \
        selection='random', PlotPath=True, FileName=F_ENet_Double, verbose=False)
    t_sec = time.time() - t
    for i in VIP_idx: # add missing VIP features 
        if i not in idx:
            idx.append(i)
    print("\n", 'Elastic Net worked ', t_sec, 'sec')  
    print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    print('MinCorrelation = ', MinCorr_Double)
    print('Calculating correlation matrix')
    C = np.cov(x_std, rowvar=False, bias=True)
    writeResults = pd.ExcelWriter(F_Out_Double, engine='openpyxl')
    mse_list = []
    rmse_list = []
    r2_list = []
    aIC_list = []
    nonzero_list = []
    features_list = []
    t = time.time()        
    print('Start Backward sequential elimination')
    while len(idx) > (1+len(VIP_idx)):
        FeatureSize = len(idx)
        if FeatureSize > LastIterationsToStoreSingle:
            print('Features left: ', FeatureSize)
        idx_backward = library3.BackwardElimination(X_train, Y_train, X_test, Y_test,\
            FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
            N_of_features_to_select=FeatureSize-1, idx=idx, VIP_idx=VIP_idx, PlotPath=False, StorePath=False, \
            FileName=None, N_of_last_iterations_to_store=None, verbose=False)
        if len(idx_backward) > LastIterationsToStoreSingle:
            idx = idx_backward
            continue
        idx_corr = library3.get_full_list(idx, x_std.shape[1])
        print('Features left: ', len(idx_backward))
        idx = library3.rank_features(x_std, y_std, idx_backward)
        idx_alternative = library3.FindBestSet(x_std, y_std, idx, idx_corr, VIP_idx=VIP_idx, Method='MSE', verbose=True)
        library3.Results_to_xls3(writeResults, str(len(idx_alternative)),\
            idx_alternative, X_train, Y_train, X_test, Y_test, FeaturesAll, FeaturesReduced)
        idx = idx_alternative
        nonzero, mse, rmse, r2, aIC = library3.get_fit_score(X_train, X_test, Y_train, Y_test, idx=idx)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        aIC_list.append(aIC)
        nonzero_list.append(nonzero)
        features_list.append(idx)
    t_sec = time.time() - t
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    writeResults.save()
    library3.plot_rmse(F_Plot_Double, nonzero_list, rmse_list, r2_list)
    
    directory = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    if not os.path.exists(directory):
        os.makedirs(directory)    
    try:
        shutil.copyfile('SystemDescriptor.', directory + '\\' + 'SystemDescriptor.')
        shutil.copyfile('Structure.xlsx', directory + '\\' + 'Structure.xlsx')
        shutil.copyfile('Harmonic Features Reduced List.xlsx', directory + '\\' + 'Harmonic Features Reduced List.xlsx')
        shutil.move(F_Out_Single, directory + '\\' + F_Out_Single)
        shutil.move(F_ENet_Single, directory + '\\' + F_ENet_Single)
        shutil.move(F_Plot_Single + '_RMSE' + '.png', directory + '\\' + F_Plot_Single + '_RMSE' + '.png')
        shutil.move(F_Plot_Single + '_R2' + '.png', directory + '\\' + F_Plot_Single + '_R2' + '.png')
        shutil.move(F_Out_Double, directory + '\\' + F_Out_Double)
        shutil.move(F_ENet_Double, directory + '\\' + F_ENet_Double)
        shutil.move(F_Plot_Double + '_RMSE' + '.png', directory + '\\' + F_Plot_Double + '_RMSE' + '.png')
        shutil.move(F_Plot_Double + '_R2' + '.png', directory + '\\' + F_Plot_Double + '_R2' + '.png')
    except:
        pass
    print('DONE')

        
        
        
        
        
        
        
        
        
        
        
        