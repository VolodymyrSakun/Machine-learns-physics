from structure import library2
import pandas as pd
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

if __name__ == '__main__':
    
    # Global variables
    ProcedureFileName = 'procedure.txt'
    L1 = 0.7
    eps = 1e-3
    n_alphas = 100
    #FinalAdjustment = 'Backward sequential'
    FinalAdjustment = 'Backward sequential and Search Alternative'
    LastIterationsToStore = 20
    MinCorr = 1e-7
    
    # Read features and structures from files stored by "Generate combined features"
    X, Y, FeaturesAll, FeaturesReduced = library2.ReadFeatures('Harmonic Features.csv', \
        'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', verbose=False)
    # split data in order to separate training set and test set
    # all response variables Y must be 1D arrays
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    Y_train = Y_train.reshape(-1, 1)
    x_std = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
    y_std = scale(Y_train, axis=0, with_mean=True, with_std=False, copy=True)
    #x_std = X_train
    #y_std = Y_train
    Y_train = Y_train.reshape(-1)
    y_std = y_std.reshape(-1)
    # redirect output to file
    # f_new, f_old = RedirectPrintToFile(ProcedureFileName)
    print('Least squares for full set of features')
    print('Number of features = ', X_train.shape[1])
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
    lr.fit(X_train, Y_train)
    coef = lr.coef_
    y_pred = lr.predict(X_train)
    mse = skm.mean_squared_error(Y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y_train, y_pred)
    print('MSE= ', mse)
    print('RMSE= ', rmse)
    print('R2= ', r2)
    
    idx_initial = []
    for i in range(0, X_train.shape[1], 1):
        idx_initial.append(i)
        
    print('Elastic Net Fit')
    print('L1 portion = ', L1)
    print('Epsilon = ', eps)
    print('Number of alphas =', n_alphas)
    print('Number of features go to elastic net regularisation = ', len(FeaturesReduced))
    print('Final adjustment selected = ', FinalAdjustment)
    alphas = np.logspace(-3, -6, num=100, endpoint=True, base=10.0)
    
    
    t = time.time()
    # get elastic net path
    # select the best subset from elastic net path
    # THIS SET WORKS
    # VIP features - single distances to power -1
    idx, alpha, mse, nonz = library2.SelectBestSubsetFromElasticNetPath(x_std, y_std,\
        Method='grid', MSE_threshold=None, R2_threshold=None, L1_ratio=L1, Eps=eps,\
        N_alphas=n_alphas, Alphas=None, max_iter=10000, tol=0.0001, cv=None, n_jobs=1, \
        selection='random', PlotPath=True, verbose=False)
    t_sec = time.time() - t
    print("\n", 'Elastic Net worked ', t_sec, 'sec')
    
    
    """
    l1 = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5 ,0.6 ,0.7 ,0.8, 0.9]
    l1 = 0.05
    # ElasticNetCV
    # select the best subset from elastic net path
    idx, alpha = library1.SelectBestSubsetFromElasticNetPath(x_std, y_std, Method='CV',\
        MSE_threshold=None, R2_threshold=None, L1_ratio=l1, Eps=1e-5, N_alphas=200,\
        Alphas=None, max_iter=10000, tol=0.0001, cv=30, n_jobs=-1,\
        selection='random', PlotPath=True, verbose=True)
    """
    
    
    if FinalAdjustment == 'Backward sequential':
        print("\n", 'Features left for Backward Elimination = ', len(idx))
        t = time.time()
    # eliminate least important features    
        features_index = library2.BackwardElimination(X_train, Y_train, X_test, Y_test,\
            FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
            N_of_features_to_select=20, idx=idx, PlotPath=True, StorePath=True, \
            FileName='Backward sequential', N_of_last_iterations_to_store=LastIterationsToStore,\
            verbose=True)
        t_sec = time.time() - t
        print('Backward Elimination worked ', t_sec, 'sec')
        
    if FinalAdjustment == 'Backward sequential and Search Alternative':
        print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
        print('MinCorrelation = ', MinCorr)
        print('Correlation matrix')
        C = np.cov(x_std, rowvar=False, bias=True)
        writeResults = pd.ExcelWriter('Backward Sequential Best Fit.xlsx', engine='openpyxl')
        mse_list = []
        rmse_list = []
        r2_list = []
        aIC_list = []
        nonzero_list = []
        t = time.time()        
        while len(idx) > 2:
            FeatureSize = len(idx)
            idx_backward = library2.BackwardElimination(X_train, Y_train, X_test, Y_test,\
                FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
                N_of_features_to_select=FeatureSize-1, idx=idx, PlotPath=False, StorePath=False, \
                FileName='Backward sequential', N_of_last_iterations_to_store=None, verbose=False)
            if len(idx_backward) > LastIterationsToStore:
                idx = idx_backward
                continue
            idx_corr = library2.ClassifyCorrelatedFeatures(x_std, idx_backward, MinCorrelation=MinCorr,\
                Model=1, Corr_Matrix=C, verbose=False)
            print(len(idx_backward))
            idx_alternative = library2.FindBestSetMP(x_std, y_std, idx_backward, idx_corr, Method='MSE', verbose=True)
            library2.Results_to_xls2(writeResults, str(len(idx_alternative)),\
                idx_alternative, X_train, Y_train, X_test, Y_test, FeaturesAll, FeaturesReduced)
            idx = idx_alternative
            nonzero, mse, rmse, r2, aIC = library2.get_fit_score(X_train, X_test, Y_train, Y_test, idx=idx)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            aIC_list.append(aIC)
            nonzero_list.append(nonzero)
        t_sec = time.time() - t
        print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
        writeResults.save()
        library2.plot_rmse("Results", nonzero_list, rmse_list, r2_list)
    # return output back to console
    # RedirectPrintToConsole(f_new, f_old)
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        