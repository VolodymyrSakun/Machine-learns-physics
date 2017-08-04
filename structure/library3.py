# Library of classes and functions
# Classes:
# class ENet
# ENet.fit
# class BF
# BF.fit
# Functions:
# RedirectPrintToFile
# RedirectPrintToConsole
# argmaxabs
# argminabs
# Scaler_L2
# Results_to_xls
# BackwardElimination
# RemoveWorstFeature
# SelectBestSubsetFromElasticNetPath
# CalculateVif
# ReadFeatures
# DecisionTreeEstimator
# RandomForestEstimator
# ClassifyCorrelatedFeatures
# get_full_list
# StoreFeaturesDescriprion
# get_fit_score, 2, 3, 4
# plot_rmse
# Fit
# FindBestSet
# ForwardSequential
# store_structure
# rank_features

import numpy as np
import pandas as pd
import sklearn.metrics as skm
import copy
import pickle 
import sys
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import enet_path
import multiprocessing as mp
from joblib import Parallel, delayed

# redirect "print" functin to file
def RedirectPrintToFile(ProcedureFileName):
    f_old = sys.stdout
    f = open(ProcedureFileName, 'w')
    sys.stdout = f
    return f, f_old

# redirect "print" functin to console
def RedirectPrintToConsole(f, f_old):
    sys.stdout = f_old
    f.close()
    return

def argmaxabs(A):
    tmp = abs(A[0])
    idx = 0
    for i in range(1, len(A), 1):
        if abs(A[i]) > tmp:
            tmp = abs(A[i])
            idx = i
    return idx

def argminabs(A):
    tmp = abs(A[0])
    idx = 0
    for i in range(1, len(A), 1):
        if abs(A[i]) < tmp:
            tmp = abs(A[i])
            idx = i
    return idx

def Scaler_L2(X):
    X_new = np.zeros(shape=(X.shape[0], X.shape[1]), dtype = float)
    col = np.zeros(shape=(X.shape[0]), dtype = float)
    m = X.shape[0]# n of rows
    n = X.shape[1]# n of columns
    for j in range(0, n, 1):
        col[:] = X[:, j]
        Denom = np.sqrt(np.dot(np.transpose(col), col, out=None))
        for i in range(0, m, 1):
            X_new[i, j] = X[i, j] / Denom
    return X_new

# reads feature set and objects that describes it
def ReadFeatures(F_features, F_structure_FeaturesAll, F_structure_FeaturesReduced, F_System, verbose=False):
# F_features - .csv file that contains data stored by Generape combined features.py
# F_structure_FeaturesAll - .dat file that contains object with all informations about features
# F_structure_FeaturesReduced - .dat file that contains object with all informations about reduced features
# returns:
# X - [n x m] numpy array of features
# rows correspond to observations
# columns correspond to features
# Y - [n x 1] numpy array, recponse variable
# FeaturesAll - class1.InvPowDistancesFeature object. Contains all features
# FeaturesReduced - class1.InvPowDistancesFeature object. Contains combined features
    if verbose:
        print('Reading data from file')
    dataset = pd.read_csv(F_features)
    NofFeatures = dataset.shape[1] - 1 # read numder of distances from dataset
    X = dataset.iloc[:, 0:NofFeatures].values
    Y = dataset.iloc[:, -1].values
    if len(Y.shape) == 2:
        Y = Y.reshape(-1)
    del dataset['energy']
# load reduced features and energy from file
    f = open(F_structure_FeaturesReduced, "rb")
    FeaturesReduced = pickle.load(f)
    f.close()
# load list FeaturesAll from file
    f = open(F_structure_FeaturesAll, "rb")
    FeaturesAll = pickle.load(f)
    f.close()
# load system object from file
    f = open(F_System, "rb")
    system = pickle.load(f)
    f.close()    
    return X, Y, FeaturesAll, FeaturesReduced, system

# select the best subset from elastic net path
def SelectBestSubsetFromElasticNetPath(x_std, y_std, Method='CV', MSE_threshold=None, R2_threshold=None, \
    L1_ratio=0.01, Eps=1e-3, N_alphas=200, Alphas=None, max_iter=10000, tol=0.0001, \
    cv=30, n_jobs=-1, selection='random', PlotPath=False, FileName=None, verbose=False):
# returns list of indices of selected features
# default setting: use ElasticNetCV assigned by Method
# x_std - standardized set of features
# y_std - response with mean = 0
# Method='CV' - use ElasticNEtCV, Method='grid' - use enet_path
# MSE_threshold - treshhold for selecting min nonzero coefficients which gives fit better than MSE_threshold
# R2_threshold - treshhold for selecting min nonzero coefficients which gives fit better than R2_threshold
# L1_ratio - portion of L1 regularization. For Method='CV' can be array of floats
# eps - Length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3. Relevant if alphas is not assigned
# N_alphas - size of grid
# Alphas - array of regularization strengths
# cv - number of cross-folds. only for Method='CV'
# selection='cyclic' or 'random'. If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default. This (setting to ‘random’) often leads to significantly faster convergence especially when tol is higher than 1e-4.
# PlotPath - plot 2 graphs: Number of nonzero coefficients vs. alpha and MSE vs. alpha. Only for Method='grid'

    if (Method == 'CV'):
        enet_cv = ElasticNetCV(l1_ratio=L1_ratio, eps=Eps, n_alphas=N_alphas, alphas=Alphas, fit_intercept=False,\
            normalize=False, precompute='auto', max_iter=max_iter, tol=0.0001, cv=cv, copy_X=True, \
            verbose=verbose, n_jobs=n_jobs, positive=False, random_state=101, selection=selection)
        enet_cv.fit(x_std, y_std)
        coef = enet_cv.coef_
        alpha = enet_cv.alpha_
        alphas = enet_cv.alphas_
        nonzero_idx = []
        for j in range(0, len(coef), 1):
            if coef[j] != 0:
                nonzero_idx.append(j)

    if (Method == 'grid'):    
        alphas , coefs, _ = enet_path(x_std, y_std, l1_ratio=L1_ratio, eps=Eps, \
            n_alphas=N_alphas, alphas=Alphas, precompute='auto', Xy=None, copy_X=True,\
            coef_init=None, verbose=verbose, return_n_iter=False, positive=False, check_input=True)
       
        lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
        mse_list = []
        nonzero_count_list = []
        for i in range(0, coefs.shape[1], 1): # columns
            nonzero_idx = []
            for j in range(0, coefs.shape[0], 1):
                if coefs[j, i] != 0:
                    nonzero_idx.append(j)
            if len(nonzero_idx) == 0:
                mse_list.append(1e+100)
                continue
            nonzero_count_list.append(len(nonzero_idx))
            x = np.zeros(shape=(x_std.shape[0], len(nonzero_idx)), dtype=float)
            x[:, :] = x_std[:, nonzero_idx]
            lr.fit(x, y_std)
            y_pred_lr = lr.predict(x)
            mse_lr = skm.mean_squared_error(y_std, y_pred_lr)
            mse_list.append(mse_lr)
        count_min = coefs.shape[0]
        idx = -1
        # if threshold for MSE is provided
        if MSE_threshold is not None:
            for i in range(0, len(mse_list), 1):
                if mse_list[i] < MSE_threshold:
                    if nonzero_count_list[i] < count_min:
                        idx = i
                        count_min = nonzero_count_list[i]
        else:
            idx = np.argmin(mse_list) # index of column of enet_path. best fit
        for i in range(0, len(mse_list), 1):
            if mse_list[i] == 1e+100:
                mse_list[i] = 0
        if idx == -1:
            print('MSE for all path is greater than threshold provided :', MSE_threshold)
            print('Best fit is assigned according to min MSE from the path')
            idx = np.argmin(mse_list) # index of column of enet_path. best fit
        alpha = alphas[idx]
# store indices of nonzero coefficients of best fit        
        nonzero_idx = []
        for j in range(0, coefs.shape[0], 1):
            if coefs[j, idx] != 0:
                nonzero_idx.append(j)
                
    if PlotPath and (Method == 'grid'):
        nonzero = []
        for i in range(0, coefs.shape[1], 1):
            nonzero_count = np.count_nonzero(coefs[:, i])
            nonzero.append(nonzero_count)
        fig = plt.figure(1, figsize = (19, 10))
        plt.subplot(211)
        subplot11 = plt.gca()
        plt.plot(alphas, nonzero, ':')
        subplot11.set_xscale('log')
        plt.xlabel('Alphas')
        plt.ylabel('Number of nonzero coefficients')
        plt.title('Elastic Net Path')
        plt.subplot(212)
        subplot21 = plt.gca()
        plt.plot(alphas, mse_list, ':')
        subplot21.set_xscale('log')
        subplot21.set_yscale('log')
        plt.xlabel('Alphas')
        plt.ylabel('MSE')
        plt.title('Mean squared error vs. regularization strength')
        plt.show()
        plt.savefig(FileName, bbox_inches='tight')
        plt.close(fig)
    if (Method == 'CV'):
        return nonzero_idx, alpha
    if (Method == 'grid'):
        return nonzero_idx, alpha, mse_list, nonzero_count_list

def BackwardElimination(X_train, Y_train, X_test, Y_test, FeaturesAll, \
    FeaturesReduced, Method='fast', Criterion='p-Value', N_of_features_to_select=20, \
    idx=None, VIP_idx=None, PlotPath=False, StorePath=False, \
    FileName=None, N_of_last_iterations_to_store=10, verbose=False):
# returns list of indices of selected features
# idx should contain approximatelly 200 indices
# X_train - numpy array, training set of nonstandartized features
# rows - observations
# columns - variables
# Y_train - numpy array training set of nonstandardized response
# X_test - numpy array, test set of nonstandartized features
# Y_train - numpy array test set of nonstandardized response
# FeaturesAll - class1.InvPowDistancesFeature object. Contains all features
# FeaturesReduced - class1.InvPowDistancesFeature object. Contains combined features
# Method = 'fast' selects feature to drop based on greatest p-Value or smallest coefficient
# Criterion = 'coef' based on dropping the feature wich has smallest standartised coeffitient
# coef method is not relevant for this type of problem
# Criterion = 'p-Value' based on dropping the feature wich has greatest p-Value
# fast but can remove necessary features
# Method = 'sequential' based on sequential analysis of features
# Criterion = 'MSE' drops feature when smallest change in MSE occurs
# Criterion = 'R2' drops feature when smallest change in R2 occurs
# idx list of features where algorithm starts to eliminate them
# N_of_features_to_select - desirable number of features to have at the end of elimination
# if Criterion = 'p-Value' and N_of_features_to_select is not provided algorithm will stop itself either at 2 feature left or when 
# all features are significant and those p-values = 0
# PlotPath if true plot path
# StorePath if true store path in excel file
# N_of_last_iterations_to_store - number of last iteration will be stored in .xlsx
# FileName - file name to store plots and data
# R2.png will be added to filename for R2 plot
# RMSE.png will be added to filename for RMSE plot
# .xlsx will be added to filename for excel file
    if VIP_idx is None:
        VIP_idx = []
    else:
        if N_of_features_to_select < len(VIP_idx):
            print('Desired number of features cannot be equal or less than number of VIP features')
            return -1
    if PlotPath and (FileName is None):
        print('Please indicate name of file to store data and / or graphs')
        return -2
    if (FileName is not None) and StorePath:    
        writeResults = pd.ExcelWriter(FileName + '.xlsx')
    else:
        writeResults = None
# features_index - list of indices of selected features
    if idx is None:
        features_index = list(range(0, X_train.shape[1], 1))
        del(idx)
    else:
        features_index = list(idx)
        del(idx)
    Size_train = Y_train.shape[0]
    Size_test = Y_test.shape[0]
    x_sel = np.zeros(shape=(Size_train, len(features_index))) # create working array of features 
    j = 0
# copy features defined by features_index from training set to x_sel
    for i in features_index:
        x_sel[:, j] = X_train[:, i]
        j += 1
# standardize selected features from training set
    x_train_std = StandardScaler(with_mean=True, with_std=True).fit_transform(x_sel)
    Y_train = Y_train.reshape(-1, 1)
    y_train_std = StandardScaler(with_mean=True, with_std=False).fit_transform(Y_train)
    Y_train = Y_train.reshape(-1)
    y_train_std = y_train_std.reshape(-1)
    ols_coef = np.zeros(shape=(2), dtype=float)
    pvalues = np.ones(shape=(2), dtype=float)
    nonzero_count_list = []
    rsquared_list = []
    rmse_OLS_list = []
    ols_coef_list = []
    aIC_list = []
    drop_order = [] # contain indices of features according to initial feature list idx
    rsquared = 1
    nonzero_count = N_of_features_to_select
    nonzero_count += 1
    if verbose:
        print('Start removing features')
    if Method == 'fast':
        while (max(pvalues) != 0) and (len(ols_coef) > 1) and (nonzero_count > N_of_features_to_select):
            ols = sm.OLS(endog = y_train_std, exog = x_train_std, hasconst = False).fit()
            y_pred = ols.predict(x_train_std)
            ols_coef = ols.params
            nonzero_count = np.count_nonzero(ols_coef)
            if verbose:
                print('Features left in model: ', nonzero_count)
            pvalues = ols.pvalues
            rsquared = ols.rsquared
            mse_OLS = skm.mean_squared_error(y_train_std, y_pred)
            rmse_OLS = np.sqrt(mse_OLS)
            rmse_OLS_list.append(rmse_OLS)
            rsquared_list.append(rsquared*100)
            ols_coef_list.append(ols_coef)
            nonzero_count_list.append(nonzero_count)
            for i in VIP_idx:
                pvalues[i] = 0 # keep all VIP features
                ols_coef[i] = 1e+100
            if Criterion == 'p-Value':
                drop = np.argmax(pvalues) # removes index of feature with max p-Value
            if Criterion == 'coef':
                drop = argminabs(ols_coef) # removes index of feature with smallest coefficient
            drop_order.append(features_index[drop])
            x_train_std = np.delete(x_train_std, drop, 1) # last 1 = column, 0 = row
            del(features_index[drop])
        # end while
    # end if
    if Method == 'sequential':
        lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
        while (nonzero_count > N_of_features_to_select):
            lr.fit(x_train_std, y_train_std)
            lr_coef = lr.coef_
            nonzero_count = np.count_nonzero(lr_coef)
            if verbose:
                print('Features left in model: ', nonzero_count)
            y_pred = lr.predict(x_train_std)
            mse = skm.mean_squared_error(y_train_std, y_pred)
            rmse = np.sqrt(mse)
            r2 = skm.r2_score(y_train_std, y_pred)
            rSS_ols = 0
            for m in range(0, Size_train, 1):
                rSS_ols += (y_pred[m] - y_train_std[m])**2
            aIC_ols = 2 * nonzero_count + Size_train * np.log(rSS_ols)
            z = np.zeros(shape=(Size_train, 1), dtype=float) # always zeros
            tmp = np.zeros(shape=(Size_train, 1), dtype=float) # temporary storage of one feature
            mse_array = np.zeros(shape=(len(features_index)), dtype=float)
            r2_array = np.zeros(shape=(len(features_index)), dtype=float)
            aIC_array = np.zeros(shape=(len(features_index)), dtype=float)
            rmse_OLS_list.append(rmse)
            rsquared_list.append(r2*100)
            aIC_list.append(aIC_ols)
            ols_coef_list.append(lr_coef)
            nonzero_count_list.append(nonzero_count)
# change this loop for MP
            for i in range(0, len(features_index), 1): # local index
                k = features_index[i] # global index
                if k in VIP_idx:
                    mse_array[i] = 1e+100 # keep all VIP features
                    r2_array[i] = 0
                    aIC_array[i] = 1e+100  
                else:
                    tmp[:, 0] = x_train_std[:, i]
                    x_train_std[:, i] = z[:, 0]
                    lr.fit(x_train_std, y_train_std)
                    y_pred = lr.predict(x_train_std)
                    mse = skm.mean_squared_error(y_train_std, y_pred)
                    r2 = skm.r2_score(y_train_std, y_pred)
                    mse_array[i] = mse
                    r2_array[i] = r2
                    rSS_ols = 0
                    for m in range(0, Size_train, 1):
                        rSS_ols += (y_pred[m] - y_train_std[m])**2
                    aIC_ols = 2 * nonzero_count + Size_train * np.log(rSS_ols)
                    aIC_array[i] = aIC_ols
                    x_train_std[:, i] = tmp[:, 0]
            # end for
            if Criterion == 'MSE':
                drop = np.argmin(mse_array)
            if Criterion == 'R2':
                drop = np.argmax(r2_array)
            if Criterion == 'AIC':
                drop = np.argmin(aIC_array)
            drop_order.append(features_index[drop])
            x_train_std = np.delete(x_train_std, drop, 1) # last 1 = column, 0 = row
            del(features_index[drop])
        # end of while
    # end of if
# add last dropped feature to list  
    features_index.append(drop_order[-1])
    del(drop_order[-1])
    del(rmse_OLS_list[-1])
    del(rsquared_list[-1])
    if Method == 'sequential':
        del(aIC_list[-1])
    del(ols_coef_list[-1])
    del(nonzero_count_list[-1])
    idx_return = copy.deepcopy(features_index)
    if (N_of_last_iterations_to_store is not None) and (writeResults is not None):
        if N_of_last_iterations_to_store > len(ols_coef_list):
            N_of_last_iterations_to_store = len(ols_coef_list)
        count = 0
# fit raw data
        lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
        while count < N_of_last_iterations_to_store:
            nonzero_count = len(features_index)
            nonzero_count_str = str(nonzero_count)
            Table = pd.DataFrame(np.zeros(shape = (nonzero_count, 17)).astype(float), \
                columns=['Feature index','Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2',\
                'Intermolecular 2','Bond 3','Power 3','Intermolecular 3', \
                'Number of distances in feature','Normalized coefficients',\
                'Normalized RMSE','Normalized R2', 'Coefficients',\
                'RMSE','R2'], dtype=str)
            max_distances_in_feature = 1
            X_train_sel = np.zeros(shape=(Size_train, len(features_index))) # create working array of features 
            X_test_sel = np.zeros(shape=(Size_test, len(features_index))) # create working array of features 
            j = 0
            for i in features_index:
                X_train_sel[:, j] = X_train[:, i]
                X_test_sel[:, j] = X_test[:, i]
                j += 1
            lr.fit(X_train_sel, Y_train)
            lr_coef = lr.coef_
            y_pred = lr.predict(X_test_sel)
            mse = skm.mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = skm.r2_score(Y_test, y_pred) 
            j = 0 # index of reduced set
            for i in features_index: # index of full set
                Table.loc[j]['Feature index'] = i
                Table.loc[j]['Bond 1'] = FeaturesReduced[i].DtP1.Distance.Atom1.Symbol + '-' + FeaturesReduced[i].DtP1.Distance.Atom2.Symbol
                Table.loc[j]['Power 1'] = FeaturesReduced[i].DtP1.Power
                if FeaturesReduced[i].DtP1.Distance.isIntermolecular: 
                    Table.loc[j]['Intermolecular 1'] = 'Yes'
                else:
                    Table.loc[j]['Intermolecular 1'] = 'No'
                if FeaturesReduced[i].nDistances >= 2:
                    Table.loc[j]['Bond 2'] = FeaturesReduced[i].DtP2.Distance.Atom1.Symbol + '-' + FeaturesReduced[i].DtP2.Distance.Atom2.Symbol
                    Table.loc[j]['Power 2'] = FeaturesReduced[i].DtP2.Power
                    if max_distances_in_feature < 2:
                        max_distances_in_feature = 2
                    if FeaturesReduced[i].DtP2.Distance.isIntermolecular:
                        Table.loc[j]['Intermolecular 2'] = 'Yes'
                    else:
                        Table.loc[j]['Intermolecular 2'] = 'No'
                else:
                    Table.loc[j]['Bond 2'] = ''
                    Table.loc[j]['Power 2'] = ''
                    Table.loc[j]['Intermolecular 2'] = ''
                counter = 0
                current_feature_type = FeaturesReduced[i].FeType
                for k in range(0, len(FeaturesAll), 1):
                    if FeaturesAll[k].FeType == current_feature_type:
                        counter += 1
                Table.loc[j]['Number of distances in feature'] = counter
                Table.loc[j]['Normalized coefficients'] = ols_coef_list[len(ols_coef_list)-1-count][j]
                Table.loc[j]['Coefficients'] = lr_coef[j]
                if j == 0:
                    Table.loc[0]['Normalized RMSE'] = rmse_OLS_list[len(rmse_OLS_list)-1-count]
                    Table.loc[0]['Normalized R2'] = rsquared_list[len(rsquared_list)-1-count]
                    Table.loc[0]['RMSE'] = rmse
                    Table.loc[0]['R2'] = r2
                else:
                    Table.loc[j]['Normalized RMSE'] = ''
                    Table.loc[j]['Normalized R2'] = ''
                    Table.loc[j]['RMSE'] = ''
                    Table.loc[j]['R2'] = ''
                j += 1
            # end of for
            if max_distances_in_feature == 1:
                del(Table['Bond 2'])
                del(Table['Power 2'])
                del(Table['Intermolecular 2'])
            if writeResults is not None:
                Table.to_excel(writeResults, nonzero_count_str)
            features_index.append(drop_order[len(drop_order)-1-count])
            count += 1
        # end of while        
    if PlotPath:
        # Plot RMSE vs. active coefficients number        
        plt.figure(100, figsize = (19, 10))
        plt.plot(nonzero_count_list, rmse_OLS_list, ':')
        plt.xlabel('Active coefficients')
        plt.ylabel('Root of mean square error')
        plt.title('Backward elimination. Root of mean square error vs Active coefficiants')
        plt.axis('tight')
        plt.savefig(FileName + '_RMSE', bbox_inches='tight')
        # Plot R2 vs. active coefficiens number
        plt.figure(101, figsize = (19, 10))
        plt.plot(nonzero_count_list, rsquared_list, ':')
        plt.xlabel('Active coefficients')
        plt.ylabel('R2')
        plt.title('Backward elimination. R2 vs Active coefficiants')
        plt.axis('tight')
        plt.savefig(FileName + '_R2', bbox_inches='tight')
    # end of plot
    Results = pd.DataFrame(np.zeros(shape = (len(nonzero_count_list), 1)).astype(float),\
        columns=['Empty'], dtype=float)
    Results.insert(1, 'N Non-zero coef', nonzero_count_list)
    Results.insert(2, 'RMSE', rmse_OLS_list)
    Results.insert(3, 'R2', rsquared_list)
    Results.insert(4, 'Drop order', drop_order)
    del Results['Empty']
    if writeResults is not None:
        Results.to_excel(writeResults,'Summary')
    if (StorePath is not None) and (writeResults is not None):
        writeResults.save()
    return idx_return
# end of BackwardElimination

def RemoveWorstFeature(x, y, Method='fast', Criterion='p-Value',\
    idx=None, VIP_idx=None, sort=True, verbose=True):
# x and y are supposed to be standardized
    if VIP_idx is None:
        VIP_idx = []
# features_index - list of indices of selected features
    if idx is None:
        feature_idx = list(range(0, x.shape[1], 1))
    else:
        feature_idx = list(idx)
        del(idx)
    if len(VIP_idx) >= len(feature_idx): # nothing to remove
        return -1
    Size = x.shape[0] # number of observations
    size = len(feature_idx) # size of selected features list
    x_sel = np.zeros(shape=(Size, size)) # create working array of features 
    j = 0
# copy features defined by features_index from training set to x_sel
    for i in feature_idx:
        x_sel[:, j] = x[:, i]
        j += 1
    if Method == 'fast':
        ols = sm.OLS(endog = y, exog = x_sel, hasconst = False).fit()
        pvalues = ols.pvalues
        if sort:
# bubble sort from low to high. less important features with greater p-Value first
            pvalues = Sort(pvalues, direction='Lo-Hi')
        Found = False
        while not Found:
            drop = np.argmax(pvalues)
            index = feature_idx[drop]
            if index in VIP_idx:
                pvalues[drop] = -1
            else:
                Found = True
        del(feature_idx[drop])
        return feature_idx
    if Method == 'sequential':
        lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
        z = np.zeros(shape=(Size, 1), dtype=float)
        tmp = np.zeros(shape=(Size, 1), dtype=float)
        mse_list = []
        for i in range(0, size, 1):
            tmp[:, 0] = x_sel[:, i] # save column
            x_sel[:, i] = z[:, 0] # copy zeros to column
            lr.fit(x_sel, y)
            y_pred = lr.predict(x_sel)
            mse = skm.mean_squared_error(y, y_pred)
            mse_list.append(mse)
            x_sel[:, i] = tmp[:, 0] # restore column
        if sort:
# bubble sort from low to high. less important features first
            mse_list, feature_idx = Sort(mse_list, feature_idx, direction='Lo-Hi')
        Found = False
        while not Found:
            drop = np.argmin(mse_list)
            index = feature_idx[drop]
            if index in VIP_idx:
                mse_list[drop] = 1e+100
            else:
                Found = True
        del(feature_idx[drop])   
        return feature_idx
    return
# end of RemoveWorstFeature

# Calculate variance inflation factor for all features
def CalculateVif(X):
# returns dataframe with variance inflation factors
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif = vif.round(1)
    return vif

def DecisionTreeEstimator(x_std, y_std, MaxFeatures=10, verbose=False):
# returns list of max_features elements of most important features
    from sklearn.tree import DecisionTreeRegressor
    if verbose:
        print('Decision Tree Estimator. Number of features to return = ', MaxFeatures)
    r1 = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, \
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
        max_features=MaxFeatures, random_state=101, max_leaf_nodes=None, \
        min_impurity_split=1e-57, presort=False)
    r1.fit(x_std, y_std)
    feature_importance_tree = r1.feature_importances_
    idx = []
    for i in range(0, MaxFeatures, 1):
        tmp = np.argmax(feature_importance_tree)
        idx.append(tmp)
        feature_importance_tree[tmp] = 0
    return idx


def RandomForestEstimator(x_std, y_std, MaxFeatures=10, NofTrees=10, verbose=False):
# returns list of max_features elements of most important features
    from sklearn.ensemble import RandomForestRegressor
    if verbose:
        print('Random Forest Estimator. Number of features to return = ', MaxFeatures)
    r2 = RandomForestRegressor(n_estimators=NofTrees, criterion='mse', max_depth=None, \
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
        max_features=MaxFeatures, max_leaf_nodes=None, min_impurity_split=1e-57, \
        bootstrap=True, oob_score=False, n_jobs=-1, random_state=101, verbose=0, \
        warm_start=False)
    r2.fit(x_std, y_std)
    feature_importance_forest = r2.feature_importances_
    idx = []
    for i in range(0, MaxFeatures, 1):
        tmp = np.argmax(feature_importance_forest)
        idx.append(tmp)
        feature_importance_forest[tmp] = 0
    return idx

def get_full_list(idx, NofFeatures):
    idx_corr = []
    n = np.array(range(0, NofFeatures, 1), dtype=int)
    for i in range(0, len(idx), 1):
        idx_corr.append(n)
    return idx_corr

def ClassifyCorrelatedFeatures(X, features_idx, MinCorrelation, Model=1, Corr_Matrix=None, verbose=False):
# creates list of correlated features 
# X - [n x m] numpy array of features
# rows correspond to observations
# columns correspond to features
# features_idx - indices of selected features
# correlation value is given by MinCorrelation
# Model 1: correlated features may overlap each other
# Model 2: correlated features do not overlap each other. Works faster

    def InList(List, Value):
# returns True if Value is in List        
        for i in range(0, len(List), 1):
            for j in range(0, len(List[i]), 1):
                if List[i][j] == Value:
                    return True
        return False
    
    if verbose:
        print('Calculate correlation matrix')
    if Corr_Matrix is not None:
        C = Corr_Matrix
    else:
        C = np.cov(X, rowvar=False, bias=True)
    list1 = []
    list2 = []
    features_lars_idx = list(features_idx)
    for i in range(0, len(features_lars_idx), 1):
        idx = features_lars_idx[i]
        list1.append(list(''))
        list2.append(list(''))
        list1[i].append(idx)
        list2[i].append(idx)
    NofFeatures = X.shape[1]
    k = 0
    for i in features_lars_idx:
        for j in range(0, NofFeatures, 1):
            if i == j:
                continue
            if C[i, j] > MinCorrelation:
                if j not in list1:
                    list1[k].append(j)
                if not InList(list2, j):
                    list2[k].append(j)
        k += 1
    if Model == 1:
        return list1
    if Model == 2:
        return list2
    
# store in sheet of .xlsx file description of fit with real coefficients
def Results_to_xls(writeResults, SheetName, selected_features_list, FeaturesAll,\
        FeaturesReduced, data_to_xls):
# writeResults = pd.ExcelWriter('FileName.xlsx', engine='openpyxl')
# after calling this function(s) data must be stored in the file by calling writeResults.save()
# SheetName - name of xlsx sheet
# selected_features_list - indices of features to proceed
# X - [n x m] numpy array of not normalized features
# rows correspond to observations
# columns correspond to features
# Y - [n x 1] numpy array, recponse variable
# FeaturesAll - class1.InvPowDistancesFeature object. Contains all features
# FeaturesReduced - class1.InvPowDistancesFeature object. Contains combined features
    NofSelectedFeatures = len(selected_features_list)
    mse_train = data_to_xls[0][2]
    rmse_train = data_to_xls[0][3]
    r2_train = data_to_xls[0][4]
    if len(data_to_xls) > 1:
        coef = data_to_xls[1][0]
        mse_test = data_to_xls[1][2]
        rmse_test = data_to_xls[1][3]
        r2_test = data_to_xls[1][4]
    Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 24)).astype(float), \
        columns=['Feature index','Feature type','Bond 1','Power 1','Intermolecular 1',\
        'Distance 1 type','Bond 2','Power 2', 'Intermolecular 2','Distance 2 type',\
        'Order 1','Degree 1','HaType 1','Order 2','Degree 2','HaType 2',\
        'Number of distances in feature','Coefficients','MSE Train','RMSE Train',\
        'R2 Train','MSE Test','RMSE Test','R2 Test'], dtype=str)
    max_distances_in_feature = 1
    max_harmonics_in_feature = 0
    for i in range(0, NofSelectedFeatures, 1):
        index = selected_features_list[i]
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
        if FeaturesReduced[index].nHarmonics >= 1:   
            Results.loc[i]['Order 1'] = FeaturesReduced[index].Harmonic1.Order
            Results.loc[i]['Degree 1'] = FeaturesReduced[index].Harmonic1.Degree
            Results.loc[i]['HaType 1'] = FeaturesReduced[index].Harmonic1.HaType
            if max_harmonics_in_feature < 1:
                max_harmonics_in_feature = 1
        else:
            Results.loc[i]['Order 1'] = ''
            Results.loc[i]['Degree 1'] = ''
            Results.loc[i]['HaType 1'] = ''
        if FeaturesReduced[index].nHarmonics >= 2:   
            Results.loc[i]['Order 2'] = FeaturesReduced[index].Harmonic2.Order
            Results.loc[i]['Degree 2'] = FeaturesReduced[index].Harmonic2.Degree
            Results.loc[i]['HaType 2'] = FeaturesReduced[index].Harmonic2.HaType
            if max_harmonics_in_feature < 2:
                max_harmonics_in_feature = 2
        else:
            Results.loc[i]['Order 2'] = ''
            Results.loc[i]['Degree 2'] = ''
            Results.loc[i]['HaType 2'] = ''            
        counter = 0
        current_feature_type = FeaturesReduced[index].FeType
        for j in range(0, len(FeaturesAll), 1):
            if FeaturesAll[j].FeType == current_feature_type:
                counter += 1
        Results.loc[i]['Number of distances in feature'] = counter
        Results.loc[i]['Coefficients'] = coef[i]
        if i == 0:
            Results.loc[0]['MSE Train'] = mse_train
            Results.loc[0]['RMSE Train'] = rmse_train
            Results.loc[0]['R2 Train'] = r2_train
            Results.loc[0]['MSE Test'] = mse_test
            Results.loc[0]['RMSE Test'] = rmse_test
            Results.loc[0]['R2 Test'] = r2_test
        else:
            Results.loc[i]['MSE Train'] = ''
            Results.loc[i]['RMSE Train'] = ''
            Results.loc[i]['R2 Train'] = ''
            Results.loc[i]['MSE Test'] = ''
            Results.loc[i]['RMSE Test'] = ''
            Results.loc[i]['R2 Test'] = ''
    if max_distances_in_feature == 1:
        del(Results['Bond 2'])
        del(Results['Power 2'])
        del(Results['Intermolecular 2'])
        del(Results['Distance 2 type'])
    if max_harmonics_in_feature < 2:
        del(Results['Order 2'])
        del(Results['Degree 2'])
        del(Results['HaType 2'])
    if max_harmonics_in_feature < 1:
        del(Results['Order 1'])
        del(Results['Degree 1'])
        del(Results['HaType 1'])
    Results.to_excel(writeResults, sheet_name=SheetName)
    return
# end of Results_to_xls

def StoreFeaturesDescriprion(FileName, FeaturesAll, FeaturesReduced):
    writeResults = pd.ExcelWriter(FileName, engine='openpyxl')
    NofSelectedFeatures = len(FeaturesReduced)
    selected_features_list = list(range(0,NofSelectedFeatures,1))
    Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 17)).astype(float), \
        columns=['Feature index','Feature type','Bond 1','Power 1','Intermolecular 1','Distance 1 type','Bond 2','Power 2', \
        'Intermolecular 2', 'Distance 2 type','Order 1','Degree 1','HaType 1','Order 2','Degree 2','HaType 2','Number of distances in feature'], dtype=str)
    max_distances_in_feature = 1
    max_harmonics_in_feature = 0
    for i in range(0, NofSelectedFeatures, 1):
        index = selected_features_list[i]
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
        if FeaturesReduced[index].nHarmonics >= 1:   
            Results.loc[i]['Order 1'] = FeaturesReduced[index].Harmonic1.Order
            Results.loc[i]['Degree 1'] = FeaturesReduced[index].Harmonic1.Degree
            Results.loc[i]['HaType 1'] = FeaturesReduced[index].Harmonic1.HaType
            if max_harmonics_in_feature < 1:
                max_harmonics_in_feature = 1
        else:
            Results.loc[i]['Order 1'] = ''
            Results.loc[i]['Degree 1'] = ''
            Results.loc[i]['HaType 1'] = ''
        if FeaturesReduced[index].nHarmonics >= 2:   
            Results.loc[i]['Order 2'] = FeaturesReduced[index].Harmonic2.Order
            Results.loc[i]['Degree 2'] = FeaturesReduced[index].Harmonic2.Degree
            Results.loc[i]['HaType 2'] = FeaturesReduced[index].Harmonic2.HaType
            if max_harmonics_in_feature < 2:
                max_harmonics_in_feature = 2
        else:
            Results.loc[i]['Order 2'] = ''
            Results.loc[i]['Degree 2'] = ''
            Results.loc[i]['HaType 2'] = ''            
        counter = 0
        current_feature_type = FeaturesReduced[index].FeType
        for j in range(0, len(FeaturesAll), 1):
            if FeaturesAll[j].FeType == current_feature_type:
                counter += 1
        Results.loc[i]['Number of distances in feature'] = counter
    if max_distances_in_feature == 1:
        del(Results['Bond 2'])
        del(Results['Power 2'])
        del(Results['Intermolecular 2'])
        del(Results['Distance 2 type'])
    if max_harmonics_in_feature < 2:
        del(Results['Order 2'])
        del(Results['Degree 2'])
        del(Results['HaType 2'])
    if max_harmonics_in_feature < 1:
        del(Results['Order 1'])
        del(Results['Degree 1'])
        del(Results['HaType 1'])
    Results.to_excel(writeResults)
    writeResults.save()  
    return

# Get fit score for training set only
def get_train_fit_score(X_train, Y_train, idx=None):
    if idx is None:
        x_sel = X_train
    else: # copy only selected indices
        size = len(idx)
        x_sel = np.zeros(shape=(X_train.shape[0], size), dtype=float) # matrix with selected features
        for i in range(0, size, 1):
            x_sel[:, i] = X_train[:, idx[i]]
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
    lr.fit(x_sel, Y_train)
    coef = lr.coef_
    y_pred = lr.predict(x_sel)
    mse = skm.mean_squared_error(Y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y_train, y_pred)
    nonzero_count = np.count_nonzero(coef)
    return coef, nonzero_count, mse, rmse, r2

# LinearRegression
def get_fit_score(X_train, Y_train, X_test, Y_test, idx=None, Test=True):
    if idx is None:
        x_sel_train = X_train
        x_sel_test = X_test
    else:
        size = len(idx)
        x_sel_train = np.zeros(shape=(X_train.shape[0], size), dtype=float) # matrix with selected features
        x_sel_test = np.zeros(shape=(X_test.shape[0], size), dtype=float) # matrix with selected features
        for i in range(0, size, 1):
            x_sel_train[:, i] = X_train[:, idx[i]]
            x_sel_test[:, i] = X_test[:, idx[i]]
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
    lr.fit(x_sel_train, Y_train)
    coef = lr.coef_
    if Test:
        y_pred = lr.predict(x_sel_test)
        mse = skm.mean_squared_error(Y_test, y_pred)
        r2 = skm.r2_score(Y_test, y_pred)
    else:
        y_pred = lr.predict(x_sel_train)
        mse = skm.mean_squared_error(Y_train, y_pred)
        r2 = skm.r2_score(Y_train, y_pred)
    rmse = np.sqrt(mse)
    nonzero_count = np.count_nonzero(coef)
    return coef, nonzero_count, mse, rmse, r2

# statsmodels.ols
def get_fit_score2(X_train, Y_train, X_test, Y_test, idx=None):
    if idx is None:
        x_sel_train = X_train
        x_sel_test = X_test
    else:
        size = len(idx)
        x_sel_train = np.zeros(shape=(X_train.shape[0], size), dtype=float) # matrix with selected features
        x_sel_test = np.zeros(shape=(X_test.shape[0], size), dtype=float) # matrix with selected features
        for i in range(0, size, 1):
            x_sel_train[:, i] = X_train[:, idx[i]]
            x_sel_test[:, i] = X_test[:, idx[i]]           
    ols = sm.OLS(endog = Y_train, exog = x_sel_train, hasconst = False).fit()
    y_pred = ols.predict(x_sel_test)
    ols_coef = ols.params
    nonzero_count = np.count_nonzero(ols_coef)
    r2 = ols.rsquared
    mse = skm.mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    return ols_coef, nonzero_count, mse, rmse, r2

# normalize before fit
def get_fit_score3(X_train, Y_train, X_test, Y_test, idx=None):
    x_scale = StandardScaler(copy=True, with_mean=True, with_std=True)
    y_scale = StandardScaler(copy=True, with_mean=True, with_std=False)
    x_scale.fit(X_train)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    y_scale.fit(Y_train)
    x_train_std = x_scale.transform(X_train)
    x_test_std = x_scale.transform(X_test)
    y_train_std = y_scale.transform(Y_train)
    y_test_std = y_scale.transform(Y_test)
    x_var = x_scale.var_
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    y_train_std = y_train_std.reshape(-1)
    y_test_std = y_test_std.reshape(-1)
    
    if idx is None:
        x_sel_train = x_train_std
        x_sel_test = x_test_std
        idx = list(range(0, X_train.shape[1], 1))
        size = len(idx)
    else:
        size = len(idx)
        x_sel_train = np.zeros(shape=(X_train.shape[0], size), dtype=float) # matrix with selected features
        x_sel_test = np.zeros(shape=(X_test.shape[0], size), dtype=float) # matrix with selected features
        for i in range(0, size, 1):
            x_sel_train[:, i] = x_train_std[:, idx[i]]
            x_sel_test[:, i] = x_test_std[:, idx[i]]
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    lr.fit(x_sel_train, y_train_std)
    coef = lr.coef_
    y_pred = lr.predict(x_sel_test)
    mse = skm.mean_squared_error(y_test_std, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y_test, y_pred)
    nonzero_count = np.count_nonzero(coef)
    orig_coef = np.zeros(shape=(size), dtype=float)
    for i in range(0, size, 1):
        orig_coef[i] = coef[i] / np.sqrt(x_var[idx[i]])
    return orig_coef, nonzero_count, mse, rmse, r2

# use only training set with manual standardization
def get_fit_score4(X_train, Y_train, idx=None):
    x_scale = StandardScaler(copy=True, with_mean=True, with_std=True)
    y_scale = StandardScaler(copy=True, with_mean=True, with_std=False)
    x_scale.fit(X_train)
    Y_train = Y_train.reshape(-1, 1)
    y_scale.fit(Y_train)
    x_std = x_scale.transform(X_train)
    y_std = y_scale.transform(Y_train)
    x_var = x_scale.var_
    Y_train = Y_train.reshape(-1)
    y_std = y_std.reshape(-1)
    if idx is None:
        x_sel = x_std
        idx = list(range(0, x_std.shape[1], 1))
        size = len(idx)
    else:
        size = len(idx)
        x_sel = np.zeros(shape=(x_std.shape[0], size), dtype=float) # matrix with selected features
        for i in range(0, size, 1):
            x_sel[:, i] = x_std[:, idx[i]]
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    lr.fit(x_sel, y_std)
    coef = lr.coef_
    y_pred = lr.predict(x_sel)
    mse = skm.mean_squared_error(y_std, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(y_std, y_pred)
    nonzero_count = np.count_nonzero(coef)
    orig_coef = np.zeros(shape=(size), dtype=float)
    for i in range(0, size, 1):
        orig_coef[i] = coef[i] / np.sqrt(x_var[idx[i]])
    return orig_coef, nonzero_count, mse, rmse, r2

def plot_rmse(FileName, nonzero_count_list, rmse_list, r2_list):
# Plot RMSE vs. active coefficients number        
    fig = plt.figure(100, figsize = (19, 10))
    plt.plot(nonzero_count_list, rmse_list, ':')
    plt.xlabel('Active coefficients')
    plt.ylabel('Root of mean square error')
    plt.title('Root of mean square error vs Active coefficiants')
    plt.axis('tight')
    plt.show()
    plt.savefig(FileName + '_RMSE', bbox_inches='tight')
    plt.close(fig)
    # Plot R2 vs. active coefficiens number
    fig = plt.figure(101, figsize = (19, 10))
    plt.plot(nonzero_count_list, r2_list, ':')
    plt.xlabel('Active coefficients')
    plt.ylabel('R2')
    plt.title('R2 vs Active coefficiants')
    plt.axis('tight')
    plt.show()
    plt.savefig(FileName + '_R2', bbox_inches='tight')
    plt.close(fig)
    return

def Fit(x_std, y_std, active_features, corr_list, VIP_idx=None):# local index
    if VIP_idx is None:
        VIP_idx = []
    size = len(active_features)
    Size = x_std.shape[0]
    x_sel = np.zeros(shape=(Size, size), dtype=float)
    tmp = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
# creating selected features array
    for i in range(0, size, 1):
        x_sel[:, i] = x_std[:, active_features[i]]
    idx_array = np.zeros(shape=(size), dtype=int)
    mse_array = np.zeros(shape=(size), dtype=float)
    for i in range(0, size, 1):# i - local index
        k = active_features[i] # global index
        if k in VIP_idx: # do not change this feature. get fit only for one
            lr.fit(x_sel, y_std)
            y_pred = lr.predict(x_sel)
            mse = skm.mean_squared_error(y_std, y_pred)
            idx_array[i] = k # global index 
            mse_array[i] = mse # mse
        else:        
            replace_list = corr_list[i]
            size_current = len(replace_list)
            mse_array_current = np.zeros(shape=(size_current), dtype=float)
            idx_array_current = np.zeros(shape=(size_current), dtype=float)
            for j in range(0, size_current, 1):
                tmp[:, 0] = x_sel[:, i]
                x_sel[:, i] = x_std[:, replace_list[j]]
                lr.fit(x_sel, y_std)
                y_pred = lr.predict(x_sel)
                mse = skm.mean_squared_error(y_std, y_pred)
                mse_array_current[j] = mse
                idx_array_current[j] = replace_list[j] # global index
                x_sel[:, i] = tmp[:, 0]
            idx = np.argmin(mse_array_current) # local index of best mse
            mse = mse_array_current[idx] # best mse for one feature in corr list
            idx_array[i] = idx_array_current[idx] # global index of best mse
            mse_array[i] = mse
    return [idx_array, mse_array]

def FindBestSet(x_std, y_std, features_idx, corr_list, VIP_idx=None, Method='MSE', verbose=False):
# finds alternative fit using classified feature list produced by ClassifyCorrelatedFeatures  
# returns list of indices of features
# x_std - [n x m] numpy array of standardized features
# rows correspond to observations
# columns correspond to features
# y_std - [n x 1] numpy array, standardized recponse variable
# features_idx - indices of selected features
# classified_list - list of classified features
# Fit = 1 or 2. Two different approaches
# Method='MSE' based on minimum value of ordinary list squared fit
# Method='R2' based on maximum value of R2
    active_features = copy.deepcopy(features_idx)
    Size = x_std.shape[0]
    size = len(active_features)
    x_sel = np.zeros(shape=(Size, size), dtype=float)
    # creating selected features array
    for i in range(0, size, 1):
        x_sel[:, i] = x_std[:, active_features[i]]
    # initial fit
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    lr.fit(x_sel, y_std)
    y_pred = lr.predict(x_sel)
    mse = skm.mean_squared_error(y_std, y_pred)
    best_mse = mse
    Found = True
    while Found:
        Found = False
        output = Fit(x_std, y_std, active_features, corr_list, VIP_idx=VIP_idx)
        for j in range(0, size, 1):
            if output[1][j] < best_mse:
                best_idx_local = j # local index
                best_idx_global = output[0][j] # global index
                best_mse = output[1][j]
                Found = True
        if Found:
            active_features[best_idx_local] = best_idx_global
            if verbose:
                print('Current Best Fit = ', best_mse)
    return active_features

def FindBestSetMP(x_std, y_std, features_idx, corr_list, Method='MSE', verbose=True):
# finds alternative fit using classified feature list produced by ClassifyCorrelatedFeatures  
# returns list of indices of features
# x_std - [n x m] numpy array of standardized features
# rows correspond to observations
# columns correspond to features
# y_std - [n x 1] numpy array, standardized recponse variable
# features_idx - indices of selected features
# classified_list - list of classified features
# Fit = 1 or 2. Two different approaches
# Method='MSE' based on minimum value of ordinary list squared fit
# Method='R2' based on maximum value of R2
    active_features = copy.deepcopy(features_idx)
    x_sel = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    tmp = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    # creating selected features array
    for i in active_features:
        if i == active_features[0]:# first column
            x_sel[:, 0] = x_std[:, i]
        else:
            tmp[:, 0] = x_std[:, i]
            x_sel = np.concatenate((x_sel, tmp), axis=1)
    # initial fit
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    lr.fit(x_sel, y_std)
    y_pred = lr.predict(x_sel)
    mse = skm.mean_squared_error(y_std, y_pred)
    best_mse = mse
    Found = True
    nCPU = mp.cpu_count()
    nFeatures = len(active_features)
    nFeatures_per_CPU = int(nFeatures / nCPU)
    if nFeatures_per_CPU == 0:
        nFeatures_per_CPU = 1
    j = 0 # 0 .. nFeatures
    idx_list = []
    while j < nFeatures:
        idx_small = []
        for i in range(0, nFeatures_per_CPU, 1): # 0 .. nFeatures_per_CPU
            if j < nFeatures:
                idx_small.append(j)
            j += 1
        idx_list.append(idx_small)
    ran = list(range(0, len(idx_list), 1))
    print(idx_list)
    while Found:
        Found = False
        jobs = (delayed(Fit)(x_std, y_std, idx_list[i], active_features, corr_list) for i in ran)
        output = Parallel(n_jobs=nCPU)(jobs)
        for i in range(0, len(output), 1):
            for j in range(0, len(output[i][1]), 1):
                if output[i][1][j] < best_mse:
                    best_idx_local = i*len(output[i][1])+j # local index
                    best_idx_global = output[i][0][j] # global index
                    best_mse = output[i][1][j]
                    Found = True
        if Found:
            if verbose:
                print(best_mse)
            active_features[best_idx_local] = best_idx_global
    return active_features

def FindBestSetMP2(x_std, y_std, features_idx, corr_list, Method='MSE', n_jobs=-1, verbose=True):
# finds alternative fit using classified feature list produced by ClassifyCorrelatedFeatures  
# returns list of indices of features
# x_std - [n x m] numpy array of standardized features
# rows correspond to observations
# columns correspond to features
# y_std - [n x 1] numpy array, standardized recponse variable
# features_idx - indices of selected features
# classified_list - list of classified features
# Fit = 1 or 2. Two different approaches
# Method='MSE' based on minimum value of ordinary list squared fit
# Method='R2' based on maximum value of R2
    active_features = copy.deepcopy(features_idx)
    x_sel = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    tmp = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    # creating selected features array
    for i in active_features:
        if i == active_features[0]:# first column
            x_sel[:, 0] = x_std[:, i]
        else:
            tmp[:, 0] = x_std[:, i]
            x_sel = np.concatenate((x_sel, tmp), axis=1)
    # initial fit
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    lr.fit(x_sel, y_std)
    y_pred = lr.predict(x_sel)
    mse = skm.mean_squared_error(y_std, y_pred)
    best_mse = mse
    Found = True
    if n_jobs == -1:
        nCPU = mp.cpu_count()
    else:
        nCPU = n_jobs
    pool = mp.Pool(processes=nCPU)
    nFeatures = len(active_features)
    nFeatures_per_CPU = int(nFeatures / nCPU)
    if nFeatures_per_CPU == 0:
        nFeatures_per_CPU = 1
    j = 0 # 0 .. nFeatures
    idx_list = []
    while j < nFeatures:
        idx_small = []
        for i in range(0, nFeatures_per_CPU, 1): # 0 .. nFeatures_per_CPU
            if j < nFeatures:
                idx_small.append(j)
            j += 1
        idx_list.append(idx_small)
    ran = list(range(0, len(idx_list), 1))
    while Found:
        Found = False
        out = []
        for i in ran:
            out.append(pool.apply_async(Fit, args=(x_std, y_std, idx_list[i], active_features, corr_list)))
        output = [p.get() for p in out]    
        Idx = []
        Mse = []
        for i in range(0, len(output), 1):
            for j in range(0, len(output[i][0]), 1):
                Idx.append(output[i][0][j])
                Mse.append(output[i][1][j])
        idx_min = np.argmin(Mse)
        if Mse[idx_min] < best_mse:
            best_idx_local = idx_min
            best_mse = Mse[best_idx_local]
            best_idx_global = Idx[idx_min]
            Found = True
        del(output)                  
        if Found:
            if verbose:
                print('Best MSE = ', best_mse)
            active_features[best_idx_local] = best_idx_global
    pool.close()
    pool.join()
    return active_features

def ForwardSequential(x_std, y_std, nVariables=10, idx=None):
    if (idx is not None) and (nVariables <= len(idx)):
        print('Number of variables cannot be less or equal length of VIP features')
        quit()
    NofFeatures = x_std.shape[1]
    Size = x_std.shape[0]
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    if idx is None: # determine first feature if initial fit does not exist
        x_selected = np.zeros(shape=(Size, 1), dtype=float) # matrix with selected features
        mse_array = np.zeros(shape=(NofFeatures), dtype=float)
        for i in range(0, NofFeatures, 1):
            x_selected[:, 0] = x_std[:, i]
            lr.fit(x_selected, y_std)
            y_pred = lr.predict(x_selected)
            mse = skm.mean_squared_error(y_std, y_pred)
            mse_array[i] = mse
        idx1 = np.argmin(mse_array) # get most important feature
        idx = [] #indices of selected features
        idx.append(idx1)
    x_selected = np.zeros(shape=(Size, len(idx)), dtype=float) # matrix with selected features
    for i in range(0, len(idx), 1):
        x_selected[:, i] = x_std[:, idx[i]]
    z = np.zeros(shape=(Size, 1), dtype=float)
    j = len(idx)
    while j < nVariables:
        print(j)
        mse_array = np.ones(shape=(NofFeatures), dtype=float)
        z = np.zeros(shape=(Size, 1), dtype=float)
        x_selected = np.concatenate((x_selected, z), axis=1)
        for i in range(0, NofFeatures, 1):
            if i in idx:
                continue
            x_selected[:, -1] = x_std[:, i]
            lr.fit(x_selected, y_std)
            y_pred = lr.predict(x_selected)
            mse = skm.mean_squared_error(y_std, y_pred)
            mse_array[i] = mse
        idx_mse_min = np.argmin(mse_array) # index of feature in current matrix with smallest RMSE
        idx.append(idx_mse_min)
        x_selected[:, -1] = x_std[:, idx_mse_min]
        j += 1
    return idx

def store_structure(FileName, Atoms, Distances, DtP_Double_list, FeaturesAll):
    Atom_table = pd.DataFrame(np.zeros(shape = (len(Atoms), 4)).astype(int), \
        columns=['Symbol','Index','Type','Molecular Index'], dtype=str)

    Dist_table = pd.DataFrame(np.zeros(shape = (len(Distances), 10)).astype(int), \
        columns=['IsIntermolecular','DiType','Atom1 Symbol','Atom1 Index',\
        'Atom1 AtType','Atom1 Molecular Index','Atom2 Symbol','Atom2 Index',\
        'Atom2 AtType','Atom2 Molecular Index'], dtype=str)

    Dtp_table = pd.DataFrame(np.zeros(shape = (len(DtP_Double_list), 12)).astype(int), \
        columns=['Power','DtpType','IsIntermolecular','DiType','Atom1 Symbol','Atom1 Index',\
        'Atom1 AtType','Atom1 Molecular Index','Atom2 Symbol','Atom2 Index',\
        'Atom2 AtType','Atom2 Molecular Index'], dtype=str)

    Feature_table = pd.DataFrame(np.zeros(shape = (len(FeaturesAll), 28)).astype(int), \
        columns=['#Distances','FeType','Category Molecular','Category Atomic','Power1','DtpType1','IsIntermolecular1',\
        'DiType1','Atom11 Symbol','Atom11 Index','Atom11 AtType','Atom11 Molecular Index',\
        'Atom12 Symbol','Atom12 Index','Atom12 AtType','Atom12 Molecular Index',\
        'Power2','DtpType2','IsIntermolecular2','DiType2','Atom21 Symbol','Atom21 Index',\
        'Atom21 AtType','Atom21 Molecular Index','Atom22 Symbol','Atom22 Index',\
        'Atom22 AtType','Atom22 Molecular Index'], dtype=str)
    
    for i in range(0, len(Atoms), 1):
        Atom_table.loc[i]['Symbol'] = Atoms[i].Symbol
        Atom_table.loc[i]['Index'] = Atoms[i].Index
        Atom_table.loc[i]['Type'] = Atoms[i].AtType
        Atom_table.loc[i]['Molecular Index'] = Atoms[i].MolecularIndex
        
    for i in range(0, len(Distances), 1):
        Dist_table.loc[i]['IsIntermolecular'] = Distances[i].isIntermolecular
        Dist_table.loc[i]['DiType'] = Distances[i].DiType
        Dist_table.loc[i]['Atom1 Symbol'] = Distances[i].Atom1.Symbol
        Dist_table.loc[i]['Atom1 Index'] = Distances[i].Atom1.Index
        Dist_table.loc[i]['Atom1 AtType'] = Distances[i].Atom1.AtType
        Dist_table.loc[i]['Atom1 Molecular Index'] = Distances[i].Atom1.MolecularIndex
        Dist_table.loc[i]['Atom2 Symbol'] = Distances[i].Atom2.Symbol
        Dist_table.loc[i]['Atom2 Index'] = Distances[i].Atom2.Index
        Dist_table.loc[i]['Atom2 AtType'] = Distances[i].Atom2.AtType
        Dist_table.loc[i]['Atom2 Molecular Index'] = Distances[i].Atom2.MolecularIndex
  
    for i in range(0, len(DtP_Double_list), 1):
        Dtp_table.loc[i]['Power'] = DtP_Double_list[i].Power
        Dtp_table.loc[i]['DtpType'] = DtP_Double_list[i].DtpType
        Dtp_table.loc[i]['IsIntermolecular'] = DtP_Double_list[i].Distance.isIntermolecular
        Dtp_table.loc[i]['DiType'] = DtP_Double_list[i].Distance.DiType
        Dtp_table.loc[i]['Atom1 Symbol'] = DtP_Double_list[i].Distance.Atom1.Symbol
        Dtp_table.loc[i]['Atom1 Index'] = DtP_Double_list[i].Distance.Atom1.Index
        Dtp_table.loc[i]['Atom1 AtType'] = DtP_Double_list[i].Distance.Atom1.AtType
        Dtp_table.loc[i]['Atom1 Molecular Index'] = DtP_Double_list[i].Distance.Atom1.MolecularIndex
        Dtp_table.loc[i]['Atom2 Symbol'] = DtP_Double_list[i].Distance.Atom2.Symbol
        Dtp_table.loc[i]['Atom2 Index'] = DtP_Double_list[i].Distance.Atom2.Index
        Dtp_table.loc[i]['Atom2 AtType'] = DtP_Double_list[i].Distance.Atom2.AtType
        Dtp_table.loc[i]['Atom2 Molecular Index'] = DtP_Double_list[i].Distance.Atom2.MolecularIndex


    for i in range(0, len(FeaturesAll), 1):
        if FeaturesAll[i].nDistances == 1:
            Feature_table.loc[i]['#Distances'] = FeaturesAll[i].nDistances
            Feature_table.loc[i]['FeType'] = FeaturesAll[i].FeType
            Feature_table.loc[i]['Category Molecular'] = int(FeaturesAll[i].FeType[-2])
            Feature_table.loc[i]['Category Atomic'] = int(FeaturesAll[i].FeType[-1])
            Feature_table.loc[i]['Power1'] = FeaturesAll[i].DtP1.Power
            Feature_table.loc[i]['DtpType1'] = FeaturesAll[i].DtP1.DtpType
            Feature_table.loc[i]['IsIntermolecular1'] = FeaturesAll[i].DtP1.Distance.isIntermolecular
            Feature_table.loc[i]['DiType1'] = FeaturesAll[i].DtP1.Distance.DiType
            Feature_table.loc[i]['Atom11 Symbol'] = FeaturesAll[i].DtP1.Distance.Atom1.Symbol
            Feature_table.loc[i]['Atom11 Index'] = FeaturesAll[i].DtP1.Distance.Atom1.Index
            Feature_table.loc[i]['Atom11 AtType'] = FeaturesAll[i].DtP1.Distance.Atom1.AtType
            Feature_table.loc[i]['Atom11 Molecular Index'] = FeaturesAll[i].DtP1.Distance.Atom1.MolecularIndex
            Feature_table.loc[i]['Atom12 Symbol'] = FeaturesAll[i].DtP1.Distance.Atom2.Symbol
            Feature_table.loc[i]['Atom12 Index'] = FeaturesAll[i].DtP1.Distance.Atom2.Index
            Feature_table.loc[i]['Atom12 AtType'] = FeaturesAll[i].DtP1.Distance.Atom2.AtType
            Feature_table.loc[i]['Atom12 Molecular Index'] = FeaturesAll[i].DtP1.Distance.Atom2.MolecularIndex
            Feature_table.loc[i]['Power2'] = ''
            Feature_table.loc[i]['DtpType2'] = ''
            Feature_table.loc[i]['IsIntermolecular2'] = ''
            Feature_table.loc[i]['DiType2'] = ''
            Feature_table.loc[i]['Atom21 Symbol'] = ''
            Feature_table.loc[i]['Atom21 Index'] = ''
            Feature_table.loc[i]['Atom21 AtType'] = ''
            Feature_table.loc[i]['Atom21 Molecular Index'] = ''
            Feature_table.loc[i]['Atom22 Symbol'] = ''
            Feature_table.loc[i]['Atom22 Index'] = ''
            Feature_table.loc[i]['Atom22 AtType'] = ''
            Feature_table.loc[i]['Atom22 Molecular Index'] = ''
        if FeaturesAll[i].nDistances == 2:
            Feature_table.loc[i]['#Distances'] = FeaturesAll[i].nDistances
            Feature_table.loc[i]['FeType'] = FeaturesAll[i].FeType
            Feature_table.loc[i]['Category Molecular'] = int(FeaturesAll[i].FeType[-2])
            Feature_table.loc[i]['Category Atomic'] = int(FeaturesAll[i].FeType[-1])
            Feature_table.loc[i]['Power1'] = FeaturesAll[i].DtP1.Power
            Feature_table.loc[i]['DtpType1'] = FeaturesAll[i].DtP1.DtpType
            Feature_table.loc[i]['IsIntermolecular1'] = FeaturesAll[i].DtP1.Distance.isIntermolecular
            Feature_table.loc[i]['DiType1'] = FeaturesAll[i].DtP1.Distance.DiType
            Feature_table.loc[i]['Atom11 Symbol'] = FeaturesAll[i].DtP1.Distance.Atom1.Symbol
            Feature_table.loc[i]['Atom11 Index'] = FeaturesAll[i].DtP1.Distance.Atom1.Index
            Feature_table.loc[i]['Atom11 AtType'] = FeaturesAll[i].DtP1.Distance.Atom1.AtType
            Feature_table.loc[i]['Atom11 Molecular Index'] = FeaturesAll[i].DtP1.Distance.Atom1.MolecularIndex
            Feature_table.loc[i]['Atom12 Symbol'] = FeaturesAll[i].DtP1.Distance.Atom2.Symbol
            Feature_table.loc[i]['Atom12 Index'] = FeaturesAll[i].DtP1.Distance.Atom2.Index
            Feature_table.loc[i]['Atom12 AtType'] = FeaturesAll[i].DtP1.Distance.Atom2.AtType
            Feature_table.loc[i]['Atom12 Molecular Index'] = FeaturesAll[i].DtP1.Distance.Atom2.MolecularIndex
            Feature_table.loc[i]['Power2'] = FeaturesAll[i].DtP2.Power
            Feature_table.loc[i]['DtpType2'] = FeaturesAll[i].DtP2.DtpType
            Feature_table.loc[i]['IsIntermolecular2'] = FeaturesAll[i].DtP2.Distance.isIntermolecular
            Feature_table.loc[i]['DiType2'] = FeaturesAll[i].DtP2.Distance.DiType
            Feature_table.loc[i]['Atom21 Symbol'] = FeaturesAll[i].DtP2.Distance.Atom1.Symbol
            Feature_table.loc[i]['Atom21 Index'] = FeaturesAll[i].DtP2.Distance.Atom1.Index
            Feature_table.loc[i]['Atom21 AtType'] = FeaturesAll[i].DtP2.Distance.Atom1.AtType
            Feature_table.loc[i]['Atom21 Molecular Index'] = FeaturesAll[i].DtP2.Distance.Atom1.MolecularIndex
            Feature_table.loc[i]['Atom22 Symbol'] = FeaturesAll[i].DtP2.Distance.Atom2.Symbol
            Feature_table.loc[i]['Atom22 Index'] = FeaturesAll[i].DtP2.Distance.Atom2.Index
            Feature_table.loc[i]['Atom22 AtType'] = FeaturesAll[i].DtP2.Distance.Atom2.AtType
            Feature_table.loc[i]['Atom22 Molecular Index'] = FeaturesAll[i].DtP2.Distance.Atom2.MolecularIndex

    writer = pd.ExcelWriter(FileName, engine='xlsxwriter')
    Atom_table.to_excel(writer, sheet_name='Atoms')
    Dist_table.to_excel(writer, sheet_name='Distances')
    Dtp_table.to_excel(writer, sheet_name='Distances with Powers')
    Feature_table.to_excel(writer, sheet_name='DD Features')

    workbook = writer.book
    format1 = workbook.add_format({'align':'center','valign':'vcenter'})
    format2 = workbook.add_format({'align':'center','valign':'vcenter','font_color': 'red'})
    worksheet1 = writer.sheets['Atoms']
    worksheet1.set_column('A:E', 16, format1)
    worksheet2 = writer.sheets['Distances']
    worksheet2.set_column('A:K', 16, format1)
    worksheet3 = writer.sheets['Distances with Powers']
    worksheet3.set_column('A:M', 16, format1)
    worksheet4 = writer.sheets['DD Features']    
    worksheet4.set_column('A:AC', 12, format1)
    worksheet4.set_column('J:J', 15, format2)
    worksheet4.set_column('N:N', 15, format2)
    worksheet4.set_column('V:V', 15, format2)
    worksheet4.set_column('Z:Z', 15, format2)
    writer.save() 

def rank_features(x_std, y_std, idx, direction='Lo-Hi'):
    # direction='Lo-Hi or direction='Hi-Lo'
    size = len(idx) # size of selected features list
    Size = x_std.shape[0] # number of observations
    z = np.zeros(shape=(Size, 1), dtype=float)
    tmp = np.zeros(shape=(Size, 1), dtype=float)
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    x_sel = np.zeros(shape=(Size, size), dtype=float)
    mse_list = list(range(0, size, 1))
# creating selected features array
    for i in range(0, size, 1):
        x_sel[:, i] = x_std[:, idx[i]] # copy selected features from initial set
    for i in range(0, size, 1):
        tmp[:, 0] = x_sel[:, i] # save column
        x_sel[:, i] = z[:, 0] # copy zeros to column
        lr.fit(x_sel, y_std)
        y_pred = lr.predict(x_sel)
        mse = skm.mean_squared_error(y_std, y_pred)
        mse_list[i] = mse 
        x_sel[:, i] = tmp[:, 0] # restore column
    if direction=='Lo-Hi':
# bubble sort from low to high. lower mse first = least important features first
        mse_list, idx = Sort(mse_list, idx, direction='Lo-Hi')
    else:
# bubble sort from high to low. Greater mse first = most important features first
        mse_list, idx = Sort(mse_list, idx, direction='Hi-Lo')
    return idx

class ENet:
    idx = None
    F_ENet = 'ENet path.png'
    L1 = 0.7
    eps = 1e-3
    nAlphas = 100
    alphas = None
    def __init__(self, F_ENet='ENet path.png', L1=0.7, eps=1e-3, nAlphas=100, alphas=None):
        self.F_ENet = F_ENet
        self.L1 = L1
        self.eps = eps
        self.nAlphas = nAlphas
        if alphas is not None:
            self.alphas = alphas
        return
    def fit(self, x, y, VIP_idx=None):
        if VIP_idx is None:
            VIP_idx = []
        idx, alpha, mse, nonz = SelectBestSubsetFromElasticNetPath(x, y,\
            Method='grid', MSE_threshold=None, R2_threshold=None, L1_ratio=self.L1, Eps=self.eps,\
            N_alphas=self.nAlphas, Alphas=self.alphas, max_iter=10000, tol=0.0001, cv=None, n_jobs=1, \
            selection='random', PlotPath=True, FileName=self.F_ENet, verbose=True)
        for i in VIP_idx: # add missing VIP features 
            if i not in idx:
                idx.append(i)
        self.idx = idx
        return
    
class BF:
    LastIterationsToStore = 15
    UseCorrelationMatrix = False
    MinCorr = None
    F_Xlsx = 'BF.xlsx'
    F_Plot = 'Results'
    Slope = 0.001
    VIP_number = None
    F_Coef = 'Coefficients.dat'
    
    def __init__(self, LastIterationsToStore=15, UseCorrelationMatrix=False,\
        MinCorr=None, F_Xlsx='BF.xlsx', F_Plot = 'Results', F_Coef = 'Coefficients.dat',\
        Slope = 0.001, VIP_number=None):
        self.LastIterationsToStore = LastIterationsToStore
        self.UseCorrelationMatrix = UseCorrelationMatrix
        self.MinCorr = MinCorr
        self.F_Xlsx = F_Xlsx
        self.F_Plot = F_Plot
        self.F_Coef = F_Coef
        self.Slope = Slope
        self.VIP_number = VIP_number
        return
    
    def fit(self, x_std, y_std, X_train, Y_train, X_test, Y_test, FeaturesAll,\
            FeaturesReduced, idx=None, VIP_idx=None, Method='sequential',\
            Criterion='MSE', GetVIP=False, BestFitMethod='Fast', MaxLoops=10,\
            MaxBottom=3, verbose=1):
        if VIP_idx is None:
            VIP_idx = []
        if idx is None:
            idx = []
            for i in range(0, x_std.shape[1], 1):
                idx.append(i)            
        if self.UseCorrelationMatrix:
            C = np.cov(x_std, rowvar=False, bias=True)
        writeResults = pd.ExcelWriter(self.F_Xlsx, engine='openpyxl')
        idx_list = []
        coef_list = []
        mse_test_list = []
        mse_train_list = []
        rmse_test_list = []
        rmse_train_list = []
        r2_test_list = []
        r2_train_list = []
        nonzero_list = []
        features_list = []
        if verbose > 0:
            print('Initial index set before backward elimination')
            print(idx)
            print('Start Backward sequential elimination')
        idx = rank_features(x_std, y_std, idx, direction='Hi-Lo') # most imprtant features first
        while (len(idx) > (0+len(VIP_idx))) and (len(idx) > 1):
            FeatureSize = len(idx)
            if FeatureSize > self.LastIterationsToStore:
                if verbose > 0:
                    print('Features left: ', FeatureSize)
                    idx = RemoveWorstFeature(x_std, y_std, Method='sequential',\
                        Criterion='MSE', idx=idx, VIP_idx=VIP_idx, sort=True, verbose=verbose)
                    print('Remaining set:', idx)
                    continue
            if self.UseCorrelationMatrix:
                idx_corr = ClassifyCorrelatedFeatures(x_std, idx,\
                    MinCorrelation=self.MinCorr, Model=1, Corr_Matrix=C, verbose=verbose)
            else:    
                idx_corr = get_full_list(idx, x_std.shape[1])
            if verbose > 0:
                print('Features left: ', len(idx))
            if BestFitMethod == 'Fast':
                idx_alternative = FindBestSet(x_std, y_std, idx, idx_corr, VIP_idx=VIP_idx, Method='MSE', verbose=verbose)
            else:
                idx_alternative = FindBestSetTree(x_std, y_std, idx, idx_corr,\
                    VIP_idx=VIP_idx, MaxLoops=MaxLoops, MaxBottom=MaxBottom, verbose=verbose)
            idx = rank_features(x_std, y_std, idx_alternative, direction='Hi-Lo') # most imprtant features first
            print('Best subset', idx)
            _, _, mse_train, rmse_train, r2_train = get_fit_score(X_train, Y_train, X_test, Y_test, idx=idx, Test=False)
            coef, nonzero, mse_test, rmse_test, r2_test = get_fit_score(X_train, Y_train, X_test, Y_test, idx=idx, Test=True)
            data_to_xls = []
            data_to_xls.append((coef, nonzero, mse_train, rmse_train, r2_train)) # score based on train set
            data_to_xls.append((coef, nonzero, mse_test, rmse_test, r2_test)) # score based on test set
            Results_to_xls(writeResults, str(FeatureSize), idx, FeaturesAll, FeaturesReduced,\
                data_to_xls)
            coef_list.append(coef)
            idx_list.append(idx)
            mse_test_list.append(mse_test)
            mse_train_list.append(mse_train)
            rmse_test_list.append(rmse_test)
            rmse_train_list.append(rmse_train)
            r2_test_list.append(r2_test)
            r2_train_list.append(r2_train)
            nonzero_list.append(nonzero)
            features_list.append(idx)
            idx = RemoveWorstFeature(x_std, y_std, Method='sequential',\
                Criterion='MSE', idx=idx, VIP_idx=VIP_idx, sort=True, verbose=verbose)
            print('Remaining set:', idx)
        writeResults.save()
        plot_rmse(self.F_Plot + '_Test', nonzero_list, rmse_test_list, r2_test_list) 
        plot_rmse(self.F_Plot + '_Train', nonzero_list, rmse_train_list, r2_train_list) 
        if GetVIP:
            size_list = len(nonzero_list)
            print('Before sorting')
            print('List of VIP features: ', VIP_idx)
            print('nonzero list:', nonzero_list)
            print('mse list:', mse_test_list)
            print('Full list', features_list )
            nonzero_list, mse_list, features_list = Sort(nonzero_list, mse_train_list, features_list, direction='Lo-Hi')
            for i in range(0, size_list-1, 1):
                mse = mse_list[i] # mse supposed to be greater than mse_next
                mse_next =  mse_list[i+1]
                delta = mse - mse_next # supposed to be positive
                fraction = delta / mse_next
                if (fraction < 0) or (abs(fraction) < self.Slope) or\
                    (nonzero_list[i] >= self.VIP_number):
                    VIP_idx = features_list[i]
                    if fraction < 0:
                        print('VIP selection criteria: Slope became positive')
                    if abs(fraction) < self.Slope:
                        print('VIP selection criteria: Slope less than ', self.Slope)
                    if nonzero_list[i] >= self.VIP_number:
                        print('VIP selection criteria: desired number of VIP features = ', self.VIP_number, 'has been reached')
                    print('After sorting')
                    print('number of VIP', nonzero_list[i])
                    print('List of VIP features: ', VIP_idx)
                    print('nonzero list:', nonzero_list)
                    print('mse list:', mse_list)
                    print('Full list', features_list )
                    
                    return VIP_idx
        else:
            coef_to_store = []
            for i in range(0, len(nonzero_list), 1):    
                coef_to_store.append((nonzero_list[i], idx_list[i], coef_list[i]))
# save coefficients list into file
            f = open(self.F_Coef, "wb")
            pickle.dump(coef_to_store, f)
            f.close() 
        return
    
def FindBestSetTree(x_std, y_std, active_features, corr_list, VIP_idx=None,\
    MaxLoops=10, MaxBottom=3, verbose=1):
    
    class Node():
        idx = None
        mse = None
        parent = None
        children = []
        level = None
        bottom = False
        finished = False
        Prev = None
        Next = None
        def __init__(self, idx, mse, parent=None):
            self.idx = idx
            self.mse = mse
            self.parent = parent
            self.children = []
            self.bottom = False
            self.finished = False
            for i in range(0, len(idx), 1):
                self.children.append(None)
            if self.parent is None:
                self.level = 0
            else:
                self.level = self.parent.level + 1
            return

    def add_child(idx, mse, node=None, child_number=None):
        child = Node(idx, mse, parent=node) # new node
        node.children[child_number] = child # link from parent to child
        return child

    def get_fit(x_std, y_std, active_features, corr_list, index, VIP_idx=None,\
                verbose=verbose):# local index
        if VIP_idx is None:
            VIP_idx = []
        features = copy.copy(active_features)
        global_old = features[index]
        size = len(features)
        Size = x_std.shape[0]
        x_sel = np.zeros(shape=(Size, size), dtype=float)
        tmp = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
        lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    # creating selected features array
        for i in range(0, size, 1):
            x_sel[:, i] = x_std[:, features[i]]
        lr.fit(x_sel, y_std)
        y_pred = lr.predict(x_sel)
        mse_initial = skm.mean_squared_error(y_std, y_pred)
        if global_old in VIP_idx:
            return features, mse_initial, False
        if verbose > 1:
            print('Initial MSE = ', mse_initial)
        replace_list = corr_list[index]
        size_current = len(replace_list)
        idx_array = np.zeros(shape=(size_current), dtype=int)
        mse_array = np.zeros(shape=(size_current), dtype=float)
        for i in range(0, size_current, 1): # along replace_list
            tmp[:, 0] = x_sel[:, index] # save selected column
            x_sel[:, index] = x_std[:, replace_list[i]]
            lr.fit(x_sel, y_std)
            y_pred = lr.predict(x_sel)
            mse = skm.mean_squared_error(y_std, y_pred)
            mse_array[i] = mse
            idx_array[i] = replace_list[i] # global index
            x_sel[:, index] = tmp[:, 0]
        idx = np.argmin(mse_array) # local index of best mse
        mse_new = mse_array[idx] # best mse for one feature in corr list
        global_new = idx_array[idx] # new feature 
        if verbose > 1:
            print('Final MSE = ', mse_new)
        if global_new != global_old: # found
            features[index] = global_new
            if verbose > 1:
                print('Found better fit')
                print('Initial feature ', global_old, 'replaced by ', global_new)
                print('Initial MSE = ', mse_initial, 'Final MSE = ', mse_new)
            return features, mse_new, True
        else:
            if verbose > 1:
                print('Did not find better fit')
            return features, mse_initial, False
   
    def print_nodes(root, verbose=1):
        if root is None:
            return
        node = root
        TotalNodes = 0
        TotalBottom = 0
        TotalFinished = 0
        TotalUnfinished = 0
        while True:
            if True:
                if verbose > 1:
                    print(node.idx, 'Finished ', node.finished, 'Bottom ', node.bottom, 'Level ', node.level)
            if node.Next is not None:
                node = node.Next
                TotalNodes += 1
                if node.finished:
                    TotalFinished += 1
                else:
                    TotalUnfinished += 1
                if node.bottom:
                    TotalBottom += 1
            else:
                break
        if verbose > 0:
            print('Total nodes = ', TotalNodes)
            print('Total bottom = ', TotalBottom)
            print('Total finished = ', TotalFinished)    
            print('Total unfinished = ', TotalUnfinished)
        return

    def get_best_mse(root):
        if root is None:
            return
        node = root
        mse = root.mse
        idx = root.idx
        while True:
            if node.mse < mse:
                mse = node.mse
                idx = node.idx
            if node.Next is not None:
                node = node.Next
            else:
                break
        return idx, mse

    idx, mse_initial, Found = get_fit(x_std, y_std, active_features, corr_list, 0, VIP_idx=active_features, verbose=verbose)
    node = Node(idx, mse_initial)
    Root = node
    # solves last feature. 
    # good to rank features from high to low importance
    repeat = False
    last_node = None
    k = 0
    bot = 0
    while (node is not None) and (k < MaxLoops) and (bot < MaxBottom):
        bottom = True
        if repeat:
            prev_node = last_node
            repeat = False
        else:
            prev_node = node
        new_node = None
        for i in range(0, len(idx), 1):
            idx = copy.copy(node.idx)
            found = False
            idx, mse, found = get_fit(x_std, y_std, idx, corr_list, i, VIP_idx=VIP_idx, verbose=verbose)
            if found:
                new_node = add_child(idx, mse, node, i)
                new_node.Prev = prev_node
                prev_node.Next = new_node
                prev_node = new_node
                bottom = False
        node.bottom = bottom
        node.finished = True
        if new_node is not None:
            node = new_node   
        else:
            repeat = True
            current_node = Root
            node = None
            last_node = prev_node
            best_mse = 1e100
            best_unfinished = None
            bot = 0
            while (node is None) :
                if current_node.bottom:
                    bot += 1
                if current_node.Next is None:
                    break # end of list
                if not current_node.finished: # if note is unfinished
                    if current_node.mse < best_mse:
                        best_mse = current_node.mse # get best msi
                        best_unfinished = current_node # get best unfinished node
                current_node = current_node.Next
            if best_unfinished is not None:
                node = best_unfinished
        k += 1
    print_nodes(Root, verbose=verbose) # service function
    idx, mse = get_best_mse(Root)
    return idx

# first list in *args governs sorting, others depend on first list 
def Sort(*args, direction='Lo-Hi'):
    idx = copy.deepcopy(args)
    n = len(idx[0])
    if direction == 'Lo-Hi':
        Dir = 1
    else:
        Dir = -1
# bubble sort according to direction
    while n != 0:
        newn = 0
        for i in range(1, n, 1):
            difference = (idx[0][i-1] - idx[0][i])*Dir
            if difference > 0:
                newn = i
                for j in range(0, len(idx), 1):
                    temp = idx[j][i-1]
                    idx[j][i-1] = idx[j][i] 
                    idx[j][i] = temp
        n = newn
    return idx
