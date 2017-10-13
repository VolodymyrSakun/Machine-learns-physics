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
# FindBestSetTree
# Swap
# Sort

import numpy as np
import pandas as pd
import sklearn.metrics as skm
import os
import copy
import pickle 
import sys
import re
import time
import shutil
from sklearn.linear_model import LinearRegression
import statsmodels.regression.linear_model as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
from joblib import Parallel, delayed
from project1 import IOfunctions
import random
from project1 import structure
from project1 import library
from project1 import spherical
from project1.genetic import GA
from project1 import regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel


def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False
    
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

def InInterval(n, List):
    if List is None:
        return -1
    for i in range(0, len(List), 1):
        if (n >= List[i][0]) and (n <= List[i][1]):
            return i
    return -1 # not in region

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

def ClassifyCorrelatedFeatures(X, features_idx=None, MinCorrelation=0.95, Model=1, C=None):
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
    
    if C is None:
        C = np.cov(X, rowvar=False, bias=True)
    list1 = []
    list2 = []
    if features_idx is None:
        features_idx = list(range(0, X.shape[1], 1))
    for i in range(0, len(features_idx), 1):
        idx = features_idx[i]
        list1.append(list(''))
        list2.append(list(''))
        list1[i].append(idx)
        list2[i].append(idx)
    NofFeatures = X.shape[1]
    k = 0
    for i in features_idx:
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

def StoreLinearFeaturesDescriprion(FileName, FeaturesAll, FeaturesReduced):
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

def StoreNonlinearFeaturesDescriprion(FileName, FeaturesNonlinear):
    writeResults = pd.ExcelWriter(FileName, engine='openpyxl')
    nFeatures = len(FeaturesNonlinear)
    Results = pd.DataFrame(np.zeros(shape = (nFeatures, 7)).astype(float), \
        columns=['Feature index','Feature type','Bond','Intermolecular','Distance type',\
        'Number of distances in feature', 'Number of constants in feature'], dtype=str)
    for i in range(0, nFeatures, 1):
        Results.loc[i]['Feature index'] = i
        Results.loc[i]['Feature type'] = FeaturesNonlinear[i].FeType
        Results.loc[i]['Bond'] = FeaturesNonlinear[i].Distance.Atom1.Symbol +\
        '-' + FeaturesNonlinear[i].Distance.Atom2.Symbol
        Results.loc[i]['Distance type'] = FeaturesNonlinear[i].Distance.DiType
        if FeaturesNonlinear[i].Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
            Results.loc[i]['Intermolecular'] = 'Yes'
        else:
            Results.loc[i]['Intermolecular'] = 'No'
        Results.loc[i]['Number of distances in feature'] = FeaturesNonlinear[i].nDistances 
        Results.loc[i]['Number of constants in feature'] = FeaturesNonlinear[i].nConstants 
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

def AddBestFeature(x_std, y_std, idx=None):
    size = x_std.shape[1] # number of features
    Size = x_std.shape[0] # number of records
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    if (idx is None) or (len(idx) == 0): # determine best feature if initial fit does not exist
        x_selected = np.zeros(shape=(Size, 1), dtype=float) # matrix with selected features
        mse_array = np.zeros(shape=(size), dtype=float)
        for i in range(0, size, 1):
            x_selected[:, 0] = x_std[:, i]
            lr.fit(x_selected, y_std)
            y_pred = lr.predict(x_selected)
            mse = skm.mean_squared_error(y_std, y_pred)
            mse_array[i] = mse
        idx = [] #indices of selected features
        idx.append(np.argmin(mse_array)) # get most important feature with smallest MSE
        return idx
    x_selected = np.zeros(shape=(Size, len(idx)+1), dtype=float) # array with selected features
    for i in range(0, len(idx), 1):
        x_selected[:, i] = x_std[:, idx[i]]
    mse_array = np.ones(shape=(size), dtype=float)
    for i in range(0, size, 1):
        if i in idx:
            mse_array[i] = 1e+100
            continue
        x_selected[:, -1] = x_std[:, i]
        lr.fit(x_selected, y_std)
        y_pred = lr.predict(x_selected)
        mse = skm.mean_squared_error(y_std, y_pred)
        mse_array[i] = mse
    idx.append(np.argmin(mse_array)) # index of feature with smallest MSE
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
    if size <= 1:
        return idx
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
    
class BF:
    LastIterationsToStore = 15
    UseCorrelationMatrix = False
    MinCorr = None
    F_Xlsx = 'BF.xlsx'
    F_Plot = 'Results'
    Slope = 0.001
    VIP_number = None
    F_Coef = 'Coefficients.dat'
    UseNonlinear = False
    Normalize = False
    
    def __init__(self, LastIterationsToStore=15, UseNonlinear=False, UseCorrelationMatrix=False,\
        MinCorr=None, F_Xlsx='BF.xlsx', F_Plot = 'Results', F_Coef = 'Coefficients.dat',\
        Slope = 0.001, VIP_number=None, Normalize=False):
        self.LastIterationsToStore = LastIterationsToStore
        self.UseCorrelationMatrix = UseCorrelationMatrix
        self.MinCorr = MinCorr
        self.F_Xlsx = F_Xlsx
        self.F_Plot = F_Plot
        self.F_Coef = F_Coef
        self.Slope = Slope
        self.VIP_number = VIP_number
        self.UseNonlinear = UseNonlinear
        self.Normalize = Normalize
        return
    
    def fit(self, X_train_nonlin, X_train_lin, Y_train, X_test_nonlin, X_test_lin,\
            Y_test, FeaturesAll, FeaturesReduced, idx_nonlin=None, VIP_idx_nonlin=None,\
            idx_lin=None, VIP_idx_lin=None, Method='sequential', Criterion='MSE',\
            GetVIP=False, BestFitMethod='Fast', MaxLoops=10, MaxBottom=3, verbose=1):
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

        def Print(self):
            print('idx: ', self.idx)
            print('MSE: ', self.mse)
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
        if k % 10 == 0:
            print('Iteration: ', k)
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
            print_nodes(Root, verbose=verbose) # bottom reached
            if verbose and (node is not None):
                node.Print()
        k += 1
    print_nodes(Root, verbose=verbose) # service function
    idx, mse = get_best_mse(Root)
    return idx

def Swap(a, b):
    tmp = a
    a = b
    b = tmp
    return a, b

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
                    idx[j][i], idx[j][i-1] = Swap(idx[j][i], idx[j][i-1])
        n = newn
    return idx

def get_energy(Coef_tuple, FeaturesAll, FeaturesReduced, record_list, Record, nVariables):
    tuple_number = 0
    Found = False
    for i in range(0, len(Coef_tuple), 1):
        if Coef_tuple[i][0] == nVariables:
            tuple_number = i   
            Found = True
    if not Found:
        print('Reord with ', nVariables, ' variables does not exist')
        return
    features_set = Coef_tuple[tuple_number]
    k = Record # number of record
    E = 0
    for i in range(0, len(features_set[1]), 1): # for each nonzero coefficient
        current_feature_idx = features_set[1][i]
        current_feature_type = FeaturesReduced[current_feature_idx].FeType
        variable = 0
        for j in range(0, len(FeaturesAll), 1): # for each combined feature
            if FeaturesAll[j].FeType == current_feature_type:
                if FeaturesAll[j].nDistances == 1:
                    atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                    atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                    d = np.sqrt((record_list[k].atoms[atom1_index].x - record_list[k].atoms[atom2_index].x)**2 +\
                                (record_list[k].atoms[atom1_index].y - record_list[k].atoms[atom2_index].y)**2 +\
                                (record_list[k].atoms[atom1_index].z - record_list[k].atoms[atom2_index].z)**2)            
                    r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
                if FeaturesAll[j].nDistances == 2:
                    atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                    atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                    atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                    atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                    d1 = np.sqrt((record_list[k].atoms[atom11_index].x - record_list[k].atoms[atom12_index].x)**2 +\
                                 (record_list[k].atoms[atom11_index].y - record_list[k].atoms[atom12_index].y)**2 +\
                                 (record_list[k].atoms[atom11_index].z - record_list[k].atoms[atom12_index].z)**2)            
                    r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                    d2 = np.sqrt((record_list[k].atoms[atom21_index].x - record_list[k].atoms[atom22_index].x)**2 +\
                                 (record_list[k].atoms[atom21_index].y - record_list[k].atoms[atom22_index].y)**2 +\
                                 (record_list[k].atoms[atom21_index].z - record_list[k].atoms[atom22_index].z)**2)            
                    r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power
                    r = r1 * r2      
                variable += r
        E += variable * features_set[2][i] # combined features by coefficient
    return E

def replace_numbers(l1, l2, source_data):
    l1 = list(l1)
    l2 = list(l2) # those numbers will be replaced by numbers from l1
    if type(source_data) is not list: # replace single number
        try:
            idx = l2.index(source_data)
        except:
            return False # source data is not in mapping list
        dest = l1[idx]       
        return dest
    dest_data = copy.deepcopy(source_data)
    for i in range(0, len(source_data), 1):
        if (type(source_data[i]) is list): # 2D array
            for j in range(0, len(source_data[i]), 1):
                num_source = source_data[i][j]
                try:
                    idx = l2.index(num_source)
                except:
                    return False # source data is not in mapping list
                num_dest = l1[idx]
                dest_data[i][j] = num_dest
        else: # 1D array
            try:
                idx = l2.index(source_data[i])
            except:
                return False # source data is not in mapping list
            num_dest = l1[idx]
            dest_data[i] = num_dest
    return dest_data
    
def FilterData(F, TrainIntervals, GridStart = 0.0, GridEnd = 20.0, GridSpacing=0.1, ConfidenceInterval=1, TestFraction=0.3, TrainFraction=1):
    F_MoleculesDescriptor = 'MoleculesDescriptor.'
    F_Train = 'Training Set.x'
    F_Test = 'Test Set.x'
    F_Filter = 'Filter.dat'
    MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=F_MoleculesDescriptor)
    RandomSeed = 101
    if RandomSeed is not None:
        random.seed(RandomSeed)
    else:
        random.seed()    
    GridTrain = [] # small intervals
    GridTest = [] # small intervals
    i = GridStart
    while i < (GridEnd-GridSpacing):
        GridTrain.append((round(i, 2), round(i+GridSpacing, 2)))
        GridTest.append((round(i, 2), round(i+GridSpacing, 2)))
        i += GridSpacing
    N = np.zeros(shape=(len(GridTest)), dtype=int) # number of points in each grid  
    NTrain = list(np.zeros(shape=(len(GridTrain)), dtype=int)) # count test points
    NTest = list(np.zeros(shape=(len(GridTest)), dtype=int)) # count test points        
    Records = IOfunctions.ReadRecords(F, MoleculePrototypes) # Read records
    DMin = 1000
    DMax = 0
    nMolecules = Records[0].nMolecules
# Determine DMin, DMax and fill N    
    for record in Records:
        if nMolecules == 2:
            if record.R_Average > DMax:
                DMax = record.R_Average
            if record.R_Average < DMin:
                DMin = record.R_Average
            j = int(record.R_Average / GridSpacing)
            N[j] += 1
        else:
            if record.R_CenterOfMass_Average > DMax:
                DMax = record.R_CenterOfMass_Average
            if record.R_CenterOfMass_Average < DMin:
                DMin = record.R_CenterOfMass_Average
            j = int(record.R_Average / GridSpacing)
            N[j] += 1
# Estimate number of points per grid
    n = np.asarray(N.nonzero()).reshape(-1) # indices with nonzero records nonzero   
    nGrids = int(len(n) * ConfidenceInterval)
    N_list = list(N)
    N_Reduced = []
    while len(N_Reduced) < nGrids:
        i = np.argmax(N_list)
        N_Reduced.append(N_list[i])
        del(N_list[i])
    nPointsGrid = N_Reduced[-1]
    nTestPointsGrid = int(nPointsGrid * TestFraction)
    nTotalTrainPointsGrid = nPointsGrid - nTestPointsGrid  
    nTrainPointsGrid = int(nTotalTrainPointsGrid * TrainFraction)
    N_list = list(N)    
    i = 0
    while i < len(N_list): # remove regions where there are not enough points
# training region        
        if (InInterval(GridTest[i][0], TrainIntervals) != -1) and (InInterval(GridTest[i][1], TrainIntervals) != -1):
            if N_list[i] < nPointsGrid: # not enough points for training and test, discard
                del(N_list[i])
                del(NTrain[i])
                del(NTest[i])
                del(GridTrain[i])
                del(GridTest[i])   
            else:
                i += 1
        else: # test region            
            if N_list[i] < nTestPointsGrid: # not enough test points
                del(N_list[i])
                del(NTrain[i])
                del(NTest[i])
                del(GridTrain[i])
                del(GridTest[i]) 
            else:
                i += 1
    i = 0 # remove remaining train grid that not in training region
    while i < len(GridTrain):
        if (InInterval(GridTrain[i][0], TrainIntervals) != -1) and (InInterval(GridTrain[i][1], TrainIntervals) != -1):
            i += 1
            continue
        else:
            del(GridTrain[i])
            del(NTrain[i])
# proceed records                               
    RecordsTrain = []
    RecordsTest = []
    while len(Records) > 0:
        r = random.randrange(0, len(Records), 1)  
        record = copy.deepcopy(Records[r])
        if nMolecules == 2:
            d = record.R_Average
        else:
            d = record.R_CenterOfMass_Average
        j = InInterval(d, GridTrain) 
        if j != -1: # in training region?
            if NTrain[j] < nTrainPointsGrid: # append to training set
                NTrain[j] += 1
                RecordsTrain.append(record)
            else:  # if it is full, append to test set
                j = InInterval(d, GridTest) # which interval?
                if NTest[j] < nTestPointsGrid: 
                    NTest[j] += 1
                    RecordsTest.append(record)              
        else: # not training region
            j = InInterval(d, GridTest) # which interval?
            if NTest[j] < nTestPointsGrid: # append to test set only
                NTest[j] += 1
                RecordsTest.append(record)
        del(Records[r]) 

    IOfunctions.store_records(F_Train, RecordsTrain) # store trained set
    IOfunctions.store_records(F_Test, RecordsTest) # store test set
    TestIntervals = [] # Define test regions
    if TrainIntervals[0][0] != 0:
        TestIntervals.append((0, TrainIntervals[0][0]))
    for i in range(0, (len(TrainIntervals)-1), 1):
        if TrainIntervals[i][1] != TrainIntervals[i+1][0]:
            TestIntervals.append((TrainIntervals[i][1], TrainIntervals[i+1][0]))
    if TrainIntervals[-1][1] != GridTest[-1][1]:
        TestIntervals.append((TrainIntervals[-1][1], GridTest[-1][1]))    
    results = {'Initial dataset': F, 'Number of molecules per record': nMolecules,\
               'Train Intervals': TrainIntervals, 'Test Intervals': TestIntervals, 'Train records number': len(RecordsTrain),\
               'Train Grid': GridTrain, 'Test Grid': GridTest, 'Test records number': len(RecordsTest),\
               'Molecule prototypes': MoleculePrototypes, 'Max points per grid': nPointsGrid,\
               'Train points per grid': nTrainPointsGrid, 'Train Fraction Used': TrainFraction,\
               'Test points per grid': nTestPointsGrid, 'Confidence Interval used': ConfidenceInterval,\
               'Training Set': F_Train, 'Test Set': F_Test}
    
# save results
    f = open(F_Filter, "wb")
    pickle.dump(results, f)
    f.close()   
# store results in txt file    
    l = []
    a = results.keys()
    for i in a:
        s = str(i) + "\n"
        s1 = str(results[i]) + "\n"
        l.append(s)
        l.append(s1)
        l.append("\n")
    f = open('results.txt', "w")
    f.writelines(l)
    f.close()
    
    return results
    
def ReadData(F_data, Atoms):
        # Read coordinates from file
    f = open(F_data, "r")
    data0 = f.readlines()
    f.close()
    data = []
    for i in range(0, len(data0), 1):
        data.append(data0[i].rstrip())
    del(data0)
    # Rearrange data in structure
    i = 0 # counts lines in textdata
    j = 0 # counts atom records for each energy value
    atoms_list = [] # temporary list
    record_list = []
    while i < len(data):
        s = data[i].split() # line of text separated in list
        if len(s) == 0: # empty line
            i += 1
            continue
    # record for energy value
        elif (len(s) == 1) and library.isfloat(s[0]): 
            e = float(s[0])
            rec = structure.Record(e, atoms_list)
            record_list.append(rec)
            j = 0
            atoms_list = []
        elif (len(s) == 4): 
            x = float(s[1])
            y = float(s[2])
            z = float(s[3])
            atoms_list.append(structure.AtomCoordinates(Atoms[j], x, y, z))
            j += 1
        i += 1
    return record_list

def StoreEnergy(F_Response, record_list):
    Size = len(record_list)
    if Size > 0:
        energy = np.zeros(shape=(Size, 1), dtype=float)
        for i in range(0, Size, 1):
            energy[i, 0] = record_list[i].e # energy    
        Table = pd.DataFrame(energy, columns=['response'], dtype=float)
        f = open(F_Response, 'w')
        Table.to_csv(f, index=False)
        f.close()
    return

def StoreDistances(F_Distances, record_list, Distances):
    Size = len(record_list)
    nDistances = len(Distances)
    distances_array = np.zeros(shape=(Size, nDistances), dtype=float)
    for i in range(0, Size, 1):
        for j in range(0, nDistances, 1):
            r = np.sqrt((record_list[i].atoms[Distances[j].Atom1.Index].x - record_list[i].atoms[Distances[j].Atom2.Index].x)**2 +\
                        (record_list[i].atoms[Distances[j].Atom1.Index].y - record_list[i].atoms[Distances[j].Atom2.Index].y)**2 +\
                        (record_list[i].atoms[Distances[j].Atom1.Index].z - record_list[i].atoms[Distances[j].Atom2.Index].z)**2)            
            distances_array[i, j] = r
# save distances
    Table = pd.DataFrame(distances_array, dtype=float)
    f = open(F_Distances, 'w')
    Table.to_csv(f, index=False)
    f.close()
    return

def StoreFeatures(F_LinearFeatures, first, last, FeaturesAll, FeaturesReduced, record_list, Atoms):
# Storing energy
    NofFeaturesReduced = len(FeaturesReduced)
    NofFeatures = len(FeaturesAll)
# calculating and storing distances  
    features_array = np.zeros(shape=(last-first, len(FeaturesAll)), dtype=float) 
    for j in range(0, len(FeaturesAll), 1):
        for i in range(first, last, 1):
            if (FeaturesAll[j].nDistances == 1) and (FeaturesAll[j].Harmonic1 is None) and (FeaturesAll[j].Harmonic2 is None):
# features with only one distance. no harmonics
                atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                d = np.sqrt((record_list[i].atoms[atom1_index].x - record_list[i].atoms[atom2_index].x)**2 +\
                            (record_list[i].atoms[atom1_index].y - record_list[i].atoms[atom2_index].y)**2 +\
                            (record_list[i].atoms[atom1_index].z - record_list[i].atoms[atom2_index].z)**2)            
                r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
            if (FeaturesAll[j].nDistances == 2) and (FeaturesAll[j].Harmonic1 is None) and (FeaturesAll[j].Harmonic2 is None):
# features with two distances without harmonics
                atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                d1 = np.sqrt((record_list[i].atoms[atom11_index].x - record_list[i].atoms[atom12_index].x)**2 +\
                             (record_list[i].atoms[atom11_index].y - record_list[i].atoms[atom12_index].y)**2 +\
                             (record_list[i].atoms[atom11_index].z - record_list[i].atoms[atom12_index].z)**2)            
                r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                d2 = np.sqrt((record_list[i].atoms[atom21_index].x - record_list[i].atoms[atom22_index].x)**2 +\
                             (record_list[i].atoms[atom21_index].y - record_list[i].atoms[atom22_index].y)**2 +\
                             (record_list[i].atoms[atom21_index].z - record_list[i].atoms[atom22_index].z)**2)            
                r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power
                r = r1 * r2       
            if (FeaturesAll[j].nDistances == 2) and (FeaturesAll[j].Harmonic1 is not None) and (FeaturesAll[j].Harmonic2 is not None):
# features with two distances and two harmonics            
                atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                d1 = np.sqrt((record_list[i].atoms[atom11_index].x - record_list[i].atoms[atom12_index].x)**2 +\
                             (record_list[i].atoms[atom11_index].y - record_list[i].atoms[atom12_index].y)**2 +\
                             (record_list[i].atoms[atom11_index].z - record_list[i].atoms[atom12_index].z)**2)            
                r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                d2 = np.sqrt((record_list[i].atoms[atom21_index].x - record_list[i].atoms[atom22_index].x)**2 +\
                             (record_list[i].atoms[atom21_index].y - record_list[i].atoms[atom22_index].y)**2 +\
                             (record_list[i].atoms[atom21_index].z - record_list[i].atoms[atom22_index].z)**2)            
                r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power

                center_index = FeaturesAll[j].Harmonic1.Center.Index
                external_index = FeaturesAll[j].Harmonic1.Atom.Index
                new_origin = spherical.Point(record_list[i].atoms[center_index].x, record_list[i].atoms[center_index].y, record_list[i].atoms[center_index].z)
                external_atom = spherical.Point(record_list[i].atoms[external_index].x, record_list[i].atoms[external_index].y, record_list[i].atoms[external_index].z)
                h_list = []
                O_list = []
                for k in range(0, len(Atoms), 1): 
# find two hydrogens that belong to same molecule
                    if (FeaturesAll[j].Harmonic1.Center.MolecularIndex == Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic1.Center.AtType != Atoms[k].AtType):
                        h_list.append(Atoms[k])
# find two oxygens that belong to other molecules                            
                    if (FeaturesAll[j].Harmonic1.Center.MolecularIndex != Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic1.Center.AtType == Atoms[k].AtType):
                        O_list.append(Atoms[k])
                H1_index = h_list[0].Index
                H2_index = h_list[1].Index
                H1 = spherical.Point(record_list[i].atoms[H1_index].x, record_list[i].atoms[H1_index].y, record_list[i].atoms[H1_index].z)
                H2 = spherical.Point(record_list[i].atoms[H2_index].x, record_list[i].atoms[H2_index].y, record_list[i].atoms[H2_index].z)
                if len(O_list) == 1: # two water molecules system
                    O2_index = O_list[0].Index
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    directing_point = O2
                else:
                    O2_index = O_list[0].Index
                    O3_index = O_list[1].Index                    
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    O3 = spherical.Point(record_list[i].atoms[O3_index].x, record_list[i].atoms[O3_index].y, record_list[i].atoms[O3_index].z)
                    directing_point = spherical.get_directing_point_O2_O3(O2, O3)
                theta, phi = spherical.get_angles(new_origin, H1, H2, external_atom, directing_point)
                s1 = spherical.get_real_form2(FeaturesAll[j].Harmonic1.Order, FeaturesAll[j].Harmonic1.Degree, theta, phi)

                center_index = FeaturesAll[j].Harmonic2.Center.Index
                external_index = FeaturesAll[j].Harmonic2.Atom.Index
                new_origin = spherical.Point(record_list[i].atoms[center_index].x, record_list[i].atoms[center_index].y, record_list[i].atoms[center_index].z)
                external_atom = spherical.Point(record_list[i].atoms[external_index].x, record_list[i].atoms[external_index].y, record_list[i].atoms[external_index].z)
                h_list = []
                O_list = []
                for k in range(0, len(Atoms), 1): 
# find two hydrogens that belong to same molecule
                    if (FeaturesAll[j].Harmonic2.Center.MolecularIndex == Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic2.Center.AtType != Atoms[k].AtType):
                        h_list.append(Atoms[k])
# find two oxygens that belong to other molecules                            
                    if (FeaturesAll[j].Harmonic2.Center.MolecularIndex != Atoms[k].MolecularIndex) and\
                        (FeaturesAll[j].Harmonic2.Center.AtType == Atoms[k].AtType):
                        O_list.append(Atoms[k])
                H1_index = h_list[0].Index
                H2_index = h_list[1].Index
                H1 = spherical.Point(record_list[i].atoms[H1_index].x, record_list[i].atoms[H1_index].y, record_list[i].atoms[H1_index].z)
                H2 = spherical.Point(record_list[i].atoms[H2_index].x, record_list[i].atoms[H2_index].y, record_list[i].atoms[H2_index].z)
                if len(O_list) == 1: # two water molecules system
                    O2_index = O_list[0].Index
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    directing_point = O2
                else:
                    O2_index = O_list[0].Index
                    O3_index = O_list[1].Index                    
                    O2 = spherical.Point(record_list[i].atoms[O2_index].x, record_list[i].atoms[O2_index].y, record_list[i].atoms[O2_index].z)
                    O3 = spherical.Point(record_list[i].atoms[O3_index].x, record_list[i].atoms[O3_index].y, record_list[i].atoms[O3_index].z)
                    directing_point = spherical.get_directing_point_O2_O3(O2, O3)                    
                theta, phi = spherical.get_angles(new_origin, H1, H2, external_atom, directing_point)
                s2 = spherical.get_real_form2(FeaturesAll[j].Harmonic2.Order, FeaturesAll[j].Harmonic2.Degree, theta, phi)

                r = r1*r2*s1*s2
                
            features_array[i-first, j] = r # store to array
# sum features with equal FeType
    features_array_reduced = np.zeros(shape=(last-first, NofFeaturesReduced), dtype=float)
    for k in range(0, NofFeaturesReduced, 1):
        for j in range(0, NofFeatures, 1):
            if (FeaturesAll[j].FeType == FeaturesReduced[k].FeType):
                features_array_reduced[:, k] += features_array[:, j]


# removing NaN from dataset
#    mask = ~np.any(np.isnan(features_array_reduced), axis=1)
#    features_array_reduced = features_array_reduced[mask]
#    energy = energy[mask]
# save reduced features and energy into file
    Table = pd.DataFrame(features_array_reduced, dtype=float)
    f = open(F_LinearFeatures, 'a')
    if first == 0:
        Table.to_csv(f, index=False)
    else:
        Table.to_csv(f, index=False, header=False)
    f.close()
    return
# end of StoreFeatures

def GenerateFeatures():
    F_SystemDescriptor = 'SystemDescriptor.' # file with info about system structure
    F_Response_Train = 'ResponseTrain.csv' # response variable (y)
    F_Response_Test = 'ResponseTest.csv' # response variable (y)
    F_LinearFeaturesTrain = 'LinearFeaturesTrain.csv' # output csv file with combined features and energy
    F_LinearFeaturesTest = 'LinearFeaturesTest.csv' # output csv file with combined features and energy
    F_Distances_Train = 'Distances Train.csv' # output csv file. distances
    F_Distances_Test = 'Distances Test.csv' # output csv file. distances
    F_NonlinearFeatures = 'NonlinearFeatures.dat'
    F_LinearFeaturesAll = 'LinearFeaturesAll.dat' # output data structure which contains all features
    F_LinearFeaturesReduced = 'LinearFeaturesReduced.dat' # output data structure which contains combined features
    F_System = 'system.dat' # output data system structure
    F_record_list = 'records.dat'
    F_LinearFeaturesList = 'Linear Features Reduced List.xlsx'
    F_NonlinearFeaturesList = 'Nonlinear Features List.xlsx'
    F_Structure = 'Structure.xlsx'
    F_Filter = 'Filter.dat'
    Separators = '=|,| |:|;|: '   
    RandomSeed = 101
    if RandomSeed is not None:
        random.seed(RandomSeed)
    else:
        random.seed()
    try:        
        os.remove(F_Response_Train)
        os.remove(F_Response_Test)
        os.remove(F_LinearFeaturesTrain) # erase old files if exist
        os.remove(F_LinearFeaturesTest)
        os.remove(F_Distances_Train)
        os.remove(F_Distances_Test)
        os.remove(F_NonlinearFeatures)
        os.remove(F_LinearFeaturesAll) 
        os.remove(F_LinearFeaturesReduced) 
        os.remove(F_System)
        os.remove(F_record_list)
        os.remove(F_LinearFeaturesList)
        os.remove(F_Structure)
    except:
        pass    
    f = open(F_Filter, "rb")
    Filter = pickle.load(f)
    f.close()
    F_train_data = Filter['Training Set']
    F_test_data = Filter['Test Set']
    nTrainPoints = Filter['Train records number']
    nTestPoints = Filter['Test records number']
    # read descriptor from file
    with open(F_SystemDescriptor) as f:
        lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines] # x is string
    ProceedSingle = False
    ProceedDouble = False
    for i in range(0, len(lines), 1):
        x = lines[i]
        if len(x) == 0:
            continue
        if x[0] == '#':
            continue
        if (x.find('SingleDistancesInclude') != -1):
            s = re.split(Separators, x)
            s = list(filter(bool, s)) # removes empty records
            if 'True' in s: # proceed single distances
                ProceedSingle = True
        if (x.find('DoubleDistancesInclude') != -1):
            s = re.split(Separators, x)
            s = list(filter(bool, s)) # removes empty records
            if 'True' in s: # proceed single distances
                ProceedDouble = True
    if ProceedSingle:
        if ('&SingleDistancesDescription' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&SingleDistancesDescription') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endSingleDistancesDescription') != -1):
                    Last = i # index +1 of last record of single distance description
            SingleDescription = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                SingleDescription.append(s)
            del(First)
            del(Last)
        if ('&DefaultSingleDistances' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('SinglePowers') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    SinglePowersDefault = list(filter(bool, s)) # removes empty records
                    del(SinglePowersDefault[0])
    if ProceedDouble:
        if ('&DoubleDistancesDescription' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&DoubleDistancesDescription') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endDoubleDistancesDescription') != -1):
                    Last = i # index +1 of last record of single distance description
            DoubleDescription = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                DoubleDescription.append(s)
            del(First)
            del(Last)
        if ('&DefaultDoubleDistances' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('DoublePowers') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))
                    DoublePowersDefault = list(filter(bool, s)) # removes empty records
                    del(DoublePowersDefault[0])
                if (x.find('IncludeAllExcept') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        IncludeAllExcept = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        IncludeAllExcept = False
                if (x.find('ExcludeAllExcept') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        ExcludeAllExcept = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        ExcludeAllExcept = False 
                if (x.find('IncludeSameType') != -1):
                    s = re.split(Separators, x)
                    s = list(filter(bool, s))  
                    if (s[1] == 'True') or (s[1] == 'Yes'):
                        IncludeSameType = True
                    if (s[1] == 'False') or (s[1] == 'No'):
                        IncludeSameType = False                    
        if ('&IncludeExcludeList' in lines):
            for i in range(0, len(lines), 1):
                x = lines[i]
                if (x.find('&IncludeExcludeList') != -1):
                    First = i + 1 # index of first record of single distance description
                if (x.find('&endIncludeExcludeList') != -1):
                    Last = i # index +1 of last record of single distance description
            IncludeExcludeList = []
            for i in range(First, Last, 1):
                x = lines[i]
                s = re.split(Separators, x)
                s = list(filter(bool, s)) # removes empty records
                IncludeExcludeList.append(s)
    else:
        DtP_Double_list = []
                    
    if ('&SYSTEM' in lines):
        i = 0 # index of line in file
        Atoms = [] # create list of atom structures from file
        while ((lines[i].find('&SYSTEM') == -1) and (i < len(lines))):
            i += 1
        i += 1 # next line after &SYSTEM
        j = 0 # order in the system 
        types_list = []
        idx_list = []
        molecules_idx_list = []
        molecules = []
        k = 0
        one_molecule_atoms = []
        old_molecule_index = None
        while ((lines[i].find('&endSYSTEM') == -1) and (i < len(lines))):
            if (lines[i][0] == '#'):
                i += 1
                continue
            x = lines[i]
            if (x.find('&Molecule') != -1):
                s = re.split(Separators, x)
                del(s[0]) # 'Molecule'
                s0 = s[0]
                for l in range(1, len(s), 1):
                    s0 = s0 + ' ' + s[l]
                MoleculeName = s0
                i += 1
                continue
            s = re.split(Separators, x)
            s = list(filter(bool, s))
            symbol = s[0]
            if symbol not in types_list:
                types_list.append(symbol)
                idx_list.append(k)
                k += 1
            idx = types_list.index(symbol)    
            try:
                molecule_index = int(s[1])
                if molecule_index not in molecules_idx_list: # next molecule
                    molecules_idx_list.append(molecule_index)
                    if j == 0: # first atom
                        old_molecule_index = molecule_index
                        one_molecule_atoms.append(structure.Atom(symbol, j, idx_list[idx], old_molecule_index))
                        j += 1 # to next index
                        i += 1 # to next line
                        continue
                    for l in one_molecule_atoms: # store atoms from previous molecule
                        Atoms.append(l)
                    molecules.append(structure.Molecule(one_molecule_atoms, Name=MoleculeName))
                    one_molecule_atoms = []
                    old_molecule_index = molecule_index
                    one_molecule_atoms.append(structure.Atom(symbol, j, idx_list[idx], old_molecule_index))
                    old_molecule_index = molecule_index
                    j += 1
                else: # same molecule
                    one_molecule_atoms.append(structure.Atom(symbol, j, idx_list[idx], old_molecule_index))
                    j += 1
            except:
                print('Error reading file')
                break
            i += 1
        nMolecules = len(molecules_idx_list)
        for l in one_molecule_atoms:
            Atoms.append(l)
        molecules.append(structure.Molecule(one_molecule_atoms, Name=MoleculeName))
    else:
        print('Error reading &SYSTEM section from SystemDescriptor')
                
    Prototypes = IOfunctions.ReadMoleculeDescription(F_SystemDescriptor) # read molecule prototypes 
    MoleculeNames = []
    for i in Prototypes:
        MoleculeNames.append(i.Name)
    for i in molecules:
        if i.Name in MoleculeNames:
            idx = MoleculeNames.index(i.Name)
            i.Mass = Prototypes[idx].Mass
            for j in range(0, len(i.Atoms), 1):
                i.Atoms[j].Atom.Mass = Prototypes[idx].Atoms[j].Atom.Mass
                i.Atoms[j].Atom.Radius = Prototypes[idx].Atoms[j].Atom.Radius
                i.Atoms[j].Atom.Bonds = Prototypes[idx].Atoms[j].Atom.Bonds
                i.Atoms[j].Atom.Bonds = library.replace_numbers(i.AtomIndex, Prototypes[idx].AtomIndex, i.Atoms[j].Atom.Bonds)
    for i in molecules:
        i._refresh() # update data from prototype
    # determine number of atom types from list
    nAtTypes = len(types_list)       
    nAtoms = len(Atoms)
    Distances = [] # create list of distances from list of atoms
    for i in range(0, nAtoms, 1):
        for j in range(i+1, nAtoms, 1):
            Distances.append(structure.Distance(Atoms[i], Atoms[j]))
    DiTypeList = []
    for i in Distances:
        if i.DiType not in DiTypeList:
            DiTypeList.append(i.DiType)
    nDistances = len(Distances)
    nDiTypes = len(DiTypeList)
    system = structure.System(Atoms=Atoms, Molecules=molecules, Prototypes=Prototypes,\
        nAtoms=nAtoms, nAtTypes=nAtTypes, nMolecules=nMolecules, Distances=Distances,\
        nDistances=nDistances, nDiTypes=nDiTypes)
    FeaturesNonlinear = []    
    for i in range(0, nDistances, 1):
        FeaturesNonlinear.append(structure.FeatureNonlinear(i, Distances[i], FeType='exp', nDistances=1, nConstants=2))
    FeaturesAll = [] # list of all features    
    if ProceedSingle:
        # add DiType for each record   
        SingleDescriptionDiType = []                 
        for i in range(0, len(SingleDescription), 1):
            a1 = SingleDescription[i][0]
            a2 = SingleDescription[i][1]
            if SingleDescription[i][2] == 'intermolecular':
                inter = True
            else:
                inter = False
            for j in Distances:
                if j.isIntermolecular == inter:
                    if ((j.Atom1.Symbol == a1) and (j.Atom2.Symbol == a2)) or \
                        ((j.Atom1.Symbol == a2) and (j.Atom2.Symbol == a1)):
                        SingleDescriptionDiType.append(j.DiType)
                        break
        for i in DiTypeList:
            if i not in SingleDescriptionDiType:
                SingleDescriptionDiType.append(i)
                for j in Distances:
                    if j.DiType == i:
                        idx = j
                        break
                if idx.isIntermolecular:
                    inter = 'intermolecular'
                else:
                    inter = 'intramolecular'
                SingleDescription.append([idx.Atom1.Symbol, idx.Atom2.Symbol, inter])
                SingleDescription[-1] += SinglePowersDefault

    # make list of features with only one distance
        DtP_Single_list = []
        for i in range(0, len(Distances), 1):
            k = SingleDescriptionDiType.index(Distances[i].DiType)
            Powers = SingleDescription[k][3:]
            for j in Powers:
                DtP_Single_list.append(structure.Distance_to_Power(Distances[i], int(j)))
    
        for i in DtP_Single_list:
            FeaturesAll.append(structure.Feature(i, DtP2=None))        
            
    if ProceedDouble:
        # add DiType for each record   
        DoubleDescriptionDiType = []                 
        for i in range(0, len(DoubleDescription), 1):
            a1 = DoubleDescription[i][0]
            a2 = DoubleDescription[i][1]
            if DoubleDescription[i][2] == 'intermolecular':
                inter = True
            else:
                inter = False
            for j in Distances:
                if j.isIntermolecular == inter:
                    if ((j.Atom1.Symbol == a1) and (j.Atom2.Symbol == a2)) or \
                        ((j.Atom1.Symbol == a2) and (j.Atom2.Symbol == a1)):
                        DoubleDescriptionDiType.append(j.DiType)
                        break
        for i in DiTypeList:
            if i not in DoubleDescriptionDiType:
                DoubleDescriptionDiType.append(i)
                for j in Distances:
                    if j.DiType == i:
                        idx = j
                        break
                if idx.isIntermolecular:
                    inter = 'intermolecular'
                else:
                    inter = 'intramolecular'
                DoubleDescription.append([idx.Atom1.Symbol, idx.Atom2.Symbol, inter])
                DoubleDescription[-1] += DoublePowersDefault

    # make list of features with only one distance
        DtP_Double_list = []
        for i in range(0, len(Distances), 1):
            k = DoubleDescriptionDiType.index(Distances[i].DiType)
            Powers = DoubleDescription[k][3:]
            for j in Powers:
                DtP_Double_list.append(structure.Distance_to_Power(Distances[i], int(j)))        
        IncludeExcludeDiTypes = [] # can be empty
        for i in IncludeExcludeList:
            a11 = i[0]
            a12 = i[1]
            a21 = i[3]
            a22 = i[4]
            if i[2] == 'intermolecular':
                inter1 = True
            else:
                inter1 = False
            if i[5] == 'intermolecular':
                inter2 = True
            else:
                inter2 = False
            for j in Distances:
                if j.isIntermolecular == inter1:
                    if ((j.Atom1.Symbol == a11) and (j.Atom2.Symbol == a12)) or \
                        ((j.Atom1.Symbol == a12) and (j.Atom2.Symbol == a11)):
                        Type1 = j.DiType
                        break
            for j in Distances:
                if j.isIntermolecular == inter2:
                    if ((j.Atom1.Symbol == a21) and (j.Atom2.Symbol == a22)) or \
                        ((j.Atom1.Symbol == a22) and (j.Atom2.Symbol == a21)):
                        Type2 = j.DiType
                        break
            IncludeExcludeDiTypes.append((Type1, Type2))

        for i in range(0, len(DtP_Double_list), 1):
            for j in range(i+1, len(DtP_Double_list), 1):
                if DtP_Double_list[i].Power > DtP_Double_list[j].Power:
                    continue # skip duplicates
                if not IncludeSameType:
                    if DtP_Double_list[i].Distance.DiType == DtP_Double_list[j].Distance.DiType: # skip if distances of the same type
                        continue
                if len(IncludeExcludeDiTypes) == 0:
                    FeaturesAll.append(structure.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j]))
                else:
                    for k in IncludeExcludeDiTypes:
                        if ExcludeAllExcept:
                            if (DtP_Double_list[i].Distance.DiType == k[0] and DtP_Double_list[j].Distance.DiType == k[1]) or (DtP_Double_list[i].Distance.DiType == k[1] and DtP_Double_list[j].Distance.DiType == k[0]):
                               FeaturesAll.append(structure.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j])) # append if match
                        if IncludeAllExcept:
                            if (DtP_Double_list[i].Distance.DiType == k[0] and DtP_Double_list[j].Distance.DiType == k[1]) or (DtP_Double_list[i].Distance.DiType == k[1] and DtP_Double_list[j].Distance.DiType == k[0]):
                                continue #skip if match
                            FeaturesAll.append(structure.Feature(DtP_Double_list[i], DtP2=DtP_Double_list[j]))

    # Make list of reduced features
    FeaturesReduced = []
    FeType_list = []
    for i in range(0, len(FeaturesAll), 1):
        if (FeaturesAll[i].FeType not in FeType_list):
            FeType_list.append(FeaturesAll[i].FeType)
            FeaturesReduced.append(FeaturesAll[i])
# store global indices for each reduced feature
    for k in range(0, len(FeaturesReduced), 1):
        for j in range(0, len(FeaturesAll), 1):
            if (FeaturesAll[j].FeType == FeaturesReduced[k].FeType):
                if j not in FeaturesReduced[k].idx:
                    FeaturesReduced[k].idx.append(j)
                    
    # save list FeaturesNonlinear into file
    f = open(F_NonlinearFeatures, "wb")
    pickle.dump(FeaturesNonlinear, f)
    f.close()
    
    # save list FeaturesAll into file
    f = open(F_LinearFeaturesAll, "wb")
    pickle.dump(FeaturesAll, f)
    f.close()
    
    # save list FeaturesReduced into file
    f = open(F_LinearFeaturesReduced, "wb")
    pickle.dump(FeaturesReduced, f)
    f.close()

    # save system object into file
    f = open(F_System, "wb")
    pickle.dump(system, f)
    f.close()
    
    library.StoreNonlinearFeaturesDescriprion(F_NonlinearFeaturesList, FeaturesNonlinear) # xlsx
    library.StoreLinearFeaturesDescriprion(F_LinearFeaturesList, FeaturesAll, FeaturesReduced) # xlsx
    library.store_structure(F_Structure, Atoms, Distances, DtP_Double_list, FeaturesAll) # xlsx
    record_list_train = ReadData(F_train_data, Atoms)
    record_list_test = ReadData(F_test_data, Atoms)
    record_list = record_list_train + record_list_test
    StoreDistances(F_Distances_Train, record_list_train, Distances)
    StoreDistances(F_Distances_Test, record_list_test, Distances)  
    StoreEnergy(F_Response_Train, record_list_train)
    StoreEnergy(F_Response_Test, record_list_test)
        
    print('Number of train points', nTrainPoints)
    print('Number of test points', nTestPoints)
    print('Total number of observations = ', len(record_list))

# split array if too big
    NpArrayCapacity = 1e+8
    Size = len(record_list) # N of observations
    Length = len(FeaturesAll)
    if (Size * Length) > NpArrayCapacity:
        BufSize = int(NpArrayCapacity / Length)
    else:
        BufSize = Size
    # create endpoints for array size_list[1 - inf][0 - 1]
    i = 0 # Number of observation
    j = 0 # Number of feature
    size_list = []
    size_list_str = []
    nCPU = mp.cpu_count()
    nCPU = 1
    print('Start Multiprocessing with ', nCPU, ' cores')
    size = int(Size / nCPU)
    first = 0
    if size < BufSize:
        for i in range(0, nCPU, 1):
            first = i * size
            last = (i+1) * size
            if i == (nCPU-1):
                last = Size
            size_list.append((first, last))
            size_list_str.append(str(first) + '-' + str(last-1) + '.csv')
    else:# if number of records is huge
        i = 0
        last = 0
        while last != Size:
            first = i * BufSize
            last = (i+1) * BufSize
            if last > Size:
                last = Size
            size_list.append((first, last))
            size_list_str.append(str(first) + '-' + str(last-1) + '.csv')
            i += 1

    ran = list(range(0, len(size_list_str), 1))
    jobs = (delayed(StoreFeatures)(size_list_str[i], size_list[i][0], size_list[i][1], FeaturesAll, FeaturesReduced, record_list, Atoms) for i in ran)
    Parallel(n_jobs=nCPU)(jobs)
    print('Storing results in one file')
    f = open('Tmp.csv', "w")
    for i in range(0, len(size_list_str), 1):
        fin = open(size_list_str[i], "r")
        S = fin.readlines()
        f.writelines(S)
        fin.close()
    f.close()
    f = open('Tmp.csv', "r")
    data = f.readlines()
    f.close()
    os.remove('Tmp.csv')
    f = open(F_LinearFeaturesTrain, "w")
    i = 0
    while i <= nTrainPoints:
        f.write(data[i])
        i += 1
    f.close()
    if i < len(data):
        f = open(F_LinearFeaturesTest, "w")
        f.write(data[0])
        while i < len(data):
            f.write(data[i])
            i += 1
        f.close()       
    for i in range(0, len(size_list_str), 1):
        try:
            os.remove(size_list_str[i]) # erase old files if exist
        except:
            pass
            
def GetFit():
# Global variables
    DesiredNumberVariables = 15
    FirstAlgorithm = 'GA' # specifies algorithm that will give rough initial fit. 
    # Can be 'ENet' or 'GA' 
    UseVIP = False # if true fit will be found in two steps. First step - fit only
    # single distances, select most important (VIP) features which will be kept
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
    MinCorrMutation=0.8
    UseCorrelationBestFit=False
    MinCorrBestFit=0.9
    
    # for Best Fit algorithm. If False, all features will be used in trials 
    # to find Best Fit. Overwise, only most correlated features will be used.
    # If false, will be slow for big number of features
    # needed only if UseCorrelationMatrix=True
    # used for creating list of most correlated features for Best Fit
    # float [0 .. 1]
    F_Filter = 'Filter.dat'
    F_Out_Single = 'Single BE.xlsx'
    F_ENet_Single = 'Single ENet path'
    F_Out = 'BE.xlsx'
    F_ENet = 'ENet path'
    F_GA = 'GA path'
    F_ga_structure = 'GA_structure.dat' 
    F_gp_structure = 'GP_structure.dat' 
    F_MoleculesDescriptor = 'MoleculesDescriptor.'
#    Slope = 0.001 # stopping criterion for VIP features. Slope of RMSE graph
    VIP_number = 5 # stopping criterion for VIP features. Desired number of VIP features
# GA parameters
    PopulationSize = 100 # Population
    ChromosomeSize = DesiredNumberVariables
    MutationProbability = 0.3 # probability of mutation
    MutationInterval = [1, 3] # will be randomly chosen between min and max-1
    EliteFraction = 0.4 # fracrion of good chromosomes that will be used for crossover
    MutationCrossoverFraction = 0.3
    CrossoverFractionInterval = [0.6, 0.4] # how many genes will be taken from first and second best chromosomes (fraction)
    IterationPerRecord = 50 # Display results of current fit after N iterations
    StopTime = 600 # How long in seconds GA works without improvement
    nIter = 500 # How many populations will be used before GA stops
    RandomSeed = 101
    LinearSolver = 'sklearn' # 'sklearn', 'scipy', 'statsmodels'
    cond=1e-03 # for scipy solver
    lapack_driver='gelsy' # 'gelsd', 'gelsy', 'gelss', for scipy solver
    NonlinearFunction = 'exp'
    UseNonlinear = False # if true, only linear regression will be used
    CrossoverMethod = 'Random' # 'Random' or 'Best'
    MutationMethod = 'Correlated' # 'Random' or 'Correlated'
    verbose = True
    
    FilterResults = IOfunctions.ReadFeatures(\
        F_Nonlinear_Train='Distances Train.csv', F_Nonlinear_Test='Distances Test.csv', F_linear_Train='LinearFeaturesTrain.csv',\
        F_Response_Train='ResponseTrain.csv', F_linear_Test=None,\
        F_Response_Test=None, F_NonlinearFeatures = 'NonlinearFeatures.dat',\
        F_FeaturesAll='LinearFeaturesAll.dat', F_FeaturesReduced='LinearFeaturesReduced.dat',\
        F_System='system.dat', F_Records=None, verbose=False)    
    X_train_nonlin = FilterResults['X Nonlinear Train']
    X_test_nonlin = FilterResults['X Nonlinear Test']
    X_train_lin = FilterResults['X Linear Train']
    Y_train = FilterResults['Response Train']
    X_test_lin = FilterResults['X Linear Test']
    Y_test = FilterResults['Response Test']
    FeaturesNonlinear = FilterResults['Nonlinear Features']
    FeaturesAll = FilterResults['Linear Features All']
    FeaturesReduced = FilterResults['Linear Features Reduced']
    system = FilterResults['System']

    f = open(F_Filter, "rb") # read filter results
    Filter = pickle.load(f)
    f.close()
    F_Train = Filter['Training Set']
    F_Test = Filter['Test Set']
    SetName = Filter['Initial dataset']
    SetName = SetName.split(".")
    SetNameStr = SetName[0] # directory name
    TrainFraction = Filter['Train Fraction Used']
    TrainFractionStr = str(round(TrainFraction*100, 0)) + '%' # subdir name
    if len(TrainFractionStr) == 4:
        TrainFractionStr = '00' + TrainFractionStr
    if len(TrainFractionStr) == 5:
        TrainFractionStr = '0' + TrainFractionStr
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
        if X_test_lin is not None:
            Single_X_test = np.zeros(shape=(X_test_lin.shape[0], len(SingleFeaturesReduced)), dtype=float)
        else:
            Single_X_test = None
        j = 0
        for i in range(0, len(FeaturesReduced), 1):
            if FeaturesReduced[i].nDistances == 1:
                Single_X_train[:, j] = X_train_lin[:, i]
                if X_test_lin is not None:
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
            enet = regression.ENet(L1=L1_Single, eps=eps_single, nAlphas=n_alphas_single, alphas=Alphas, random_state=None)
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
        enet = regression.ENet(L1=L1, eps=eps, nAlphas=n_alphas, alphas=Alphas, random_state=None)
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
        chromosome = copy.deepcopy(ga.BestChromosome)
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
        if chromosome is None:
            break # number of genes in chromosome = number of VIP genes
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
# Gaussian fit    
    print('Gaussian started')  
    X_train_nonlin = FilterResults['X Nonlinear Train']
    X_test_nonlin = FilterResults['X Nonlinear Test']
    Y_train = FilterResults['Response Train']
    kernel = RBF(length_scale=2.2, length_scale_bounds=(1e-02, 1.0)) + WhiteKernel(0.0005)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer=None,\
        n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)    
#    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',\
#        n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
    gp.fit(X_train_nonlin, Y_train) # set from distances  
    
    f = open(F_ga_structure, "wb") # save GA structure
    pickle.dump(ga, f, pickle.HIGHEST_PROTOCOL)
    f.close() 

    f = open(F_gp_structure, "wb") # save GP structure
    pickle.dump(gp, f, pickle.HIGHEST_PROTOCOL)
    f.close() 
    
    directory = SetNameStr
    if not os.path.exists(directory):
        os.makedirs(directory)   
    subdirectory = directory + '\\' + TrainFractionStr
    if os.path.exists(subdirectory):
        shutil.rmtree(subdirectory, ignore_errors=False, onerror=None)
    os.makedirs(subdirectory)     
    if os.path.isfile('SystemDescriptor.'):
        shutil.copy2('SystemDescriptor.', subdirectory + '\\' + 'SystemDescriptor.')
    if os.path.isfile(F_MoleculesDescriptor):
        shutil.copy2(F_MoleculesDescriptor, subdirectory + '\\' + F_MoleculesDescriptor)
    if os.path.isfile('Structure.xlsx'):    
        shutil.move('Structure.xlsx', subdirectory + '\\' + 'Structure.xlsx')
    if os.path.isfile('Linear Features Reduced List.xlsx'): 
        shutil.move('Linear Features Reduced List.xlsx', subdirectory + '\\' +\
                        'Linear Features Reduced List.xlsx')
    if os.path.isfile('Nonlinear Features List.xlsx'): 
        shutil.move('Nonlinear Features List.xlsx', subdirectory + '\\' +\
                        'Nonlinear Features List.xlsx')
    if os.path.isfile(F_Out_Single):
        shutil.move(F_Out_Single, subdirectory + '\\' + F_Out_Single)
    if os.path.isfile(F_Out):
        shutil.move(F_Out, subdirectory + '\\' + F_Out)  
    if os.path.isfile(F_Train):
        shutil.move(F_Train, subdirectory + '\\' + F_Train) 
    if os.path.isfile(F_Test):
        shutil.move(F_Test, subdirectory + '\\' + F_Test) 
    if os.path.isfile('results.txt'):
        shutil.move('results.txt', subdirectory + '\\' + 'results.txt') 
        
# move all *.dat files        
    l = os.listdir('./')
    for name in l:
        if name.endswith('.dat'):
            if os.path.isfile(name):
                shutil.move(name, subdirectory)  # copy2, copyfile      
# move all *.png images        
    l = os.listdir('./')
    for name in l:
        if name.endswith('.png'):
            if os.path.isfile(name):
                shutil.move(name, subdirectory)
# move all *.csv data files        
    l = os.listdir('./')
    for name in l:
        if name.endswith('.csv'):
            if os.path.isfile(name):
                shutil.move(name, subdirectory)                
    return

    