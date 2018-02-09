import numpy as np
import pandas as pd
import sklearn.metrics as skm
import os
import copy
import sys
from time import time
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
from project1 import genetic
from project1 import regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel

RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"
HARTREE_TO_KJMOL = 2625.49963865

def Print(arg, color):
    sys.stdout.write(color)
    print(arg)
    sys.stdout.write(RESET)
    return

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
        return -10
    for i in range(0, len(List), 1):
        if (n >= List[i][0]) and (n <= List[i][1]):
            return i
    return -10 # not in region

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
        plt.savefig('{}{}'.format(FileName, '_RMSE.eps'), bbox_inches='tight', format='eps', dpi=1000)
        # Plot R2 vs. active coefficiens number
        plt.figure(101, figsize = (19, 10))
        plt.plot(nonzero_count_list, rsquared_list, ':')
        plt.xlabel('Active coefficients')
        plt.ylabel('R2')
        plt.title('Backward elimination. R2 vs Active coefficiants')
        plt.axis('tight')
        plt.savefig('{}{}'.format(FileName, '_R2.eps'), bbox_inches='tight', format='eps', dpi=1000)
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
    Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 12)).astype(float), \
        columns=['Feature index','Feature type','Bond 1','Power 1','Intermolecular 1',\
            'Distance 1 type','Bond 2','Power 2', 'Intermolecular 2', 'Distance 2 type',\
            'Number of distances in feature','Idx'], dtype=str)
    max_distances_in_feature = 1
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
        counter = 0
        current_feature_type = FeaturesReduced[index].FeType
        for j in range(0, len(FeaturesAll), 1):
            if FeaturesAll[j].FeType == current_feature_type:
                counter += 1
        Results.loc[i]['Number of distances in feature'] = counter
        Results.loc[i]['Idx'] = FeaturesReduced[index].idx
    if max_distances_in_feature == 1:
        del(Results['Bond 2'])
        del(Results['Power 2'])
        del(Results['Intermolecular 2'])
        del(Results['Distance 2 type'])
    Results.to_excel(writeResults)
    writeResults.save()  
    return

def StoreNonlinearFeaturesDescriprion(FileName, FeaturesNonlinear):
    writeResults = pd.ExcelWriter(FileName, engine='openpyxl')
    nFeatures = len(FeaturesNonlinear)
    Results = pd.DataFrame(np.zeros(shape = (nFeatures, 9)).astype(float), \
        columns=['Feature index','Feature type','Bond','Power','Intermolecular','Distance type',\
        'Number of distances in feature', 'Number of constants in feature', 'Idx'], dtype=str)
    for i in range(0, nFeatures, 1):
        Results.loc[i]['Feature index'] = i
        Results.loc[i]['Feature type'] = FeaturesNonlinear[i].FeType
        Results.loc[i]['Bond'] = FeaturesNonlinear[i].DtP1.Distance.Atom1.Symbol +\
        '-' + FeaturesNonlinear[i].DtP1.Distance.Atom2.Symbol
        Results.loc[i]['Power'] = FeaturesNonlinear[i].DtP1.Power
        Results.loc[i]['Distance type'] = FeaturesNonlinear[i].DtP1.Distance.DiType
        if FeaturesNonlinear[i].DtP1.Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
            Results.loc[i]['Intermolecular'] = 'Yes'
        else:
            Results.loc[i]['Intermolecular'] = 'No'
        Results.loc[i]['Number of distances in feature'] = FeaturesNonlinear[i].nDistances 
        Results.loc[i]['Number of constants in feature'] = FeaturesNonlinear[i].nConstants 
        Results.loc[i]['Idx'] = FeaturesNonlinear[i].idx 
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
    plt.savefig('{}{}'.format(FileName, '_RMSE.eps'), bbox_inches='tight', format='eps', dpi=1000)
    plt.close(fig)
    # Plot R2 vs. active coefficiens number
    fig = plt.figure(101, figsize = (19, 10))
    plt.plot(nonzero_count_list, r2_list, ':')
    plt.xlabel('Active coefficients')
    plt.ylabel('R2')
    plt.title('R2 vs Active coefficiants')
    plt.axis('tight')
    plt.show()
    plt.savefig('{}{}'.format(FileName, '_R2.eps'), bbox_inches='tight', format='eps', dpi=1000)
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
    
def CreateGrid(GridStart, GridEnd, GridSpacing):
    Grid = [] 
    i = GridStart
    while i < (GridEnd-GridSpacing):
        Grid.append((round(i, 2), round(i+GridSpacing, 2)))
        i += GridSpacing
    return Grid

def FilterData(F_Records='SET 6.x', F_MoleculesDescriptor = 'MoleculesDescriptor.',\
        TrainIntervals=[(0, 20)], F_Train = 'Training Set.x', F_Test = 'Test Set.x',\
        D_Train = 'D Train.csv', D_Test = 'D Test.csv', GridStart = 0.0, GridEnd = 20.0,\
        GridSpacing=0.1, ConfidenceInterval=1, TestFraction=0.3, TrainFraction=1, RandomSeed=None):

    MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=F_MoleculesDescriptor)
    if RandomSeed is not None:
        random.seed(RandomSeed)
    else:
        random.seed()    
    GridTrain = CreateGrid(GridStart, GridEnd, GridSpacing) # trained region bins
    GridTest = CreateGrid(GridStart, GridEnd, GridSpacing) # non-trained and trained region bins
    N = np.zeros(shape=(len(GridTest)), dtype=int) # number of points in each grid  
    NTrain = list(np.zeros(shape=(len(GridTrain)), dtype=int)) # count test points
    NTest = list(np.zeros(shape=(len(GridTest)), dtype=int)) # count test points        
    Records = IOfunctions.ReadRecordMolecules(F_Records, MoleculePrototypes) # Read records
    DMin = 1000 # will be shortest distance
    DMax = 0 # will be most distant distance
    nMolecules = Records[0].nMolecules
# Determine DMin, DMax and fill N
    for record in Records:
        if nMolecules == 2:
            if record.R_Average > DMax:
                DMax = record.R_Average
            if record.R_Average < DMin:
                DMin = record.R_Average
            if record.R_Average >= GridStart:
                j = int((record.R_Average - GridStart) / GridSpacing)
                if j < len(N):
                    N[j] += 1
        else:
            if record.R_CenterOfMass_Average > DMax:
                DMax = record.R_CenterOfMass_Average
            if record.R_CenterOfMass_Average < DMin:
                DMin = record.R_CenterOfMass_Average
            if record.R_CenterOfMass_Average >= GridStart:
                j = int((record.R_CenterOfMass_Average - GridStart) / GridSpacing)
                if j < len(N):
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
# trained region        
        if (library.InInterval(GridTest[i][0], TrainIntervals) != -10) and (library.InInterval(GridTest[i][1], TrainIntervals) != -10):
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
        if (library.InInterval(GridTrain[i][0], TrainIntervals) != -10) and (library.InInterval(GridTrain[i][1], TrainIntervals) != -10):
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
        j = library.InInterval(d, GridTrain) 
        if j != -10: # in training region?
            if NTrain[j] < nTrainPointsGrid: # append to training set
                NTrain[j] += 1
                RecordsTrain.append(record)
            else:  # if it is full, append to test set
                j = library.InInterval(d, GridTest) # which interval?
                if j != -10:
                    if NTest[j] < nTestPointsGrid: 
                        NTest[j] += 1
                        RecordsTest.append(record)              
        else: # not training region
            j = library.InInterval(d, GridTest) # which interval?
            if j != -10:
                if NTest[j] < nTestPointsGrid: # append to test set only
                    NTest[j] += 1
                    RecordsTest.append(record)
        del(Records[r]) 

    IOfunctions.store_records(F_Train, RecordsTrain) # store trained set
    IOfunctions.store_records(F_Test, RecordsTest) # store test set
    IOfunctions.store_average_distances(D_Train, RecordsTrain)
    IOfunctions.store_average_distances(D_Test, RecordsTest)
    TestIntervals = [] # Define test regions
    if TrainIntervals[0][0] != 0:
        TestIntervals.append((0, TrainIntervals[0][0]))
    for i in range(0, (len(TrainIntervals)-1), 1):
        if TrainIntervals[i][1] != TrainIntervals[i+1][0]:
            TestIntervals.append((TrainIntervals[i][1], TrainIntervals[i+1][0]))
    if TrainIntervals[-1][1] < GridTest[-1][1]:
        TestIntervals.append((TrainIntervals[-1][1], GridTest[-1][1]))
    
    results = {'Initial dataset': F_Records,'Number of molecules per record': nMolecules,\
               'Train Intervals': TrainIntervals,'Test Intervals': TestIntervals,\
               'Train records number': len(RecordsTrain),'Train Grid': GridTrain,\
               'Test Grid': GridTest, 'Test records number': len(RecordsTest),\
               'Molecule prototypes': MoleculePrototypes, 'Max points per grid': nPointsGrid,\
               'Train points per grid': nTrainPointsGrid, 'Train Fraction Used': TrainFraction,\
               'Test points per grid': nTestPointsGrid, 'Confidence Interval used': ConfidenceInterval,\
               'Training Set': F_Train, 'Test Set': F_Test, 'COM Train': D_Train, 'COM Test': D_Test}
            
    return results
    
def ReadData(F_data, Atoms):
        # Read coordinates from file
    atoms = copy.deepcopy(Atoms)
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
            record_list.append(structure.RecordAtoms(float(s[0]), atoms_list))
            j = 0
            atoms_list = []
            atoms = copy.deepcopy(Atoms)
        elif (len(s) == 4): 
            atoms[j].x = float(s[1])
            atoms[j].y = float(s[2])
            atoms[j].z = float(s[3])
            atoms_list.append(atoms[j])
            j += 1
        i += 1
    return record_list

def StoreFeatures(F_LinearFeatures, first, last, FeaturesAll, FeaturesReduced, record_list, Atoms):
# Storing energy
    NofFeaturesReduced = len(FeaturesReduced)
    NofFeatures = len(FeaturesAll)
# calculating and storing distances  
    features_array = np.zeros(shape=(last-first, len(FeaturesAll)), dtype=float) 
    for j in range(0, len(FeaturesAll), 1):
        for i in range(first, last, 1):
            if (FeaturesAll[j].nDistances == 1):
# features with only one distance
                atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                d = np.sqrt((record_list[i].atoms[atom1_index].x - record_list[i].atoms[atom2_index].x)**2 +\
                            (record_list[i].atoms[atom1_index].y - record_list[i].atoms[atom2_index].y)**2 +\
                            (record_list[i].atoms[atom1_index].z - record_list[i].atoms[atom2_index].z)**2)            
                r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
            if (FeaturesAll[j].nDistances == 2):
# features with two distances
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

def GenerateFeatures(Filter, Files, F_SystemDescriptor='SystemDescriptor.'): 

    def CreateDtPList(Distances, Description):
    # make list of distances raised to corresponding power    
        Powers = []
        for distance in Distances:
            for description in Description:
                if distance.isIntermolecular == description[2]:
                    if ((distance.Atom1.Symbol == description[0]) and (distance.Atom2.Symbol == description[1])) or\
                        ((distance.Atom1.Symbol == description[1]) and (distance.Atom2.Symbol == description[0])):
                        Powers.append(description[3])    
        DtP_list = []
        for i in range(0, nDistances, 1):
            for power in Powers[i]:
                DtP_list.append(structure.Distance_to_Power(Distances[i], power, PowerDigits=2))
        return DtP_list
    
    def CreateSingleFeaturesAll(DtP_Single_list, FeType='Linear', nDistances=1, nConstants=1):    
        FeaturesAll = [] # single linear features list
        for DtP in DtP_Single_list:
            FeaturesAll.append(structure.Feature(FeType=FeType, DtP1=DtP, DtP2=None,\
                nDistances=nDistances, nConstants=nConstants)) 
        return FeaturesAll
    
    def CreateDoubleFeaturesAll(DtP_Double_list, FeType='Linear',\
            nConstants=1, IncludeOnlySamePower=False, IncludeSameType=True):    
        FeaturesAll = [] # double linear features list
        for i in range(0, len(DtP_Double_list), 1):
            for j in range(i+1, len(DtP_Double_list), 1):
                if DtP_Double_list[i].Power > DtP_Double_list[j].Power:
                    continue # skip duplicates
                if IncludeOnlySamePower: # pairs with same power will only be added               
                    if DtP_Double_list[i].Power != DtP_Double_list[j].Power:
                        continue # include only products with same powers                 
                if not IncludeSameType: # if this parameter is false
                    if DtP_Double_list[i].Distance.DiType == DtP_Double_list[j].Distance.DiType: 
                        continue # skip if distances of the same type
                FeaturesAll.append(structure.Feature(FeType=FeType,\
                    DtP1=DtP_Double_list[i], DtP2=DtP_Double_list[j],\
                    nDistances=2, nConstants=nConstants))
        return FeaturesAll

    def CreateTripleFeaturesAll(DtP_Triple_list, FeType='Linear',\
            nConstants=1, IncludeOnlySamePower=False, IncludeSameType=True):    
        FeaturesAll = [] # double linear features list
        for i in range(0, len(DtP_Triple_list), 1):
            for j in range(i+1, len(DtP_Triple_list), 1):
                for k in range(j+1, len(DtP_Triple_list), 1):
                    if (DtP_Triple_list[i].Power > DtP_Triple_list[j].Power) or\
                        DtP_Triple_list[j].Power > DtP_Triple_list[k].Power:
                        continue # skip duplicates
                    if IncludeOnlySamePower: # pairs with same power will only be added               
                        if not ((DtP_Triple_list[i].Power == DtP_Triple_list[j].Power) and\
                            (DtP_Triple_list[j].Power == DtP_Triple_list[k].Power)):
                            continue # include only products with same powers                 
                    if not IncludeSameType: # if this parameter is false
                        if (DtP_Triple_list[i].Distance.DiType == DtP_Triple_list[j].Distance.DiType) and\
                            (DtP_Triple_list[j].Distance.DiType == DtP_Triple_list[k].Distance.DiType): 
                            continue # skip if distances of the same type
                    FeaturesAll.append(structure.Feature(FeType=FeType,\
                        DtP1=DtP_Triple_list[i], DtP2=DtP_Triple_list[j], DtP3=DtP_Triple_list[k],\
                        nDistances=3, nConstants=nConstants))
        return FeaturesAll
    
    def CreateFeaturesReduced(FeaturesAll):
    # Make list of reduced features
        FeaturesReduced = []
        FeType_list = []
        for i in range(0, len(FeaturesAll), 1):
            if (FeaturesAll[i].FeType not in FeType_list):
                FeType_list.append(FeaturesAll[i].FeType)
                feature = copy.deepcopy(FeaturesAll[i])
                FeaturesReduced.append(feature)
    # store global indices for each reduced feature
        for i in range(0, len(FeaturesReduced), 1):
            for j in range(0, len(FeaturesAll), 1):
                if (FeaturesAll[j].FeType == FeaturesReduced[i].FeType):
                    if j not in FeaturesReduced[i].idx:
                        FeaturesReduced[i].idx.append(j)
        return FeaturesReduced

    def CreateFeaturesReducedMimic(FeaturesAll):
    # Make list of reduced features
        FeaturesReduced = copy.deepcopy(FeaturesAll)
        for i in range(0, len(FeaturesReduced), 1):
            FeaturesReduced[i].idx.append(i)
        return FeaturesReduced
    
    for i in Files.values():
        EraseFile(i)
    
    Prototypes = IOfunctions.ReadMoleculeDescription(F_SystemDescriptor, keyword='MoleculeDescription')
    Atoms, Molecules = IOfunctions.ReadSystemDescription(F_SystemDescriptor, 'SYSTEM')        
    LinearSingle = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='LinearSingle')
    LinearDouble = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='LinearDouble')
    LinearTriple = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='LinearTriple')
    ExpSingle = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='ExpSingle')
    ExpDouble = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='ExpDouble')
    GaussianSingle = IOfunctions.ReadFeatureDescription(F_SystemDescriptor, keyword='GaussianSingle')
    
    nAtoms = len(Atoms)
    nMolecules = len(Molecules)
    AtomTypesList = []
    for i in Atoms: # create atom types list
        if i.AtType not in AtomTypesList:
            AtomTypesList.append(i.AtType)
    nAtomTypes = len(AtomTypesList)
    Distances = [] # create list of distances from list of atoms
    for i in range(0, nAtoms, 1):
        for j in range(i+1, nAtoms, 1):
            Distances.append(structure.Distance(Atoms[i], Atoms[j]))
    DiTypeList = [] # create distances types list
    for i in Distances:
        if i.DiType not in DiTypeList:
            DiTypeList.append(i.DiType)
    nDistances = len(Distances)
    nDiTypes = len(DiTypeList)
    MoleculeNames = []
    for i in Prototypes: # create list of molecule names
        MoleculeNames.append(i.Name)
    for i in Molecules: # update data in Molecules from Prototype
        if i.Name in MoleculeNames:
            idx = MoleculeNames.index(i.Name)
            i.Mass = Prototypes[idx].Mass
            for j in range(0, len(i.Atoms), 1):
                i.Atoms[j].Mass = Prototypes[idx].Atoms[j].Mass
                i.Atoms[j].Radius = Prototypes[idx].Atoms[j].Radius
                i.Atoms[j].Bonds = Prototypes[idx].Atoms[j].Bonds
                i.Atoms[j].Bonds = library.replace_numbers(i.AtomIndex, Prototypes[idx].AtomIndex, i.Atoms[j].Bonds)
    for i in Molecules:
        i._refresh() # update data from prototype
    # store required data in system object    
    System = structure.System(Atoms=Atoms, Molecules=Molecules, Prototypes=Prototypes,\
        nAtoms=nAtoms, nAtomTypes=nAtomTypes, nMolecules=nMolecules, Distances=Distances,\
        nDistances=nDistances, nDiTypes=nDiTypes)
    
    records_train_list = IOfunctions.ReadRecordAtoms(Filter['Training Set'], Atoms)
    records_test_list = IOfunctions.ReadRecordAtoms(Filter['Test Set'], Atoms)
    IOfunctions.StoreEnergy(Files['Response Train'], records_train_list)
    IOfunctions.StoreEnergy(Files['Response Test'], records_test_list)
    
    if LinearSingle['Include']:
        DtP_LinearSingle_list = CreateDtPList(Distances, LinearSingle['Description'])
        FeaturesLinearSingleAll = CreateSingleFeaturesAll(DtP_LinearSingle_list,\
            FeType='Linear', nDistances=1, nConstants=1)
        FeaturesLinearSingleReduced = CreateFeaturesReduced(FeaturesLinearSingleAll)
        IOfunctions.StoreLinearFeatures(Files['Linear Single Train'], FeaturesLinearSingleAll,\
            FeaturesLinearSingleReduced, records_train_list, Atoms)
        IOfunctions.StoreLinearFeatures(Files['Linear Single Test'], FeaturesLinearSingleAll,\
            FeaturesLinearSingleReduced, records_test_list, Atoms)
    else:
        FeaturesLinearSingleAll = None
        FeaturesLinearSingleReduced = None
        
    if LinearDouble['Include']:
        DtP_LinearDouble_list = CreateDtPList(Distances, LinearDouble['Description'])
        FeaturesLinearDoubleAll = CreateDoubleFeaturesAll(DtP_LinearDouble_list,\
            FeType='Linear', IncludeOnlySamePower=LinearDouble['IncludeOnlySamePower'],\
            IncludeSameType=LinearDouble['IncludeSameType'], nConstants=1)
        FeaturesLinearDoubleReduced = CreateFeaturesReduced(FeaturesLinearDoubleAll)
        IOfunctions.StoreLinearFeatures(Files['Linear Double Train'], FeaturesLinearDoubleAll,\
            FeaturesLinearDoubleReduced, records_train_list, Atoms)
        IOfunctions.StoreLinearFeatures(Files['Linear Double Test'], FeaturesLinearDoubleAll,\
            FeaturesLinearDoubleReduced, records_test_list, Atoms)        
    else:
        FeaturesLinearDoubleAll = None
        FeaturesLinearDoubleReduced = None

    if LinearTriple['Include']:
        DtP_LinearTriple_list = CreateDtPList(Distances, LinearTriple['Description'])
        FeaturesLinearTripleAll = CreateTripleFeaturesAll(DtP_LinearTriple_list,\
            FeType='Linear', IncludeOnlySamePower=LinearTriple['IncludeOnlySamePower'],\
            IncludeSameType=LinearTriple['IncludeSameType'], nConstants=1)
        FeaturesLinearTripleReduced = CreateFeaturesReduced(FeaturesLinearTripleAll)
        IOfunctions.StoreLinearFeatures(Files['Linear Triple Train'], FeaturesLinearTripleAll,\
            FeaturesLinearTripleReduced, records_train_list, Atoms)
        IOfunctions.StoreLinearFeatures(Files['Linear Triple Test'], FeaturesLinearTripleAll,\
            FeaturesLinearTripleReduced, records_test_list, Atoms)        
    else:
        FeaturesLinearTripleAll = None
        FeaturesLinearTripleReduced = None
                
    if ExpSingle['Include']:
        DtP_ExpSingle_list = CreateDtPList(Distances, ExpSingle['Description'])
        FeaturesExpSingleAll = CreateSingleFeaturesAll(DtP_ExpSingle_list,\
            FeType='Exp', nDistances=1, nConstants=2)
        IOfunctions.StoreExpSingleFeatures(Files['Exp Single Train D'],\
            Files['Exp Single Train D^n'], FeaturesExpSingleAll, records_train_list, Atoms)
        IOfunctions.StoreExpSingleFeatures(Files['Exp Single Test D'],\
            Files['Exp Single Test D^n'], FeaturesExpSingleAll, records_test_list, Atoms)
    else:
        FeaturesExpSingleAll = None
                
    if ExpDouble['Include']:    
        DtP_ExpDouble_list = CreateDtPList(Distances, ExpDouble['Description'])
        FeaturesExpDoubleAll = CreateDoubleFeaturesAll(DtP_ExpDouble_list, FeType='Exp',\
            nConstants=3, IncludeOnlySamePower=ExpDouble['IncludeOnlySamePower'],\
            IncludeSameType=ExpDouble['IncludeSameType'])
        IOfunctions.StoreExpDoubleFeatures(Files['Exp Double Train D1'],\
            Files['Exp Double Train D2'], Files['Exp Double Train D1^mD2^n'],\
            FeaturesExpDoubleAll, records_train_list, Atoms)
        IOfunctions.StoreExpDoubleFeatures(Files['Exp Double Test D1'],\
            Files['Exp Double Test D2'], Files['Exp Double Test D1^mD2^n'],\
            FeaturesExpDoubleAll, records_test_list, Atoms)
    else:
        FeaturesExpDoubleAll = None

    if GaussianSingle['Include']:
        DtP_GaussianSingle_list = CreateDtPList(Distances, GaussianSingle['Description'])
        FeaturesGaussianSingleAll = CreateSingleFeaturesAll(DtP_GaussianSingle_list,\
            FeType='Gauss', nDistances=1, nConstants=1)
        FeaturesGaussianSingleReduced = CreateFeaturesReduced(FeaturesGaussianSingleAll)
        IOfunctions.StoreLinearFeatures(Files['Gaussian Single Train'],\
            FeaturesGaussianSingleAll, FeaturesGaussianSingleReduced, records_train_list, Atoms)
        IOfunctions.StoreLinearFeatures(Files['Gaussian Single Test'],\
            FeaturesGaussianSingleAll, FeaturesGaussianSingleReduced, records_test_list, Atoms)
    else:
        FeaturesGaussianSingleAll = None
        FeaturesGaussianSingleReduced = None
        
    Structure = {'System': System,\
        'FeaturesLinearSingleAll': FeaturesLinearSingleAll,\
        'FeaturesLinearSingleReduced': FeaturesLinearSingleReduced,\
        'FeaturesLinearDoubleAll': FeaturesLinearDoubleAll,\
        'FeaturesLinearDoubleReduced': FeaturesLinearDoubleReduced,\
        'FeaturesLinearTripleAll': FeaturesLinearTripleAll,\
        'FeaturesLinearTripleReduced': FeaturesLinearTripleReduced,\
        'FeaturesExpSingleAll': FeaturesExpSingleAll,\
        'FeaturesExpDoubleAll': FeaturesExpDoubleAll,\
        'FeaturesGaussianSingleAll': FeaturesGaussianSingleAll,\
        'FeaturesGaussianSingleReduced': FeaturesGaussianSingleReduced}
    
    IOfunctions.StoreStructure(Files['Structure'], Structure)
    
    return Structure  
            
def GetFitGA(FilterDataResults, Files, GenerateFeaturesResults, F_xlsx='Fit', F_ENet='ENet path',\
        F_GA='GA path', UseVIP=False, nVIP=None, FirstAlgorithm='GA',\
        L1_Single=0.7, eps_single=1e-3, n_alphas_single=100, L1=0.7, eps=1e-3,\
        n_alphas=100, alpha_grid_start=-7, alpha_grid_end=-3, cv=30, MinChromosomeSize=5,\
        ChromosomeSize=15, StopTime=600, BestFitStopTime=100, BestFitMaxQueue=0,\
        nIter=500, PopulationSize=100, MutationProbability=0.3, MutationInterval=[1, 3],\
        EliteFraction=0.4, MutationCrossoverFraction=0.3, CrossoverFractionInterval=[0.6, 0.4],\
        UseCorrelationMutation=True, MinCorrMutation=0.8, CrossoverMethod='Random',\
        MutationMethod='Correlated', LinearSolver='sklearn', cond=1e-03,\
        lapack_driver='gelsy', UseCorrelationBestFit=False, MinCorrBestFit=0.8,\
        PrintInterval=50, RandomSeed=None, BestFitPathLen=0, verbose=True):

# 'sklearn', 'scipy', 'statsmodels'
# 'gelsd', 'gelsy', 'gelss', for scipy solver
# 'Random' or 'Best'
# 'Random' or 'Correlated'
    
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
        
    if UseVIP and X_LinearSingle_train is not None:
        if nVIP is None:
            if GenerateFeaturesResults.System.nAtoms == 6: # two water molecules system
                nVIP = 5
            if GenerateFeaturesResults.System.nAtoms == 9: # three water molecules system
                nVIP = 3

        t = time()       
        ga = genetic.GA(PopulationSize=PopulationSize, ChromosomeSize=ChromosomeSize,\
            MutationProbability=MutationProbability, MutationInterval=MutationInterval,\
            EliteFraction=EliteFraction, MutationCrossoverFraction=MutationCrossoverFraction,\
            CrossoverFractionInterval=CrossoverFractionInterval, PrintInterval=PrintInterval,\
            StopTime=StopTime, RandomSeed=RandomSeed, verbose=verbose,\
            UseCorrelationMutation=UseCorrelationMutation, MinCorrMutation=MinCorrMutation,\
            UseCorrelationBestFit=UseCorrelationBestFit, MinCorrBestFit=MinCorrBestFit)   

        if (FirstAlgorithm == 'ENet'):
            print('Only linear features will be considered')
            Alphas = np.logspace(alpha_grid_start, alpha_grid_end, num=n_alphas_single,\
                                 endpoint=True, base=10.0, dtype=float)
            enet = regression.ENet(L1=L1_Single, eps=eps_single, nAlphas=None,\
                                   alphas=Alphas, random_state=None)
            print('Number of features go to elastic net regularisation = ', X_LinearSingle_train.shape[1])
            enet.fit(X_LinearSingle_train, Y_train, VIP_idx=None, Criterion='Mallow', normalize=True,\
                max_iter=1000, tol=0.0001, cv=cv, n_jobs=1, selection='random', verbose=verbose)
            enet.plot_path(1, F_ENet='{} {}'.format(F_ENet, 'Single.png'))
            idx = enet.idx
            Gene_list = []
            for i in idx:
                Gene_list.append(genetic.GA.Gene(i, Type=0))
            chromosome = ga.Chromosome(Gene_list)
            chromosome.erase_score()
            chromosome.score(x_expD_train=None, x_expDn_train=None, x_lin_train=X_LinearSingle_train,\
                y_train=Y_train, x_expD_test=None, x_expDn_test=None, x_lin_test=X_LinearSingle_train,\
                y_test=Y_test,\
                LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
            if ga.LinearSolver == 'statsmodels':
                chromosome.rank_sort_pValue()
            else:
                chromosome.rank(x_expD=None, x_expDn=None, x_lin=X_LinearSingle_train,\
                    y=Y_train,\
                    LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)
                chromosome.sort(order='Most important first')  
                ga.n_lin = X_LinearSingle_train.shape[1]
            t_sec = time() - t
            print("\n", 'Elastic Net worked ', t_sec, 'sec')  
            print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
        if FirstAlgorithm == 'GA':
            print('Genetic Algorithm for all features')
            ga.fit(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_LinearSingle_train,\
                y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_LinearSingle_test,\
                y_test=Y_test, idx_exp=None, idx_lin=None, VIP_idx_exp=None,\
                VIP_idx_lin=None, CrossoverMethod=CrossoverMethod, MutationMethod=MutationMethod,\
                LinearSolver=LinearSolver, cond=cond, lapack_driver=lapack_driver, nIter=nIter)
            t_sec = time() - t
            print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
            chromosome = ga.BestChromosome        
            print("\n", 'Features left for Backward Elimination and Search Alternative = ',\
                  chromosome.Size)
            ga.PlotChromosomes(2, ga.BestChromosomes, XAxis='time', YAxis='R2 Train',\
                PlotType='Scatter', F=F_GA)
        while chromosome.Size > 1:
            if chromosome.Size <= ga.ChromosomeSize:
                chromosome = ga.BestFit(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train,\
                    x_lin=X_LinearSingle_train, y=Y_train, verbose=True) # returns ranked and sorted chromosome              
                chromosome.score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_LinearSingle_train,\
                    y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_LinearSingle_test,\
                    y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver) 
                chromosome.Origin = 'Best Fit'
                chromosome_copy = copy.deepcopy(chromosome)
                chromosome_copy.print_score()
                ga.DecreasingChromosomes.append(chromosome_copy)
            chromosome = ga.RemoveWorstGene(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train,\
                x_lin=X_LinearSingle_train, y=Y_train, verbose=True)
        t_sec = time() - t
        ga.PlotChromosomes(3, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='RMSE Train',\
            PlotType='Line', F='Single')
        ga.PlotChromosomes(4, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='R2 Adjusted Train',\
            PlotType='Line', F='Single')
        print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
        F_single_xlsx = '{} {}'.format(F_xlsx, 'Single.xlsx')
        ga.Results_to_xlsx(F_single_xlsx, ga.DecreasingChromosomes,\
            FeaturesNonlinear=GenerateFeaturesResults['FeaturesExpSingleAll'],\
            FeaturesAll=GenerateFeaturesResults['FeaturesLinearSingleAll'],\
            FeaturesReduced=GenerateFeaturesResults['FeaturesLinearSingleReduced'])        
        for i in ga.DecreasingChromosomes:
            if i.Size == nVIP:
                VIP_idx_exp = i.get_genes_list(Type=1)
                VIP_idx_lin = i.get_genes_list(Type=0)
    else:
        VIP_idx_exp = []
        VIP_idx_lin = []      
# proceed all features 
    t = time()           
    ga = genetic.GA(PopulationSize=PopulationSize, ChromosomeSize=ChromosomeSize,\
        MutationProbability=MutationProbability, MutationInterval=MutationInterval,\
        EliteFraction=EliteFraction, MutationCrossoverFraction=MutationCrossoverFraction,\
        CrossoverFractionInterval=CrossoverFractionInterval, PrintInterval=PrintInterval,\
        StopTime=StopTime, RandomSeed=RandomSeed, verbose=verbose,\
        UseCorrelationMutation=UseCorrelationMutation, MinCorrMutation=MinCorrMutation,\
        UseCorrelationBestFit=UseCorrelationBestFit, MinCorrBestFit=MinCorrBestFit)
# linear only for now
#    idx = list(range(0, X_Linear_train.shape[1], 1))
#    ga.n_lin = X_Linear_train.shape[1]   
    if (FirstAlgorithm == 'ENet'):
        print('Only linear features will be considered')
        Alphas = np.logspace(alpha_grid_start, alpha_grid_end, num=n_alphas,\
            endpoint=True, base=10.0, dtype=float)
        enet = regression.ENet(L1=L1, eps=eps, nAlphas=None, alphas=Alphas, random_state=None)
        print('Number of features go to elastic net regularisation = ', X_Linear_train.shape[1])
        enet.fit(X_Linear_train, Y_train, VIP_idx=VIP_idx_lin, Criterion='Mallow', normalize=True,\
            max_iter=1000, tol=0.0001, cv=cv, n_jobs=1, selection='random', verbose=verbose)        
        enet.plot_path(2, F_ENet='{} {}'.format(F_ENet, '.png'))
        idx = enet.idx
        Gene_list = []
        for i in idx:
            Gene_list.append(genetic.GA.Gene(i, Type=0))
        chromosome = ga.Chromosome(Gene_list)
        chromosome.erase_score()
        chromosome.score(x_expD_train=None, x_expDn_train=None, x_lin_train=X_Linear_train,\
            y_train=Y_train, x_expD_test=None, x_expDn_test=None, x_lin_test=X_Linear_test,\
            y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond,\
            lapack_driver=ga.lapack_driver)        
        chromosome.rank_sort(x_expD=None, x_expDn=None, x_lin=X_Linear_train, y=Y_train,\
            LinearSolver=ga.LinearSolver, cond=ga.cond, lapack_driver=ga.lapack_driver)        
        t_sec = time() - t
        print("\n", 'Elastic Net worked ', t_sec, 'sec')  
        print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    if FirstAlgorithm == 'GA':
        print('Genetic Algorithm for all features')
        ga.fit(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_Linear_train, y_train=Y_train,\
            x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_Linear_test, y_test=Y_test,\
            idx_exp=None, idx_lin=None, VIP_idx_exp=VIP_idx_exp,\
            VIP_idx_lin=VIP_idx_lin, CrossoverMethod=CrossoverMethod, MutationMethod=MutationMethod,\
            LinearSolver=LinearSolver, cond=cond, lapack_driver=lapack_driver, nIter = nIter)
        t_sec = time() - t
        print("\n", 'Genetic Algorithm worked ', t_sec, 'sec')
        chromosome = copy.deepcopy(ga.BestChromosome)
        print("\n", 'Features left for Backward Elimination and Search Alternative = ',\
              chromosome.Size)
        ga.PlotChromosomes(3, ga.BestChromosomes, XAxis='time', YAxis='R2 Train',\
            PlotType='Scatter', F=F_GA)
    while chromosome.Size >= MinChromosomeSize:
        if chromosome.Size <= ga.ChromosomeSize:
#            chromosome = ga.BestFit(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train, x_lin=X_Linear_train,\
#                y=Y_train, verbose=True)
            chromosome = ga.BestFit2(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train, x_lin=X_Linear_train,\
                y=Y_train, epoch=BestFitStopTime, q_max=BestFitMaxQueue, BestFitPathLen=BestFitPathLen, start_time=time(), verbose=True)
            chromosome.score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_Linear_train,\
                y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_Linear_test,\
                y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond,\
                lapack_driver=ga.lapack_driver) 
            chromosome.Origin = 'Best Fit'
            chromosome_copy = copy.deepcopy(chromosome)
            chromosome_copy.print_score()
            ga.DecreasingChromosomes.append(chromosome_copy)
# chromosome must be sorted before             
        chromosome = ga.RemoveWorstGene(chromosome, x_expD=X_ExpSingleD_train, x_expDn=X_ExpSingleDn_train,\
            x_lin=X_Linear_train, y=Y_train, verbose=True)
        if chromosome is None:
            break # number of genes in chromosome = number of VIP genes
    t_sec = time() - t
    ga.PlotChromosomes(4, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='RMSE Train',\
        PlotType='Line', F='Default')
    ga.PlotChromosomes(5, ga.DecreasingChromosomes, XAxis='Nonzero', YAxis='R2 Adjusted Train',\
        PlotType='Line', F='Default')
    ga.PlotChromosomes(6, ga.BestFitPath, XAxis='time', YAxis='RMSE Train',\
        PlotType='Scatter', F='Best Fit Path')
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    
    gene0 = genetic.Gene(0, Type=0, p_Value=None, rank=None)
    gene1 = genetic.Gene(15, Type=0, p_Value=None, rank=None)
    gene2 = genetic.Gene(30, Type=0, p_Value=None, rank=None)
    gene3 = genetic.Gene(5, Type=0, p_Value=None, rank=None)
    gene4 = genetic.Gene(11, Type=0, p_Value=None, rank=None)
    etalon = genetic.Chromosome([gene0, gene1, gene2, gene3, gene4])
    etalon.score(x_expD_train=X_ExpSingleD_train, x_expDn_train=X_ExpSingleDn_train, x_lin_train=X_Linear_train,\
        y_train=Y_train, x_expD_test=X_ExpSingleD_test, x_expDn_test=X_ExpSingleDn_test, x_lin_test=X_Linear_test,\
        y_test=Y_test, LinearSolver=ga.LinearSolver, cond=ga.cond,\
        lapack_driver=ga.lapack_driver)
    if etalon is not None:
        ga.BestFitPath.append(etalon)  
        ga.DecreasingChromosomes.append(etalon)
    ga.Results_to_xlsx('{} {}'.format(F_xlsx, '.xlsx'), \
        FeaturesNonlinear=GenerateFeaturesResults['FeaturesExpSingleAll'],\
        FeaturesAll=FeaturesLinearAll, FeaturesReduced=FeaturesLinearReduced,\
        X_Linear=X_Linear_train)
    return ga

# linear search
def GetFitGP(Files, GaussianPrecision=10, GaussianStart=0.01, GaussianEnd=20, GaussianLen=5):
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    # Gaussian fit    
    print('Gaussian started')      
    k = 0 # estimate best gaussian
    gpR2_array = np.zeros(shape=(GaussianLen), dtype=float)
    Start = GaussianStart
    End = GaussianEnd
    while k < GaussianPrecision:
        print('Start = ', Start)
        print('End = ', End)        
        grid = np.linspace(Start, End, GaussianLen)
        for i in range(0, GaussianLen, 1):
            if gpR2_array[i] == 0:
                kernel = RBF(length_scale=grid[i], length_scale_bounds=(1e-10, 1e+10)) + WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-10, 1e+1))# + RationalQuadratic(length_scale=1.2, alpha=0.78)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
                    n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
                gp.fit(X_Gaussian_train, Y_train) # set from distances
                gpR2 = gp.score(X_Gaussian_test, Y_test)
                print(grid[i], '  ', gpR2)
                gpR2_array[i] = gpR2
        index = np.argmax(gpR2_array)
        gpR2_new = np.zeros(shape=(GaussianLen), dtype=float)
        print('Index = ', index)
        print('length_scale ', grid[index])
        print('R2 ', gpR2_array[index])
        if index == 0:
            sys.stdout.write(RED)                                
            print('Check gaussian lower interval')
            sys.stdout.write(RESET)
            Start = grid[index]
            gpR2_new[0] = gpR2_array[index]
        else:
            Start = grid[index-1]
            gpR2_new[0] = gpR2_array[index-1]
        if index == (GaussianLen-1):
            sys.stdout.write(RED)                                
            print('Check gaussian upper interval')
            sys.stdout.write(RESET)
            End = grid[index]
            gpR2_new[-1] = gpR2_array[index]
        else:
            End = grid[index+1]
            gpR2_new[-1] = gpR2_array[index+1]
        k += 1
        gpR2_array = copy.deepcopy(gpR2_new)
   
    kernel = RBF(length_scale=grid[index], length_scale_bounds=(1e-10, 1e+10)) + WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
        n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
    gp.fit(X_Gaussian_train, Y_train) # set from distances
    gpR2 = gp.score(X_Gaussian_test, Y_test)
    Print(gpR2, RED)
    return gp
    
# basinhopping optimizer
def GetFitGP2(Files):
    
    def fun(x):
        kernel = RBF(length_scale=x[0], length_scale_bounds=(1e-3, 1e+2)) +\
            WhiteKernel(noise_level=x[1], noise_level_bounds=(1e-10, 1e+1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
            n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None)
        gp.fit(X_Gaussian_train, Y_train) # set from distances
        gpR2 = gp.score(X_Gaussian_test, Y_test)
        print('R2 = ', gpR2, 'scale = ', x[0], 'noise = ', x[1])
        return -gpR2

    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    # Gaussian fit 
    
    from scipy.optimize import basinhopping
    x0 = np.array([1, 1e-4])
    res = basinhopping(fun, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None,\
        take_step=None, accept_test=None, callback=None, interval=50,\
        disp=True, niter_success=2, seed=None)
    
    return res
    
# build in optimizer
def GetFitGP3(Files):
    print('Gaussian started') 
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-1, 1e+2)) +\
        WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-20, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer='fmin_l_bfgs_b',\
        n_restarts_optimizer=2, normalize_y=True, copy_X_train=True, random_state=101)
    gp.fit(X_Gaussian_train, Y_train) # set from distances
    gpR2 = gp.score(X_Gaussian_test, Y_test)
    print('R2 = ', gpR2)    
    return gp

# user defined kernel
def GetFitGP4(Files):
    print('Gaussian started') 
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test'])  
    
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-1, 1e+2)) +\
        WhiteKernel(noise_level=5e-4, noise_level_bounds=(1e-20, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer='fmin_l_bfgs_b',\
        n_restarts_optimizer=2, normalize_y=True, copy_X_train=True, random_state=101)
    gp.fit(X_Gaussian_train, Y_train) # set from distances
    gpR2 = gp.score(X_Gaussian_test, Y_test)
    print('R2 = ', gpR2)    
    return gp

# hill climbing for Gaussian Process
    
class NodeHill(dict):
    
    def __init__(self, parent=None, length_scale=None, noise_level=None,\
        length_scale_inc=None, noise_level_inc=None, length_scale_bounds=None,\
        noise_level_bounds=None, function=None):
        
        self.parent = parent
        self.children = []
        self.length_scale = length_scale
        self.noise_level = noise_level # y
        self.length_scale_inc = length_scale_inc
        self.noise_level_inc = noise_level_inc
        self.length_scale_bounds = length_scale_bounds
        self.noise_level_bounds = noise_level_bounds
        self.function=function
        self.fitness = self.get_fitness() 
        
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return ''.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
        
    def get_fitness(self):
        return self.function(self.length_scale, self.noise_level)

    def create_children(self):    
        if self.children != []:
            return
        length_scale_new = self.length_scale - self.length_scale_inc
        if length_scale_new > self.length_scale_bounds[0]: # greater than left bound
            child = NodeHill(parent=self, length_scale=length_scale_new,\
                noise_level=self.noise_level, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)
            self.children.append(child)
        length_scale_new = self.length_scale + self.length_scale_inc
        if length_scale_new < self.length_scale_bounds[1]: # smaller than right bound
            child = NodeHill(parent=self, length_scale=length_scale_new,\
                noise_level=self.noise_level, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)            
            self.children.append(child)
        noise_level_new = self.noise_level - self.noise_level_inc
        if noise_level_new > self.noise_level_bounds[0]: # greater than left bound
            child = NodeHill(parent=self, length_scale=self.length_scale,\
                noise_level=noise_level_new, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)
            self.children.append(child)
        noise_level_new = self.noise_level + self.noise_level_inc
        if noise_level_new < self.noise_level_bounds[0]: # smaller than right bound
            child = NodeHill(parent=self, length_scale=self.length_scale,\
                noise_level=noise_level_new, length_scale_inc=self.length_scale_inc,\
                noise_level_inc=self.noise_level_inc, length_scale_bounds=self.length_scale_bounds,\
                noise_level_bounds=self.noise_level_bounds, function=self.function)
            self.children.append(child)            
        return
    
    def get_highest_valued_child(self):
        if self.children == []:
            return None
        best = self.children[0]
        for i in range(1, len(self.children), 1):
            if best.fitness < self.children[i].fitness:
                best = self.children[i]
        return best

def GetFitGP5(Files, length_scale_start, noise_level_start, length_scale_bounds,\
    noise_level_bounds, length_scale_inc=0.1, noise_level_inc=0.1,\
    length_scale_inc_min=0.01, noise_level_inc_min=0.01, simulation=None, random_state=None):
    
    def f(x, y):
        kernel = RBF(length_scale=x, length_scale_bounds=None) +\
            WhiteKernel(noise_level=y, noise_level_bounds=None)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
            n_restarts_optimizer=0, normalize_y=True, copy_X_train=True,\
            random_state=random_state)
        gp.fit(X_Gaussian_train, Y_train) # set from distances
        gpR2 = gp.score(X_Gaussian_test, Y_test)
        return gpR2

    def hill(length_scale_start, noise_level_start):
        Length_scale_inc = length_scale_inc * (length_scale_bounds[0] + length_scale_bounds[1]) / 2
        Length_scale_inc_min = length_scale_inc_min * (length_scale_bounds[0] + length_scale_bounds[1]) / 2
        Noise_level_inc = noise_level_inc * (noise_level_bounds[0] + noise_level_bounds[1]) / 2
        Noise_level_inc_min = noise_level_inc_min * (noise_level_bounds[0] + noise_level_bounds[1]) / 2        
        current = NodeHill(parent=None, length_scale=length_scale_start,\
            noise_level=noise_level_start, length_scale_inc=Length_scale_inc,\
            noise_level_inc=Noise_level_inc, length_scale_bounds=length_scale_bounds,\
            noise_level_bounds=noise_level_bounds, function=f)
        count = 0 
        while True:
            count += 1
            print('R2=', current.fitness, 'length_scale=', current.length_scale,\
                 'noise_level=', current.noise_level, 'length_scale_inc=',\
                 current.length_scale_inc, 'noise_level_inc=', current.noise_level_inc)
            current.create_children()
            successor = current.get_highest_valued_child()            
            if successor.fitness <= current.fitness: # looking for max
                Finish = True
                current.children = [] # erase old children
                Length_scale_inc = Length_scale_inc / 2
                Noise_level_inc = Noise_level_inc / 2
                if Length_scale_inc > Length_scale_inc_min:
                    current.length_scale_inc = Length_scale_inc
                    Finish = False
                if Noise_level_inc > Noise_level_inc_min:
                    current.noise_level_inc = Noise_level_inc
                    Finish = False
                if Finish:
                    return current, count
                else:
                    continue # with old node but smaller intervals
            else:
                current = successor
    
    print('Gaussian started') 
    X_Gaussian_train = IOfunctions.ReadCSV(Files['Gaussian Single Train'])
    X_Gaussian_test = IOfunctions.ReadCSV(Files['Gaussian Single Test'])
    Y_train = IOfunctions.ReadCSV(Files['Response Train'])
    Y_test = IOfunctions.ReadCSV(Files['Response Test']) 
    nodes = []
    node, count = hill(length_scale_start, noise_level_start)
    nodes.append(node)
    if simulation is not None:
        for i in range(0, 10, 1):
            length_scale_start = random.random() * (length_scale_bounds[1] - length_scale_bounds[0]) + length_scale_bounds[0]
            noise_level_start = random.random() * (noise_level_bounds[1] - noise_level_bounds[0]) + noise_level_bounds[0]    
            node, count = hill(length_scale_start, noise_level_start)
            nodes.append(node)
    best_node = nodes.pop(0)
    while len(nodes) > 0:
        node = nodes.pop(0)
        if node.fitness > best_node.fitness:
            best_node = node                 
    kernel = RBF(length_scale=best_node.length_scale, length_scale_bounds=length_scale_bounds) +\
        WhiteKernel(noise_level=best_node.noise_level, noise_level_bounds=noise_level_bounds)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None,\
        n_restarts_optimizer=0, normalize_y=True, copy_X_train=True,\
        random_state=random_state)
    gp.fit(X_Gaussian_train, Y_train) # set from distances    
    
    return gp

def EraseFile(F):
    try:        
        os.remove(F)
    except:
        return False
    return True

def CopyFile(F, Dir):
    if os.path.isfile(F):
        shutil.copy2(F, Dir)    
        return True
    else:
        return False

def MoveFile(F, Dir):
    if os.path.isfile(F):
        shutil.move(F, Dir)    
        return True
    else:
        return False


        