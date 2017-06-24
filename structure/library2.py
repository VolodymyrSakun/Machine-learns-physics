# Library of functions 
# argmaxabs
# argminabs
# Results_to_xls
# BackwardElimination
# SelectBestSubsetFromElasticNetPath
# CalculateVif
# ReadFeatures
# DecisionTreeEstimator
# RandomForestEstimator
# ClassifyCorrelatedFeatures
# FindAlternativeFit

import numpy as np
import pandas as pd
import sklearn.metrics as skm
import copy
import pickle 
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
def ReadFeatures(F_features, F_structure_FeaturesAll, F_structure_FeaturesReduced, verbose=False):
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
    return X, Y, FeaturesAll, FeaturesReduced

# select the best subset from elastic net path
def SelectBestSubsetFromElasticNetPath(x_std, y_std, Method='CV', MSE_threshold=None, R2_threshold=None, \
    L1_ratio=0.01, Eps=1e-3, N_alphas=200, Alphas=None, max_iter=10000, tol=0.0001, \
    cv=30, n_jobs=-1, selection='random', PlotPath=False, verbose=False):
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
        plt.figure(1, figsize = (19, 10))
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
        plt.savefig('Enet_Path', bbox_inches='tight')
    if (Method == 'CV'):
        return nonzero_idx, alpha
    if (Method == 'grid'):
        return nonzero_idx, alpha, mse_list, nonzero_count_list

# store in sheet of .xlsx file description of fit with real coefficients
def Results_to_xls(writeResults, SheetName, selected_features_list, X, Y, FeaturesAll, FeaturesReduced, split_ratio=0.1, rand_state=101):
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, random_state=rand_state)
    Size_train = len(Y_train)
    Size_test = len(Y_test)
    NofSelectedFeatures = len(selected_features_list)
    x_sel_train = np.zeros(shape=(Size_train, 1), dtype=float) # matrix with selected features
    x_sel_test = np.zeros(shape=(Size_test, 1), dtype=float) # matrix with selected features
    z_train = np.zeros(shape=(Size_train, 1), dtype=float)
    z_test = np.zeros(shape=(Size_test, 1), dtype=float)
    for i in range(0, NofSelectedFeatures, 1):
        if i == 0:
            x_sel_train[:, 0] = X_train[:, selected_features_list[i]]
            x_sel_test[:, 0] = X_test[:, selected_features_list[i]]
        else:
            x_sel_train[:, -1] = X_train[:, selected_features_list[i]]
            x_sel_test[:, -1] = X_test[:, selected_features_list[i]]
        x_sel_train = np.concatenate((x_sel_train, z_train), axis=1)
        x_sel_test = np.concatenate((x_sel_test, z_test), axis=1)
# fit selected features     
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
    lr.fit(x_sel_train, Y_train)
    coef = lr.coef_
    y_pred = lr.predict(x_sel_test)
    mse = skm.mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y_test, y_pred)
    nonzero_count = np.count_nonzero(coef)
    rSS = 0
    for i in range(0, Size_test, 1):
        rSS += (y_pred[i] - Y_test[i])**2
    aIC = 2 * nonzero_count + Size_test * np.log(rSS)
# do not forget to check for zeros in coef later    
    Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 13)).astype(float), \
        columns=['Feature index','Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2', \
        'Intermolecular 2', 'Number of distances in feature','Coefficients','MSE','RMSE','R2','AIC'], dtype=str)
    max_distances_in_feature = 1
    for i in range(0, NofSelectedFeatures, 1):
        index = selected_features_list[i]
        Results.loc[i]['Feature index'] = index
        Results.loc[i]['Bond 1'] = FeaturesReduced[index].DtP1.Distance.Atom1.Symbol + '-' + FeaturesReduced[index].DtP1.Distance.Atom2.Symbol
        Results.loc[i]['Power 1'] = FeaturesReduced[index].DtP1.Power
        if FeaturesReduced[index].DtP1.Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
            Results.loc[i]['Intermolecular 1'] = 'Yes'
        else:
            Results.loc[i]['Intermolecular 1'] = 'No'
        if FeaturesReduced[index].nDistances >= 2:
            Results.loc[i]['Bond 2'] = FeaturesReduced[index].DtP2.Distance.Atom1.Symbol + '-' + FeaturesReduced[index].DtP2.Distance.Atom2.Symbol
            Results.loc[i]['Power 2'] = FeaturesReduced[index].DtP2.Power
            if max_distances_in_feature < 2:
                max_distances_in_feature = 2
            if FeaturesReduced[index].DtP2.Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
                Results.loc[i]['Intermolecular 2'] = 'Yes'
            else:
                Results.loc[i]['Intermolecular 2'] = 'No'
        else:
            Results.loc[i]['Bond 2'] = ' '
            Results.loc[i]['Power 2'] = ' '
            Results.loc[i]['Intermolecular 2'] = ' '
        counter = 0
        current_feature_type = FeaturesReduced[index].FeType
        for j in range(0, len(FeaturesAll), 1):
            if FeaturesAll[j].FeType == current_feature_type:
                counter += 1
        Results.loc[i]['Number of distances in feature'] = counter
        Results.loc[i]['Coefficients'] = coef[i]
        if i == 0:
            Results.loc[0]['MSE'] = mse
            Results.loc[0]['RMSE'] = rmse
            Results.loc[0]['R2'] = r2
            Results.loc[0]['AIC'] = aIC
        else:
            Results.loc[i]['MSE'] = ''
            Results.loc[i]['RMSE'] = ''
            Results.loc[i]['R2'] = ''
            Results.loc[i]['AIC'] = ''
    
    if max_distances_in_feature == 1:
        del(Results['Bond 2'])
        del(Results['Power 2'])
        del(Results['Intermolecular 2'])

    Results.to_excel(writeResults, sheet_name=SheetName)
    return
# end of Results_to_xls

def BackwardElimination(X_train, Y_train, X_test, Y_test, FeaturesAll, \
    FeaturesReduced, Method='fast', Criterion='p-Value', N_of_features_to_select=20, \
    idx=None, PlotPath=False, StorePath=False, \
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
        while nonzero_count > N_of_features_to_select:
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
            for i in range(0, len(features_index), 1):
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
                drop = np.argmin(r2_array)
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
#        C = np.cov(X, rowvar=False, bias=True)
        A = pd.DataFrame(X, dtype=float)
        B = A.corr()
        B.fillna(0, inplace=True) # replace NaN with 0
        C = B.as_matrix()
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
    
def FindAlternativeFit(x_std, y_std, features_idx, classified_list, Method='MSE', verbose=False):
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
    import statsmodels.regression.linear_model as sm
    if verbose:
        print('setup initial feature set from lasso / lars / elastic net')
    active_features = copy.deepcopy(features_idx)
    x_sel = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    tmp = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
# creating selected features array
    for i in active_features:
        if i == active_features[0]:
            x_sel[:, 0] = x_std[:, i]
        else:
            tmp[:, 0] = x_std[:, i]
            x_sel = np.concatenate((x_sel, tmp), axis=1)
# initial fit
    ols = sm.OLS(endog = y_std, exog = x_sel, hasconst = False).fit()
    y_pred_ols = ols.predict(x_sel)
    mse_best = skm.mean_squared_error(y_std, y_pred_ols)
    r2_best = ols.rsquared
    if verbose:
        print('find alternatives')
    corr_list = classified_list
    results = copy.deepcopy(corr_list)
    for i in range(0, len(corr_list), 1):
        k = 0
        for j in active_features:
            x_sel[:, k] = x_std[:, j]
            k += 1
        ols = sm.OLS(endog = y_std, exog = x_sel, hasconst = False).fit()
        y_pred_ols = ols.predict(x_sel)
        mse_better = skm.mean_squared_error(y_std, y_pred_ols)
        r2_better = ols.rsquared       
        for j in range(0, len(corr_list[i]), 1):
            if (j != 0) and (corr_list[i][j] in active_features):
                results[i][j] = ((-1, -1, []))
                continue
            old = active_features[i]
            active_features[i] = corr_list[i][j]
            x_sel[:, i] = x_std[:, corr_list[i][j]]
            # fit
            ols = sm.OLS(endog = y_std, exog = x_sel, hasconst = False).fit()
            y_pred_ols = ols.predict(x_sel)
            r2_ols = ols.rsquared
            mse_ols = skm.mean_squared_error(y_std, y_pred_ols)
            results[i][j] = ((mse_ols, r2_ols, copy.deepcopy(active_features)))
            active_features[i] = old
        idx = 0
        for l in range(0, len(results[i])):
            if results[i][l][0] == -1:
                continue
            mse_new = results[i][l][0]
            r2_new = results[i][l][1]
            if Method == 'MSE':
                if mse_new < mse_better:
                    idx = l
                    mse_better = mse_new
                if mse_new < mse_best:
                    mse_best = mse_new
                    r2_best = results[i][l][1]
            if Method == 'R2':
                if r2_new > r2_better:
                    idx = l
                    r2_better = r2_new
                if r2_new > r2_best:
                    r2_best = r2_new
                    mse_best = results[i][l][0]
        new_value = corr_list[i][idx]
        if new_value in active_features:
            continue
        else:
            active_features[i] = new_value

    return active_features, results
    
    
# store in sheet of .xlsx file description of fit with real coefficients
def Results_to_xls2(writeResults, SheetName, selected_features_list, X_train, Y_train,\
        X_test, Y_test, FeaturesAll, FeaturesReduced, split_ratio=0.1, rand_state=101):
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
    Size_train = len(Y_train)
    Size_test = len(Y_test)
    NofSelectedFeatures = len(selected_features_list)
    x_sel_train = np.zeros(shape=(Size_train, 1), dtype=float) # matrix with selected features
    x_sel_test = np.zeros(shape=(Size_test, 1), dtype=float) # matrix with selected features
    z_train = np.zeros(shape=(Size_train, 1), dtype=float)
    z_test = np.zeros(shape=(Size_test, 1), dtype=float)
    for i in range(0, NofSelectedFeatures, 1):
        if i == 0:
            x_sel_train[:, 0] = X_train[:, selected_features_list[i]]
            x_sel_test[:, 0] = X_test[:, selected_features_list[i]]
        else:
            x_sel_train[:, -1] = X_train[:, selected_features_list[i]]
            x_sel_test[:, -1] = X_test[:, selected_features_list[i]]
        x_sel_train = np.concatenate((x_sel_train, z_train), axis=1)
        x_sel_test = np.concatenate((x_sel_test, z_test), axis=1)
# fit selected features     
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
    lr.fit(x_sel_train, Y_train)
    coef = lr.coef_
    y_pred = lr.predict(x_sel_test)
    mse = skm.mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y_test, y_pred)
    nonzero_count = np.count_nonzero(coef)
    rSS = 0
    for i in range(0, Size_test, 1):
        rSS += (y_pred[i] - Y_test[i])**2
    aIC = 2 * nonzero_count + Size_test * np.log(rSS)
# do not forget to check for zeros in coef later    
    Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 13)).astype(float), \
        columns=['Feature index','Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2', \
        'Intermolecular 2', 'Number of distances in feature','Coefficients','MSE','RMSE','R2','AIC'], dtype=str)
    max_distances_in_feature = 1
    for i in range(0, NofSelectedFeatures, 1):
        index = selected_features_list[i]
        Results.loc[i]['Feature index'] = index
        Results.loc[i]['Bond 1'] = FeaturesReduced[index].DtP1.Distance.Atom1.Symbol + '-' + FeaturesReduced[index].DtP1.Distance.Atom2.Symbol
        Results.loc[i]['Power 1'] = FeaturesReduced[index].DtP1.Power
        if FeaturesReduced[index].DtP1.Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
            Results.loc[i]['Intermolecular 1'] = 'Yes'
        else:
            Results.loc[i]['Intermolecular 1'] = 'No'
        if FeaturesReduced[index].nDistances >= 2:
            Results.loc[i]['Bond 2'] = FeaturesReduced[index].DtP2.Distance.Atom1.Symbol + '-' + FeaturesReduced[index].DtP2.Distance.Atom2.Symbol
            Results.loc[i]['Power 2'] = FeaturesReduced[index].DtP2.Power
            if max_distances_in_feature < 2:
                max_distances_in_feature = 2
            if FeaturesReduced[index].DtP2.Distance.isIntermolecular: # 1 = Intermolecular 0 = Intramolecular
                Results.loc[i]['Intermolecular 2'] = 'Yes'
            else:
                Results.loc[i]['Intermolecular 2'] = 'No'
        else:
            Results.loc[i]['Bond 2'] = ' '
            Results.loc[i]['Power 2'] = ' '
            Results.loc[i]['Intermolecular 2'] = ' '
        counter = 0
        current_feature_type = FeaturesReduced[index].FeType
        for j in range(0, len(FeaturesAll), 1):
            if FeaturesAll[j].FeType == current_feature_type:
                counter += 1
        Results.loc[i]['Number of distances in feature'] = counter
        Results.loc[i]['Coefficients'] = coef[i]
        if i == 0:
            Results.loc[0]['MSE'] = mse
            Results.loc[0]['RMSE'] = rmse
            Results.loc[0]['R2'] = r2
            Results.loc[0]['AIC'] = aIC
        else:
            Results.loc[i]['MSE'] = ''
            Results.loc[i]['RMSE'] = ''
            Results.loc[i]['R2'] = ''
            Results.loc[i]['AIC'] = ''
    
    if max_distances_in_feature == 1:
        del(Results['Bond 2'])
        del(Results['Power 2'])
        del(Results['Intermolecular 2'])

    Results.to_excel(writeResults, sheet_name=SheetName)
    return
# end of Results_to_xls

def StoreFeaturesDescriprion(FileName, FeaturesAll, FeaturesReduced):
    writeResults = pd.ExcelWriter(FileName, engine='openpyxl')
    NofSelectedFeatures = len(FeaturesReduced)
    selected_features_list = list(range(0,NofSelectedFeatures,1))
    Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 11)).astype(float), \
        columns=['Feature index','Feature type','Bond 1','Power 1','Intermolecular 1','Distance 1 type','Bond 2','Power 2', \
        'Intermolecular 2', 'Distance 2 type','Number of distances in feature'], dtype=str)
    max_distances_in_feature = 1
    for i in range(0, NofSelectedFeatures, 1):
        index = selected_features_list[i]
        Results.loc[i]['Feature index'] = index
        Results.loc[i]['Feature type'] = FeaturesReduced[index].FeType
        Results.loc[i]['Bond 1'] = FeaturesReduced[index].DtP1.Distance.Atom1.Symbol + '-' + FeaturesReduced[index].DtP1.Distance.Atom2.Symbol
        Results.loc[i]['Power 1'] = FeaturesReduced[index].DtP1.Power
        Results.loc[i]['Distance 1 type'] = FeaturesReduced[index].DtP1.DtpType
        if FeaturesReduced[index].DtP1.Distance.isIntermolecular: # True = Intermolecular False = Intramolecular
            Results.loc[i]['Intermolecular 1'] = 'Yes'
        else:
            Results.loc[i]['Intermolecular 1'] = 'No'
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
        FeaturesReduced[0].FeType
        FeaturesReduced[0].DtP1.DtpType
    if max_distances_in_feature == 1:
        del(Results['Bond 2'])
        del(Results['Power 2'])
        del(Results['Intermolecular 2'])
        del(Results['Distance 2 type'])
    Results.to_excel(writeResults)
    writeResults.save()  
    return

def get_fit_score(X_train, X_test, Y_train, Y_test, idx=None):
    Size_train = len(Y_train)
    Size_test = len(Y_test)
    if idx is None:
        x_sel_train = X_train
        x_sel_test = X_test
    else:
        NofSelectedFeatures = len(idx)
        x_sel_train = np.zeros(shape=(Size_train, 1), dtype=float) # matrix with selected features
        x_sel_test = np.zeros(shape=(Size_test, 1), dtype=float) # matrix with selected features
        z_train = np.zeros(shape=(Size_train, 1), dtype=float)
        z_test = np.zeros(shape=(Size_test, 1), dtype=float)
        for i in range(0, NofSelectedFeatures, 1):
            if i == 0:
                x_sel_train[:, 0] = X_train[:, idx[i]]
                x_sel_test[:, 0] = X_test[:, idx[i]]
            else:
                x_sel_train[:, -1] = X_train[:, idx[i]]
                x_sel_test[:, -1] = X_test[:, idx[i]]
            x_sel_train = np.concatenate((x_sel_train, z_train), axis=1)
            x_sel_test = np.concatenate((x_sel_test, z_test), axis=1)
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
    lr.fit(x_sel_train, Y_train)
    coef = lr.coef_
    y_pred = lr.predict(x_sel_test)
    mse = skm.mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y_test, y_pred)
    nonzero_count = np.count_nonzero(coef)
    rSS = 0
    for i in range(0, Size_test, 1):
        rSS += (y_pred[i] - Y_test[i])**2
    aIC = 2 * nonzero_count + Size_test * np.log(rSS)
    return nonzero_count, mse, rmse, r2, aIC

def plot_rmse(FileName, nonzero_count_list, rmse_OLS_list, rsquared_list):
# Plot RMSE vs. active coefficients number        
    plt.figure(100, figsize = (19, 10))
    plt.plot(nonzero_count_list, rmse_OLS_list, ':')
    plt.xlabel('Active coefficients')
    plt.ylabel('Root of mean square error')
    plt.title('Root of mean square error vs Active coefficiants')
    plt.axis('tight')
    plt.savefig(FileName + '_RMSE', bbox_inches='tight')
    # Plot R2 vs. active coefficiens number
    plt.figure(101, figsize = (19, 10))
    plt.plot(nonzero_count_list, rsquared_list, ':')
    plt.xlabel('Active coefficients')
    plt.ylabel('R2')
    plt.title('R2 vs Active coefficiants')
    plt.axis('tight')
    plt.savefig(FileName + '_R2', bbox_inches='tight')
    return

def Fit(x_std, y_std, idx_cpu, active_features, corr_list):# local index
    x_sel = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    tmp = np.zeros(shape=(x_std.shape[0], 1), dtype=float)
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
# creating selected features array
    for i in active_features:
        if i == active_features[0]:# first column
            x_sel[:, 0] = x_std[:, i]
        else:
            tmp[:, 0] = x_std[:, i]
            x_sel = np.concatenate((x_sel, tmp), axis=1) 
    idx_list = []
    mse_list = []
    for i in idx_cpu:# i - local index
        mse_small_list = []
        idx_small_list = []
        corr_horiz = corr_list[i]
        for j in corr_horiz: # j - global index
            tmp[:, 0] = x_sel[:, i]
            x_sel[:, i] = x_std[:, j]
            lr.fit(x_sel, y_std)
            y_pred = lr.predict(x_sel)
            mse = skm.mean_squared_error(y_std, y_pred)
            mse_small_list.append(mse)
            idx_small_list.append(j) # global index
            x_sel[:, i] = tmp[:, 0]
        idx = np.argmin(mse_small_list) # local index of best mse
        mse = mse_small_list[idx] # best mse for one feature in corr list
        idx_list.append(idx_small_list[idx]) # global index of best mse
        mse_list.append(mse)
    return [idx_list, mse_list]

def FindBestSet(x_std, y_std, features_idx, corr_list, Method='MSE', verbose=False):
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
    nFeatures = len(active_features)
    idx_list = list(range(0, nFeatures, 1))
    while Found:
        Found = False
        output = Fit(x_std, y_std, idx_list, active_features, corr_list)
        for j in range(0, len(output[0]), 1):
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
    j = 0 # 0 .. nFeatures
    idx_list = []
    while j < nFeatures:
        idx_small = []
        for i in range(0, nFeatures_per_CPU, 1): # 0 .. nFeatures_per_CPU
            if j < nFeatures:
                idx_small.append(j)
            j += 1
        idx_list.append(idx_small)
    while Found:
        Found = False
        ran = list(range(0, nCPU, 1))
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








