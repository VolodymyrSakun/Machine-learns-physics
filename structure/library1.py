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

# store in .xlsx file fit description
def Results_to_xls(writeResults, SheetName, selected_features_list, X, Y, FeaturesAll, FeaturesReduced):
# writeResults = pd.ExcelWriter('FileName.xlsx', engine='openpyxl')
# after calling this function(s) data must be stored in the file by calling writeResults.save()
# SheetName - name of xlsx sheet
# selected_features_list - indices of features to proceed
# X - [n x m] numpy array of features
# rows correspond to observations
# columns correspond to features
# Y - [n x 1] numpy array, recponse variable
# FeaturesAll - class1.InvPowDistancesFeature object. Contains all features
# FeaturesReduced - class1.InvPowDistancesFeature object. Contains combined features
    from sklearn.linear_model import LinearRegression
    import sklearn.metrics as skm
    Size = len(Y)
    NofSelectedFeatures = len(selected_features_list)
    x_selected = np.zeros(shape=(Size, 1), dtype=float) # matrix with selected features
    z = np.zeros(shape=(Size, 1), dtype=float)
    for i in range(0, NofSelectedFeatures, 1):
        if i == 0:
            x_selected[:, 0] = X[:, selected_features_list[i]]
        else:
            x_selected[:, -1] = X[:, selected_features_list[i]]
        x_selected = np.concatenate((x_selected, z), axis=1)
# fit selected features     
    lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
    lr.fit(x_selected, Y)
    coef = lr.coef_
    y_pred = lr.predict(x_selected)
    mse = skm.mean_squared_error(Y, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(Y, y_pred)
    nonzero_count = np.count_nonzero(coef)
    rSS = 0
    for i in range(0, Size, 1):
        rSS += (y_pred[i] - Y[i])**2
    aIC = 2 * nonzero_count + Size * np.log(rSS)
# do not forget to check for zeros in coef later    
    Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 14)).astype(float), \
        columns=['Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2', \
        'Intermolecular 2','Bond 3','Power 3','Intermolecular 3', \
        'Number of distances in feature','Coefficient', 'RMSE', 'R2','AIC'], dtype=float)
    max_distances_in_feature = 1
    for i in range(0, NofSelectedFeatures, 1):
        index = selected_features_list[i]
        Results.iloc[i, 0] = FeaturesReduced[index].distances[0].Atom1.Symbol + '-' + FeaturesReduced[index].distances[0].Atom2.Symbol
        Results.iloc[i, 1] = FeaturesReduced[index].powers[0]
        if FeaturesReduced[index].distances[0].isIntermolecular != 0: # 1 = Intermolecular 0 = Intramolecular
            Results.iloc[i, 2] = 'Yes'
        else:
            Results.iloc[i, 2] = 'No'
        if FeaturesReduced[index].nDistances >= 2:
            Results.iloc[i, 3] = FeaturesReduced[index].distances[1].Atom1.Symbol + '-' + FeaturesReduced[index].distances[1].Atom2.Symbol
            Results.iloc[i, 4] = FeaturesReduced[index].powers[1]
            if max_distances_in_feature < 2:
                max_distances_in_feature = 2
            if FeaturesReduced[index].distances[1].isIntermolecular != 0: # 1 = Intermolecular 0 = Intramolecular
                Results.iloc[i, 5] = 'Yes'
            else:
                Results.iloc[i, 5] = 'No'
        else:
            Results.iloc[i, 3] = ' '
            Results.iloc[i, 4] = ' '
            Results.iloc[i, 5] = ' '
        if FeaturesReduced[index].nDistances == 3:
            if max_distances_in_feature < 3:
                max_distances_in_feature = 3
            Results.iloc[i, 6] = FeaturesReduced[index].distances[2].Atom1.Symbol + '-' + FeaturesReduced[index].distances[2].Atom2.Symbol
            Results.iloc[i, 7] = FeaturesReduced[index].powers[2]
            if FeaturesReduced[index].distances[2].isIntermolecular != 0: # 1 = Intermolecular 0 = Intramolecular
                Results.iloc[i, 8] = 'Yes'
            else:
                Results.iloc[i, 8] = 'No'
        else:
            Results.iloc[i, 6] = ' '
            Results.iloc[i, 7] = ' '
            Results.iloc[i, 8] = ' '
        counter = 0
        current_feature_type = FeaturesReduced[index].FeType
        for j in range(0, len(FeaturesAll), 1):
            if FeaturesAll[j].FeType == current_feature_type:
                counter += 1
        Results.iloc[i, -5] = counter
        Results.iloc[i, -4] = coef[i]
        if i == 0:
            Results.iloc[0, -3] = rmse
            Results.iloc[0, -2] = r2
            Results.iloc[0, -1] = aIC
        else:
            Results.iloc[i, -3] = ''
            Results.iloc[i, -2] = ''
            Results.iloc[i, -1] = ''
    
    if max_distances_in_feature <= 2:
        del(Results['Bond 3'])
        del(Results['Power 3'])
        del(Results['Intermolecular 3'])
    if max_distances_in_feature == 1:
        del(Results['Bond 2'])
        del(Results['Power 2'])
        del(Results['Intermolecular 2'])

    Results.to_excel(writeResults, sheet_name=SheetName)
    return
# end of Results_to_xls

def BackwardElimination(X, Y, FeaturesAll, FeaturesReduced, Method='fast', Criterion='p-Value', N_of_features_to_select=20, \
    idx=None, PlotPath=False, StorePath=False, \
    FileName=None, N_of_last_iterations_to_store=10, verbose=False):
# returns list of indices of selected features
# idx should contain approximatelly 200 indices
# X - numpy array, full set of nonstandartized features
# rows - observations
# columns - variables
# Y - numpy array of response
# FeaturesAll - class1.InvPowDistancesFeature object. Contains all features
# FeaturesReduced - class1.InvPowDistancesFeature object. Contains combined features
# Method = 'fast' selects feature to drop based on greatest p-Value or smallest coefficient
# Criterion = 'coef' based on dropping the feature wich has smallest standartised coeffitient
# coef method is not relevent for this type of problem
# Criterion = 'p-Value' based on dropping the feature wich has greatest p-Value
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
    from sklearn.preprocessing import StandardScaler
    import statsmodels.regression.linear_model as sm
    import sklearn.metrics as skm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    
    if PlotPath and (FileName is None):
        print('Please indicate name of file to store data and / or graphs')
        return -2
    if (FileName is not None) and StorePath:    
        writeResults = pd.ExcelWriter(FileName + '.xlsx')
    else:
        writeResults = None
# features_index - list of indices of selected features
    if idx is None:
        idx = np.linspace(start = 0, stop = X.shape[1]-1, num = X.shape[1], endpoint = True, dtype=int)
        features_index = list(range(0, X.shape[1], 1))
        del(idx)
    else:
        features_index = list(idx)
    x_sel = np.zeros(shape=(X.shape[0], len(features_index))) # create working array of features 
    j = 0
    for i in features_index:
        x_sel[:, j] = X[:, i]
        j += 1
    x_std = StandardScaler(with_mean=True, with_std=True).fit_transform(x_sel) # standardised preselected features
    y_std = StandardScaler(with_mean=True, with_std=False).fit_transform(Y)
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
    Size = y_std.shape[0]
    if verbose:
        print('Start removing features')
    if Method == 'fast':
        while (max(pvalues) != 0) and (len(ols_coef) > 1) and (nonzero_count > N_of_features_to_select):
            ols = sm.OLS(endog = y_std, exog = x_std, hasconst = False).fit()
            y_pred = ols.predict(x_std)
            ols_coef = ols.params
            nonzero_count = np.count_nonzero(ols_coef)
            if verbose:
                print('Features left in model: ', nonzero_count)
            pvalues = ols.pvalues
            rsquared = ols.rsquared
            mse_OLS = skm.mean_squared_error(y_std, y_pred)
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
            x_std = np.delete(x_std, drop, 1) # last 1 = column, 0 = row
            del(features_index[drop])
        # end while
    # end if
    if Method == 'sequential':
        lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=-1)
        while nonzero_count > N_of_features_to_select:
            lr.fit(x_std, y_std)
            lr_coef = lr.coef_
            nonzero_count = np.count_nonzero(lr_coef)
            if verbose:
                print('Features left in model: ', nonzero_count)
            y_pred = lr.predict(x_std)
            mse = skm.mean_squared_error(y_std, y_pred)
            rmse = np.sqrt(mse)
            r2 = skm.r2_score(y_std, y_pred)
            rSS_ols = 0
            for m in range(0, Size, 1):
                rSS_ols += (y_pred[m] - y_std[m])**2
            aIC_ols = 2 * nonzero_count + Size * np.log(rSS_ols)
            z = np.zeros(shape=(Size, 1), dtype=float) # always zeros
            tmp = np.zeros(shape=(Size, 1), dtype=float) # temporary storage of one feature
            mse_array = np.zeros(shape=(len(features_index)), dtype=float)
            r2_array = np.zeros(shape=(len(features_index)), dtype=float)
            aIC_array = np.zeros(shape=(len(features_index)), dtype=float)
            rmse_OLS_list.append(rmse)
            rsquared_list.append(r2*100)
            aIC_list.append(aIC_ols)
            ols_coef_list.append(lr_coef)
            nonzero_count_list.append(nonzero_count)
            for i in range(0, len(features_index), 1):
                tmp[:, 0] = x_std[:, i]
                x_std[:, i] = z[:, 0]
                lr.fit(x_std, y_std)
                y_pred = lr.predict(x_std)
                mse = skm.mean_squared_error(y_std, y_pred)
                r2 = skm.r2_score(y_std, y_pred)
                mse_array[i] = mse
                r2_array[i] = r2
                rSS_ols = 0
                for m in range(0, Size, 1):
                    rSS_ols += (y_pred[m] - y_std[m])**2
                aIC_ols = 2 * nonzero_count + Size * np.log(rSS_ols)
                aIC_array[i] = aIC_ols
                x_std[:, i] = tmp[:, 0]
            # end for
            if Criterion == 'MSE':
                drop = np.argmin(mse_array)
            if Criterion == 'R2':
                drop = np.argmin(r2_array)
            if Criterion == 'AIC':
                drop = np.argmin(aIC_array)
            drop_order.append(features_index[drop])
            x_std = np.delete(x_std, drop, 1) # last 1 = column, 0 = row
            del(features_index[drop])
        # end of while
    # end of if
    idx_return = copy.deepcopy(features_index)
    if (N_of_last_iterations_to_store is not None) and (writeResults is not None):
        if N_of_last_iterations_to_store > len(ols_coef_list):
            N_of_last_iterations_to_store = len(ols_coef_list)
        count = 0
        features_index.append(drop_order[len(drop_order)-1-count])
        while count < N_of_last_iterations_to_store:
            nonzero_count = len(features_index)
            nonzero_count_str = str(nonzero_count)
            Table = pd.DataFrame(np.zeros(shape = (nonzero_count, 13)).astype(float), columns=['Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2','Intermolecular 2','Bond 3','Power 3','Intermolecular 3', 'Number of distances in feature','Normalized coefficient','RMSE','R2'], dtype=float)
            max_distances_in_feature = 1
            j = 0 # index of reduced set
            for i in features_index: # index of full set
                Table.iloc[j, 0] = FeaturesReduced[i].distances[0].Atom1.Symbol + '-' + FeaturesReduced[i].distances[0].Atom2.Symbol
                Table.iloc[j, 1] = FeaturesReduced[i].powers[0]
                if FeaturesReduced[i].distances[0].isIntermolecular != 0: # 1 = Yes 0 = No
                    Table.iloc[j, 2] = 'Yes'
                else:
                    Table.iloc[j, 2] = 'No'
                if FeaturesReduced[i].nDistances >= 2:
                    Table.iloc[j, 3] = FeaturesReduced[i].distances[1].Atom1.Symbol + '-' + FeaturesReduced[i].distances[1].Atom2.Symbol
                    Table.iloc[j, 4] = FeaturesReduced[i].powers[1]
                    if max_distances_in_feature < 2:
                        max_distances_in_feature = 2
                    if FeaturesReduced[i].distances[1].isIntermolecular != 0: # 1 = Yes 0 = No
                        Table.iloc[j, 5] = 'Yes'
                    else:
                        Table.iloc[j, 5] = 'No'
                else:
                    Table.iloc[j, 3] = ' '
                    Table.iloc[j, 4] = ' '
                    Table.iloc[j, 5] = ' '
                if FeaturesReduced[i].nDistances == 3:
                    if max_distances_in_feature < 3:
                        max_distances_in_feature = 3
                    Table.iloc[j, 6] = FeaturesReduced[i].distances[2].Atom1.Symbol + '-' + FeaturesReduced[i].distances[2].Atom2.Symbol
                    Table.iloc[j, 7] = FeaturesReduced[i].powers[2]
                    if FeaturesReduced[i].distances[2].isIntermolecular != 0: # 1 = Yes 0 = No
                        Table.iloc[j, 8] = 'Yes'
                    else:
                        Table.iloc[j, 8] = 'No'
                else:
                    Table.iloc[j, 6] = ' '
                    Table.iloc[j, 7] = ' '
                    Table.iloc[j, 8] = ' '
                counter = 0
                current_feature_type = FeaturesReduced[i].FeType
                for k in range(0, len(FeaturesAll), 1):
                    if FeaturesAll[k].FeType == current_feature_type:
                        counter += 1
                Table.iloc[j, 9] = counter
                Table.iloc[j, 10] = ols_coef_list[len(ols_coef_list)-1-count][j]
                if j == 0:
                    Table.iloc[0, 11] = rmse_OLS_list[len(rmse_OLS_list)-1-count]
                    Table.iloc[0, 12] = rsquared_list[len(rsquared_list)-1-count]
                else:
                    Table.iloc[j, 11] = ''
                    Table.iloc[j, 12] = ''
                j += 1
            # end of for
            if max_distances_in_feature <= 2:
                del(Table['Bond 3'])
                del(Table['Power 3'])
                del(Table['Intermolecular 3'])
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
        plt.figure(1, figsize = (19, 10))
        plt.plot(nonzero_count_list, rmse_OLS_list, ':')
        plt.xlabel('Active coefficients')
        plt.ylabel('Root of mean square error')
        plt.title('Backward elimination. Root of mean square error vs Active coefficiants')
        plt.axis('tight')
        plt.savefig(FileName + '_RMSE', bbox_inches='tight')
        # Plot R2 vs. active coefficiens number
        plt.figure(2, figsize = (19, 10))
        plt.plot(nonzero_count_list, rsquared_list, ':')
        plt.xlabel('Active coefficients')
        plt.ylabel('R2')
        plt.title('Backward elimination. R2 vs Active coefficiants')
        plt.axis('tight')
        plt.savefig(FileName + '_R2', bbox_inches='tight')

    Results = pd.DataFrame(np.zeros(shape = (len(nonzero_count_list), 1)).astype(float), columns=['Empty'], dtype=float)
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

# select the best subset from elastic net path
def SelectBestSubsetFromElasticNetPath(coefs, x_std, y_std):
# returns list of indices of selected features
    from sklearn.linear_model import LinearRegression
    import sklearn.metrics as skm    
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
    mse_list = []
    for i in range(0, coefs.shape[1], 1): # columns
        nonzero_idx = []
        for j in range(0, coefs.shape[0], 1):
            if coefs[j, i] != 0:
                nonzero_idx.append(j)
        if len(nonzero_idx) == 0:
            continue
        x = np.zeros(shape=(x_std.shape[0], len(nonzero_idx)), dtype=float)
        x[:, :] = x_std[:, nonzero_idx]
        lr.fit(x, y_std)
        y_pred_lr = lr.predict(x)
        mse_lr = skm.mean_squared_error(y_std, y_pred_lr)
        mse_list.append(mse_lr)
    idx = np.argmin(mse_list)
    nonzero_idx = []
    for j in range(0, coefs.shape[0], 1):
        if coefs[j, idx] != 0:
            nonzero_idx.append(j)
    return nonzero_idx

# Calculate variance inflation factor for all features
def CalculateVif(X):
# returns dataframe with variance inflation factors
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif = vif.round(1)
    return vif

# reads feature set and objects that describes it
def ReadFeatures(F_features, F_structure_FeaturesAll, F_structure_FeaturesReduced, verbose=False):
# returns:
# X - [n x m] numpy array of features
# rows correspond to observations
# columns correspond to features
# Y - [n x 1] numpy array, recponse variable
# x_std and y_std - corresponding standardized features and response
# FeaturesAll - class1.InvPowDistancesFeature object. Contains all features
# FeaturesReduced - class1.InvPowDistancesFeature object. Contains combined features
    import pickle 
    from sklearn.preprocessing import StandardScaler
    if verbose:
        print('Reading data from file')
    dataset = pd.read_csv(F_features)
    NofFeatures = dataset.shape[1] - 1 # read numder of distances from dataset
    X = dataset.iloc[:, 0:NofFeatures].values
    Y = dataset.iloc[:, -1].values
    del dataset['energy']
# load reduced features and energy from file
    f = open(F_structure_FeaturesReduced, "rb")
    FeaturesReduced = pickle.load(f)
    f.close()
# load list FeaturesAll from file
    f = open(F_structure_FeaturesAll, "rb")
    FeaturesAll = pickle.load(f)
    f.close()
# scale
    x_std = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    y_std = StandardScaler(with_mean=True, with_std=False).fit_transform(Y)
    return X, Y, x_std, y_std, FeaturesAll, FeaturesReduced
    
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
    return idx


def RandomForestEstimator(x_std, y_std, MaxFeatures=10, NofTrees=10, verbose=False):
# returns list of max_features elements of most important features
    from sklearn.ensemble import RandomForestRegressor
    if verbose:
        print('Random Forest Estimator. Number of features to return = ', MaxFeatures)
    r2 = RandomForestRegressor(n_estimators=NofTrees, criterion='mse', max_depth=None, \
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
        max_features=MaxFeatures, max_leaf_nodes=None, min_impurity_split=1e-27, \
        bootstrap=True, oob_score=False, n_jobs=-1, random_state=101, verbose=0, \
        warm_start=False)
    r2.fit(x_std, y_std)
    feature_importance_forest = r2.feature_importances_
    idx = []
    for i in range(0, MaxFeatures, 1):
        tmp = np.argmax(feature_importance_forest)
        idx.append(tmp)
    return idx

def ClassifyCorrelatedFeatures(X, features_idx, MinCorrelation, Model=1, verbose=False):
# creates list of correlated features 
# X - [n x m] numpy array of features
# rows correspond to observations
# columns correspond to features
# features_idx - indices of selected features
# correlation value is given by MinCorrelation
# Model 1 and 2 available

    def InList(List, Value):
# returns True if Value is in List        
        for i in range(0, len(List), 1):
            for j in range(0, len(List[i]), 1):
                if List[i][j] == Value:
                    return True
        return False
    
    if verbose:
        print('Calculate correlation matrix')
    A = pd.DataFrame(X, dtype=float)
    B = A.corr()
    B.fillna(0, inplace=True) # replace NaN with 0
    C = B.as_matrix()
    features_lars1 = []
    features_lars2 = []
    features_lars_idx = list(features_idx)
    for i in range(0, len(features_lars_idx), 1):
        idx = features_lars_idx[i]
        features_lars1.append(list(''))
        features_lars2.append(list(''))
        features_lars1[i].append(idx)
        features_lars2[i].append(idx)
    NofFeatures = X.shape[1]
    k = 0
    for i in features_lars_idx:
        for j in range(0, NofFeatures, 1):
            if i == j:
                continue
            if C[i, j] > MinCorrelation:
                if j not in features_lars1:
                    features_lars1[k].append(j)
                if not InList(features_lars2, j):
                    features_lars2[k].append(j)
        k += 1
    if Model == 1:
        return features_lars1
    if Model == 2:
        return features_lars2
    
def FindAlternativeFit(x_std, y_std, features_idx, features_lars1, Fit=1, Method='MSE', verbose=False):
# finds alternative fit using classified feature list produced by ClassifyCorrelatedFeatures  
# returns list of indices of features
# x_std - [n x m] numpy array of standardized features
# rows correspond to observations
# columns correspond to features
# y_std - [n x 1] numpy array, standardized recponse variable
# features_idx - indices of selected features
# features_lars1 - list of classified features
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
    mse_ols = skm.mean_squared_error(y_std, y_pred_ols)
    if verbose:
        print('find alternatives')
    features_lars = features_lars1
    results = copy.deepcopy(features_lars)
    mse_best = 1e+100
    r2_best = 0
    for i in range(0, len(features_lars), 1):
        old_value = active_features[i]
        for j in range(0, len(features_lars[i]), 1):
            active_features[i] = features_lars[i][j]
            x_sel[:, i] = x_std[:, active_features[i]]
            # fit
            ols = sm.OLS(endog = y_std, exog = x_sel, hasconst = False).fit()
            y_pred_ols = ols.predict(x_sel)
            r2_ols = ols.rsquared
            mse_ols = skm.mean_squared_error(y_std, y_pred_ols)
            results[i][j] = ((mse_ols, r2_ols, copy.deepcopy(active_features)))
        mse_better = 1e+100
        r2_better = 0
        idx = 0
        for l in range(0, len(results[i])):
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
        if Fit == 1:
            active_features[i] = features_lars[i][idx]
        if Fit == 2:
            active_features[i] = old_value # version 2
    return active_features
    
    
