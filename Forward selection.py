# Performs forward feature selection from the feature set
# Relatively fast

import sys
import os
Current_dir = os.getcwd()
sys.path.append(Current_dir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pickle 
from structure import class1 # contains necessary structures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import statsmodels.regression.linear_model as sm

MaxFeatures = 20 # desired number of variables for final fit
F_features = 'Features and energy two distances reduced.csv' # input csv file with combined features and energy
F_structure_FeaturesReduced = 'FeaturesReduced.dat' # output data structure which contains combined features
F_structure_FeaturesAll = 'FeaturesAll.dat' # output data structure which contains all features
F_img1 = 'RMSE_OLS'
F_img2 = 'R2_OLS'
F_xlsx = 'Forward selection.xlsx'
writeResults = pd.ExcelWriter(F_xlsx)
dataset = pd.read_csv(F_features)
NofFeatures = dataset.shape[1] - 1 # read numder of distances from dataset
X = dataset.iloc[:, 0:NofFeatures].values
y = dataset.iloc[:, -1].values
Size = len(y)
del dataset['energy']
ColumnNames = list(dataset.columns)
# load reduced features and energy from file
f = open(F_structure_FeaturesReduced, "rb")
FeaturesReduced = pickle.load(f)
f.close()
# load list FeaturesAll from file
f = open(F_structure_FeaturesAll, "rb")
FeaturesAll = pickle.load(f)
f.close()

# Least squared for all features 

ols = sm.OLS(endog = y, exog = X, hasconst = False).fit()
y_pred_ols = ols.predict(X)
coef_ols = ols.params
pvalues_ols = ols.pvalues
r2_ols = ols.rsquared
mse_ols = skm.mean_squared_error(y, y_pred_ols)
rmse_ols = np.sqrt(mse_ols)
nonzero_count_ols = np.count_nonzero(coef_ols)
rSS_ols = 0
for i in range(0, Size, 1):
    rSS_ols += (y_pred_ols[i] - y[i])**2
aIC_ols = 2 * nonzero_count_ols + Size * np.log(rSS_ols)
ResultsOLS = pd.DataFrame(np.zeros(shape = (6, 2)).astype(float), dtype=float)
ResultsOLS.iloc[0, 0] = 'Ordinary least squared'
ResultsOLS.iloc[1, 0] = 'Nonzero coefficients'
ResultsOLS.iloc[2, 0] = 'MSE'
ResultsOLS.iloc[3, 0] = 'RMSE'
ResultsOLS.iloc[4, 0] = 'R2'
ResultsOLS.iloc[5, 0] = 'AIC criterion'
ResultsOLS.iloc[0, 1] = ' '
ResultsOLS.iloc[1, 1] = nonzero_count_ols
ResultsOLS.iloc[2, 1] = mse_ols
ResultsOLS.iloc[3, 1] = rmse_ols
ResultsOLS.iloc[4, 1] = r2_ols
ResultsOLS.iloc[5, 1] = aIC_ols

ResultsOLS.to_excel(writeResults,'OLS no const', index=None, header=None)

# Least squared for all features 

lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=1)
lr.fit(X, y)
y_pred_lr = lr.predict(X)
coef_lr = lr.coef_
r2_lr = skm.r2_score(y, y_pred_lr)
mse_lr = skm.mean_squared_error(y, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
nonzero_count_lr = np.count_nonzero(coef_lr)
rSS_lr = 0
for i in range(0, Size, 1):
    rSS_lr += (y_pred_lr[i] - y[i])**2
aIC_lr = 2 * nonzero_count_lr + Size * np.log(rSS_lr)
ResultsOLS = pd.DataFrame(np.zeros(shape = (6, 2)).astype(float), dtype=float)
ResultsOLS.iloc[0, 0] = 'Ordinary least squared'
ResultsOLS.iloc[1, 0] = 'Nonzero coefficients'
ResultsOLS.iloc[2, 0] = 'MSE'
ResultsOLS.iloc[3, 0] = 'RMSE'
ResultsOLS.iloc[4, 0] = 'R2'
ResultsOLS.iloc[5, 0] = 'AIC criterion'
ResultsOLS.iloc[0, 1] = ' '
ResultsOLS.iloc[1, 1] = nonzero_count_lr
ResultsOLS.iloc[2, 1] = mse_lr
ResultsOLS.iloc[3, 1] = rmse_lr
ResultsOLS.iloc[4, 1] = r2_lr
ResultsOLS.iloc[5, 1] = aIC_lr

ResultsOLS.to_excel(writeResults,'LR no const', index=None, header=None)
"""
# Ridge

ridge = Ridge(alpha=1.0, fit_intercept=False, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
ridge.fit(X, y)
y_pred_ridge = ridge.predict(X)
coef_ridge = ridge.coef_
r2_ridge = skm.r2_score(y, y_pred_ridge)
mse_ridge = skm.mean_squared_error(y, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
nonzero_count_ridge = np.count_nonzero(coef_ridge)
rSS_ridge = 0
for i in range(0, Size, 1):
    rSS_ridge += (y_pred_ridge[i] - y[i])**2
aIC_ridge = 2 * nonzero_count_ridge + Size * np.log(rSS_ridge)
ResultsOLS = pd.DataFrame(np.zeros(shape = (6, 2)).astype(float), dtype=float)
ResultsOLS.iloc[0, 0] = 'Ridge'
ResultsOLS.iloc[1, 0] = 'Nonzero coefficients'
ResultsOLS.iloc[2, 0] = 'MSE'
ResultsOLS.iloc[3, 0] = 'RMSE'
ResultsOLS.iloc[4, 0] = 'R2'
ResultsOLS.iloc[5, 0] = 'AIC criterion'
ResultsOLS.iloc[0, 1] = ' '
ResultsOLS.iloc[1, 1] = nonzero_count_ridge
ResultsOLS.iloc[2, 1] = mse_ridge
ResultsOLS.iloc[3, 1] = rmse_ridge
ResultsOLS.iloc[4, 1] = r2_ridge
ResultsOLS.iloc[5, 1] = aIC_ridge

ResultsOLS.to_excel(writeResults,'Ridge', index=None, header=None)
"""

selected_features_list = [] # list of indices of selected features
features_index = np.linspace(start = 0, stop = NofFeatures-1, num = NofFeatures, endpoint = True, dtype=int) # list of remaining reduced features
rmse_array = np.zeros(shape=(NofFeatures), dtype=float)
r2_array = np.zeros(shape=(NofFeatures), dtype=float)
x_selected = np.zeros(shape=(Size, 1), dtype=float) # matrix with selected features
# Based on best fit using one variable
print("Start fitting")
lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
rmse_list = []
r2_list = []
for i in range(0, NofFeatures, 1):
    x_selected[:, 0] = X[:, i]
    lr.fit(x_selected, y)
    coef_lr = lr.coef_
    y_pred = lr.predict(x_selected)
    mse = skm.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = skm.r2_score(y, y_pred)
    rmse_array[i] = rmse
    r2_array[i] = r2
    
m = np.argmin(rmse_array)
rmse_list.append(rmse_array[m])
r2_list.append(r2_array[m])
selected_features_list.append(features_index[m]) # add index of selected feature to list
x_selected[:, 0] = X[:, m] # add selected feature to matrix
X = np.delete(X, m, axis=1) # remove selected feature from feature matrix
features_index = np.delete(features_index, m) # remove selected feature from list of all features
j = 1
while j < MaxFeatures:
    rmse_array = np.zeros(shape=(X.shape[1]), dtype=float)
    r2_array = np.zeros(shape=(X.shape[1]), dtype=float)
    z = np.zeros(shape=(Size, 1), dtype=float)
    x_selected = np.concatenate((x_selected, z), axis=1)
    for i in range(0, X.shape[1], 1):
        x_selected[:, -1] = X[:, i]
        lr.fit(x_selected, y)
        coef_lr = lr.coef_
        y_pred = lr.predict(x_selected)
        mse = skm.mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = skm.r2_score(y, y_pred)
        rmse_array[i] = rmse
        r2_array[i] = r2
    m = np.argmin(rmse_array) # index of feature in current matrix with smallest RMSE
    rmse_list.append(rmse_array[m])
    r2_list.append(r2_array[m])
    selected_features_list.append(features_index[m]) # add index of selected feature to list
    x_selected[:, -1] = X[:, m] # add selected feature to matrix
    X = np.delete(X, m, axis=1)
    features_index = np.delete(features_index, m)
    j += 1
    print(j)
    
    
NofSelectedFeatures = len(selected_features_list)
order = np.linspace(1, NofSelectedFeatures, NofSelectedFeatures, endpoint=True, dtype=int)
plt.figure(1, figsize = (19, 10))
plt.plot(order, rmse_list, ':')
plt.xlabel('Active coefficients')
plt.ylabel('Root of mean squared error')
plt.title('Forward selection. Root of mean squared error vs Active coefficiants')
plt.axis('tight')
plt.savefig(F_img1, bbox_inches='tight')

plt.figure(2, figsize = (19, 10))
plt.plot(order, r2_list, ':')
plt.xlabel('Active coefficients')
plt.ylabel('R2')
plt.title('Forward selection. R2 vs Active coefficiants')
plt.axis('tight')
plt.savefig(F_img2, bbox_inches='tight')

Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 14)).astype(float), columns=['Selected','Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2','Intermolecular 2','Bond 3','Power 3','Intermolecular 3', 'Number of distances in feature','Coefficient', 'RMSE', 'R2'], dtype=float)
max_distances_in_feature = 1
for i in range(0, NofSelectedFeatures, 1):
    Results.iloc[i, 0] = i+1
    index = selected_features_list[i]
    Results.iloc[i, 1] = FeaturesReduced[index].distances[0].Atom1.Symbol + '-' + FeaturesReduced[index].distances[0].Atom2.Symbol
    Results.iloc[i, 2] = FeaturesReduced[index].powers[0]
    if FeaturesReduced[index].distances[0].isIntermolecular != 0: # 1 = Intermolecular 0 = Intramolecular
        Results.iloc[i, 3] = 'Yes'
    else:
        Results.iloc[i, 3] = 'No'
    if FeaturesReduced[index].nDistances >= 2:
        Results.iloc[i, 4] = FeaturesReduced[index].distances[1].Atom1.Symbol + '-' + FeaturesReduced[index].distances[1].Atom2.Symbol
        Results.iloc[i, 5] = FeaturesReduced[index].powers[1]
        if max_distances_in_feature < 2:
            max_distances_in_feature = 2
        if FeaturesReduced[index].distances[1].isIntermolecular != 0: # 1 = Intermolecular 0 = Intramolecular
            Results.iloc[i, 6] = 'Yes'
        else:
            Results.iloc[i, 6] = 'No'
    else:
        Results.iloc[i, 4] = ' '
        Results.iloc[i, 5] = ' '
        Results.iloc[i, 6] = ' '
    if FeaturesReduced[index].nDistances == 3:
        if max_distances_in_feature < 3:
            max_distances_in_feature = 3
        Results.iloc[i, 7] = FeaturesReduced[index].distances[2].Atom1.Symbol + '-' + FeaturesReduced[index].distances[2].Atom2.Symbol
        Results.iloc[i, 8] = FeaturesReduced[index].powers[2]
        if FeaturesReduced[index].distances[2].isIntermolecular != 0: # 1 = Intermolecular 0 = Intramolecular
            Results.iloc[i, 9] = 'Yes'
        else:
            Results.iloc[i, 9] = 'No'
    else:
        Results.iloc[i, 7] = ' '
        Results.iloc[i, 8] = ' '
        Results.iloc[i, 9] = ' '
    counter = 0
    current_feature_type = FeaturesReduced[index].FeType
    for j in range(0, len(FeaturesAll), 1):
        if FeaturesAll[j].FeType == current_feature_type:
            counter += 1
    Results.iloc[i, -4] = counter
    Results.iloc[i, -3] = coef_lr[i]
    Results.iloc[i, -2] = rmse_list[i]
    Results.iloc[i, -1] = r2_list[i]
    
if max_distances_in_feature <= 2:
    del(Results['Bond 3'])
    del(Results['Power 3'])
    del(Results['Intermolecular 3'])
if max_distances_in_feature == 1:
    del(Results['Bond 2'])
    del(Results['Power 2'])
    del(Results['Intermolecular 2'])

Results.to_excel(writeResults,'OLS reduced')
writeResults.save()


print("DONE")

