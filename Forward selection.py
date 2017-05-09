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
from structure import class1
from sklearn.linear_model import LinearRegression

MaxFeatures = 20 # desired number of variables for final fit
F_features = 'Features and energy two distances reduced.csv' # input csv file with combined features and energy
F_structure_FeaturesReduced = 'FeaturesReduced.dat' # output data structure which contains combined features
F_structure_FeaturesAll = 'FeaturesAll.dat' # output data structure which contains all features
F_img1 = 'RMSE_OLS'
F_img2 = 'R2_OLS'
F_xlsx = 'Forward selection.xlsx'
dataset = pd.read_csv(F_features)
# del dataset['Unnamed: 0']
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

selected_features_list = []
features_index = np.linspace(start = 0, stop = NofFeatures-1, num = NofFeatures, endpoint = True, dtype=int)
rmse_array = np.zeros(shape=(NofFeatures), dtype=float)
r2_array = np.zeros(shape=(NofFeatures), dtype=float)
x = np.zeros(shape=(Size, 1), dtype=float)
x_selected = np.zeros(shape=(Size, 1), dtype=float)
# Based on best fit using one variable
print("Start fitting")
lr = LinearRegression(fit_intercept=False, normalize=True, copy_X=True, n_jobs=-1)
rmse_list = []
r2_list = []
for i in range(0, NofFeatures, 1):
    x[:, 0] = X[:, i]
    lr.fit(x, y)
    coef_lr = lr.coef_
    y_pred = lr.predict(x)
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
X = np.delete(X, m, axis=1)
features_index = np.delete(features_index, m)
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
    m = np.argmin(rmse_array)
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

Results = pd.DataFrame(np.zeros(shape = (NofSelectedFeatures, 15)).astype(float), columns=['Selected','Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2','Intermolecular 2','Bond 3','Power 3','Intermolecular 3', 'Number of distances in feature','Cummulative Coefficient','Coefficient', 'RMSE', 'R2'], dtype=float)

for i in range(0, NofSelectedFeatures, 1):
    Results.iloc[i, 0] = i+1
    index = selected_features_list[i]
    Results.iloc[i, 1] = FeaturesReduced[index].distances[0].Atom1.Symbol + '-' + FeaturesReduced[index].distances[0].Atom2.Symbol
    Results.iloc[i, 2] = FeaturesReduced[index].powers[0]
    if FeaturesReduced[index].distances[0].isIntermolecular == 0: # 0 = Yes 1 = No
        Results.iloc[i, 3] = 'Yes'
    else:
        Results.iloc[i, 3] = 'No'
    if FeaturesReduced[index].nDistances >= 2:
        Results.iloc[i, 4] = FeaturesReduced[index].distances[1].Atom1.Symbol + '-' + FeaturesReduced[index].distances[1].Atom2.Symbol
        Results.iloc[i, 5] = FeaturesReduced[index].powers[1]
        if FeaturesReduced[index].distances[1].isIntermolecular == 0: # 0 = Yes 1 = No
            Results.iloc[i, 6] = 'Yes'
        else:
            Results.iloc[i, 6] = 'No'
    else:
        Results.iloc[i, 4] = ' '
        Results.iloc[i, 5] = ' '
        Results.iloc[i, 6] = ' '
    if FeaturesReduced[index].nDistances == 3:
        Results.iloc[i, 7] = FeaturesReduced[index].distances[2].Atom1.Symbol + '-' + FeaturesReduced[index].distances[2].Atom2.Symbol
        Results.iloc[i, 8] = FeaturesReduced[index].powers[2]
        if FeaturesReduced[index].distances[2].isIntermolecular == 0: # 0 = Yes 1 = No
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
    Results.iloc[i, -5] = counter
    Results.iloc[i, -4] = coef_lr[i]
    Results.iloc[i, -3] = Results.iloc[i, -4] / Results.iloc[i, -5]
    Results.iloc[i, -2] = rmse_list[i]
    Results.iloc[i, -1] = r2_list[i]
    
writeResults = pd.ExcelWriter(F_xlsx)
Results.to_excel(writeResults,'Summary')
writeResults.save()


print("DONE")

