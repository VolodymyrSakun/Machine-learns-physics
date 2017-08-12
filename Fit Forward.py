from project1 import library
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import shutil
from sklearn.linear_model import enet_path

nVariables = 3
RandomSeed = 101
UseCorrelationMatrix = False
MinCorr = 0.9
verbose = 1
BestFitMethod = 'Fast'
MaxLoops = 1000
MaxBottom = 200
F_Xlsx = 'Forward Fit.xlsx'
F_Plot = 'Forward Results'

# Read features and structures from files stored by "Generate combined features"
X, Y, FeaturesAll, FeaturesReduced, system, _ = library.ReadFeatures('Harmonic Features.csv', \
    'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', 'system.dat', verbose=False)
# split data in order to separate training set and test set
# all response variables Y must be 1D arrays
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RandomSeed)
Y_train = Y_train.reshape(-1, 1)
# x_scale = MinMaxScaler(feature_range=(-1, 1), copy=True)
x_scale = StandardScaler(copy=True, with_mean=True, with_std=True)
y_scale = StandardScaler(copy=True, with_mean=True, with_std=False)
x_scale.fit(X_train)
y_scale.fit(Y_train)
x_std = x_scale.transform(X_train)
x_std = library.Scaler_L2(X_train)
y_std = y_scale.transform(Y_train)
Y_train = Y_train.reshape(-1)
y_std = y_std.reshape(-1)

idx = []
writeResults = pd.ExcelWriter(F_Xlsx, engine='openpyxl')
if UseCorrelationMatrix:
    C = np.cov(x_std, rowvar=False, bias=True)
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
while len(idx) < nVariables: # starts from empty list
    idx = library.AddBestFeature(x_std, y_std, idx=idx)
    print('Active coefficients = ', len(idx))
# lower mse first = least important features first
#    idx = library.rank_features(x_std, y_std, idx, direction='Lo-Hi')
# Greater mse first = most important features first
    idx = library.rank_features(x_std, y_std, idx, direction='Hi-Lo')    
    if UseCorrelationMatrix:
        idx_corr = library.ClassifyCorrelatedFeatures(x_std, idx,\
            MinCorrelation=MinCorr, Model=1, Corr_Matrix=C, verbose=verbose)
    else:    
        idx_corr = library.get_full_list(idx, x_std.shape[1])
    if BestFitMethod == 'Fast':
        idx = library.FindBestSet(x_std, y_std, idx, idx_corr,\
            VIP_idx=None, Method='MSE', verbose=verbose)
    else:
        idx = library.FindBestSetTree(x_std, y_std, idx, idx_corr,\
            VIP_idx=None, MaxLoops=MaxLoops, MaxBottom=MaxBottom, verbose=verbose)
# lower mse first = least important features first
#    idx = library3.rank_features(x_std, y_std, idx, direction='Lo-Hi')
# Greater mse first = most important features first
    idx = library.rank_features(x_std, y_std, idx, direction='Hi-Lo')
    _, _, mse_train, rmse_train, r2_train = library.get_fit_score(X_train, Y_train, X_test, Y_test, idx=idx, Test=False)
    coef, nonzero, mse_test, rmse_test, r2_test = library.get_fit_score(X_train, Y_train, X_test, Y_test, idx=idx, Test=True)
    data_to_xls = []
    data_to_xls.append((coef, nonzero, mse_train, rmse_train, r2_train)) # score based on train set
    data_to_xls.append((coef, nonzero, mse_test, rmse_test, r2_test)) # score based on test set
    library.Results_to_xls(writeResults, str(len(idx)), idx, FeaturesAll, FeaturesReduced,\
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

    print('MSE Test = ', mse_test)
writeResults.save()
library.plot_rmse(F_Plot + '_Test', nonzero_list, rmse_test_list, r2_test_list) 
library.plot_rmse(F_Plot + '_Train', nonzero_list, rmse_train_list, r2_train_list) 

directory = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
if not os.path.exists(directory):
    os.makedirs(directory)    
if os.path.isfile('SystemDescriptor.'):
    shutil.copyfile('SystemDescriptor.', directory + '\\' + 'SystemDescriptor.')
if os.path.isfile('Structure.xlsx'):    
    shutil.copyfile('Structure.xlsx', directory + '\\' + 'Structure.xlsx')
if os.path.isfile('Harmonic Features Reduced List.xlsx'): 
    shutil.copyfile('Harmonic Features Reduced List.xlsx', directory + '\\' + 'Harmonic Features Reduced List.xlsx')
if os.path.isfile('Coefficients.dat'):
    shutil.copyfile('Coefficients.dat', directory + '\\' + 'Coefficients.dat')
if os.path.isfile(F_Xlsx):
    shutil.move(F_Xlsx, directory + '\\' + F_Xlsx)
files = os.listdir("./")
for file in files:
    if file.startswith(F_Plot):
        if os.path.isfile(file):
            shutil.move(file, directory + '\\' + file)
            continue

"""
Alphas = np.logspace(-7, -10, 4)
x = np.concatenate((x_std, -x_std), axis=1)
alphas, coefs, _, n_iter = enet_path(x_std, y_std, l1_ratio=1, eps=0.001,\
    n_alphas=100, alphas=Alphas, precompute='auto', Xy=None, copy_X=True,\
    coef_init=None, verbose=False, return_n_iter=True, positive=False,\
    check_input=True, tol=1e-3, max_iter=100000)
"""
print('DONE')


