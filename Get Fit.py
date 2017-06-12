from structure import library1
import pandas as pd
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
import numpy as np
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

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

# Global variables
ProcedureFileName = 'procedure.txt'
L1 = 0.7
eps = 1e-3
n_alphas = 100
FinalAdjustment = 'Backward sequential'
#FinalAdjustment = 'Backward sequential and Search Alternative'
LastIterationsToStore = 20
MinCorr = 0.9

# Read features and structures from files stored by "Generate combined features"
X, Y, FeaturesAll, FeaturesReduced = library1.ReadFeatures('Features and energy two distances reduced.csv', \
    'FeaturesAll.dat', 'FeaturesReduced.dat', verbose=False)
# split data in order to separate training set and test set
# all response variables Y must be 1D arrays
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
Y_train = Y_train.reshape(-1, 1)
x_std = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
y_std = scale(Y_train, axis=0, with_mean=True, with_std=False, copy=True)
Y_train = Y_train.reshape(-1)
y_std = y_std.reshape(-1)
# redirect output to file
f_new, f_old = RedirectPrintToFile(ProcedureFileName)
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
idx, alpha, mse, nonz = library1.SelectBestSubsetFromElasticNetPath(x_std, y_std,\
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
    features_index = library1.BackwardElimination(X_train, Y_train, X_test, Y_test,\
        FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
        N_of_features_to_select=2, idx=idx, PlotPath=True, StorePath=True, \
        FileName='Backward sequential', N_of_last_iterations_to_store=LastIterationsToStore,\
        verbose=False)
    t_sec = time.time() - t
    print('Backward Elimination worked ', t_sec, 'sec')
    
if FinalAdjustment == 'Backward sequential and Search Alternative':
    print("\n", 'Features left for Backward Elimination and Search Alternative = ', len(idx))
    print('MinCorrelation = ', MinCorr)
# Find correlation matrix
    A = pd.DataFrame(X, dtype=float)
    B = A.corr()
    B.fillna(0, inplace=True) # replace NaN with 0
    C = B.as_matrix()
    writeResults = pd.ExcelWriter('FileName.xlsx', engine='openpyxl')
    t = time.time()
    while len(idx) > 2:
        FeatureSize = len(idx)
        idx_backward = library1.BackwardElimination(X_train, Y_train, X_test, Y_test,\
            FeaturesAll, FeaturesReduced, Method='sequential', Criterion='MSE', \
            N_of_features_to_select=FeatureSize-1, idx=idx, PlotPath=False, StorePath=False, \
            FileName='Backward sequential', N_of_last_iterations_to_store=None, verbose=False)
        if len(idx_backward) > LastIterationsToStore:
            idx = idx_backward
            continue
        idx_corr = library1.ClassifyCorrelatedFeatures(X, idx_backward, MinCorrelation=MinCorr,\
            Model=1, Corr_Matrix=C, verbose=False)
        idx_alternative, res = library1.FindAlternativeFit(X, Y, idx_backward,\
            idx_corr, Method='MSE', verbose=False)
        library1.Results_to_xls(writeResults, str(len(idx_alternative)),\
            idx_alternative, X, Y, FeaturesAll, FeaturesReduced)
        idx = idx_alternative
    t_sec = time.time() - t
    print('Backward Elimination and Search Alternative worked ', t_sec, 'sec')
    writeResults.save()

# return output back to console
RedirectPrintToConsole(f_new, f_old)