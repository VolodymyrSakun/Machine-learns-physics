import numpy as np
import pandas as pd
from structure import library2
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

if __name__ == '__main__':
    MinCorr = 1e-7
    NofFeatures = 20
    # Read features and structures from files stored by "Generate combined features"
    X, Y, FeaturesAll, FeaturesReduced = library2.ReadFeatures('Harmonic Features.csv', \
        'HarmonicFeaturesAll.dat', 'HarmonicFeaturesReduced.dat', verbose=False)
    # split data in order to separate training set and test set
    # all response variables Y must be 1D arrays
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
    Y_train = Y_train.reshape(-1, 1)
    x_std = scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
    y_std = scale(Y_train, axis=0, with_mean=True, with_std=False, copy=True)
    Y_train = Y_train.reshape(-1)
    y_std = y_std.reshape(-1)
    Size = x_std.shape[0]
    C = np.cov(x_std, rowvar=False, bias=True)
    idx = library2.ForwardSequential(x_std, y_std, nVariables=1, idx=None)
    writeResults = pd.ExcelWriter('Forward Sequential Best Fit.xlsx', engine='openpyxl')
    mse_list = []
    rmse_list = []
    r2_list = []
    aIC_list = []
    nonzero_list = []
    while len(idx) < NofFeatures:
        nVariables = len(idx)+1
        idx_new = library2.ForwardSequential(x_std, y_std, nVariables=nVariables, idx=idx)
        idx_corr = library2.ClassifyCorrelatedFeatures(x_std, idx_new, MinCorrelation=MinCorr, Model=1, Corr_Matrix=C, verbose=False)    
        idx_bestfit = library2.FindBestSetMP2(x_std, y_std, idx_new, idx_corr, Method='MSE', verbose=True)
        idx = idx_bestfit
        library2.Results_to_xls2(writeResults, str(len(idx)),\
            idx, X_train, Y_train, X_test, Y_test, FeaturesAll, FeaturesReduced)
        nonzero, mse, rmse, r2, aIC = library2.get_fit_score(X_train, X_test, Y_train, Y_test, idx=idx)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        aIC_list.append(aIC)
        nonzero_list.append(nonzero)

    writeResults.save()
    library2.plot_rmse("Foprward Results", nonzero_list, rmse_list, r2_list)
        
"""    
    nVariables=10 # Desired number of variables
    idx=None # Initial fit if exists
    lr = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=-1)
    if idx is None: # determine first feature
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
"""

    
        
