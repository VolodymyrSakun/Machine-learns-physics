import numpy as np
import pandas as pd
from structure import library2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

if __name__ == '__main__':
    MinCorr = 1e-7
    NofFeatures = 15
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
        idx_bestfit = library2.FindBestSet(x_std, y_std, idx_new, idx_corr, Method='MSE', verbose=True)
        idx = idx_bestfit
        library2.Results_to_xls3(writeResults, str(len(idx)),\
            idx, X_train, Y_train, X_test, Y_test, FeaturesAll, FeaturesReduced)
        nonzero, mse, rmse, r2, aIC = library2.get_fit_score(X_train, X_test, Y_train, Y_test, idx=idx)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        aIC_list.append(aIC)
        nonzero_list.append(nonzero)

    writeResults.save()
    library2.plot_rmse("Foprward Results", nonzero_list, rmse_list, r2_list)
    print('DONE')
