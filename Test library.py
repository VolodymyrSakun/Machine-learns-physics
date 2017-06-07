from structure import library1
import pandas as pd

X, Y, x_std, y_std, FeaturesAll, FeaturesReduced = library1.ReadFeatures('Features and energy two distances reduced.csv', \
    'FeaturesAll.dat', 'FeaturesReduced.dat', verbose=False)
print('Enet fit')
# create excel writer
writeResults = pd.ExcelWriter('Enet.xlsx', engine='openpyxl')
from sklearn.linear_model import enet_path
# eps : Length of the path. ``eps=1e-3`` means that ``alpha_min / alpha_max = 1e-3``
Eps = 0.001 # regularization strength works fine from 1e-5 to 1e-3. Lower is better but slower
# get elastic net path
_ , coefs, _ = enet_path(x_std, y_std, l1_ratio=0.05, eps=Eps, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=True, return_n_iter=False, positive=False, check_input=True)
# select the best subset from elastic net path
idx = library1.SelectBestSubsetFromElasticNetPath(coefs, x_std, y_std)
features_index = library1.BackwardElimination(X, Y, FeaturesAll, FeaturesReduced, \
    Method='sequential', Criterion='MSE', N_of_features_to_select=2, idx=idx, \
    PlotPath=True, StorePath=True, FileName='Backward sequential', N_of_last_iterations_to_store=20, verbose=True)


#idx_corr = library1.ClassifyCorrelatedFeatures(X, features_index, MinCorrelation=0.95, Model=1, verbose=True)
#active_features = library1.FindAlternativeFit(x_std, y_std, features_index, idx_corr, Fit=2, Method='MSE', verbose=True)
#writeResults = pd.ExcelWriter('FileName.xlsx', engine='openpyxl')
#library1.Results_to_xls(writeResults, '_1', active_features, X, Y, FeaturesAll, FeaturesReduced)

#writeResults.save()

    
