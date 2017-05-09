# Performs backward elimination from full set of features 
# Removes least important variables one by one based on p-value
# until all of p-values = 0 
# Can be very slow if the feeture set is big

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pickle 
import statsmodels.regression.linear_model as sm
import scipy
from structure import class1

F_features = 'Features and energy two distances reduced.csv' # input csv file with combined features and energy
F_structure_FeaturesReduced = 'FeaturesReduced.dat' # output data structure which contains combined features
F_structure_FeaturesAll = 'FeaturesAll.dat' # output data structure which contains all features
F_img1 = 'RMSE_OLS'
F_img2 = 'R2_OLS'
F_xlsx = 'Forward selection.xlsx'
dataset = pd.read_csv(F_features)
NofDistances = dataset.shape[1] - 1 # read numder of distances from dataset
Dist = dataset.iloc[:, 0:NofDistances].values
Energy = dataset.iloc[:, -1].values
Size = len(Energy)
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
number_of_features = Dist.shape[1]

X = Dist
y = Energy

features_index = np.linspace(start = 0, stop = number_of_features-1, num = number_of_features, endpoint = True, dtype=int)
ols_coef = np.zeros(shape=(3), dtype=float)
pvalues = np.ones(shape=(3), dtype=float)
nonzero_count_list = []
rsquared_list = []
rmse_OLS_list = []
drop_order = []
while (max(pvalues) != 0) & (len(ols_coef) > 2):
    ols = sm.OLS(endog = y, exog = X, hasconst = False).fit()
    y_pred = ols.predict(X)
    ols_coef = ols.params
    pvalues = ols.pvalues
    rsquared = ols.rsquared
    mse_OLS = skm.mean_squared_error(y, y_pred)
    rmse_OLS = np.sqrt(mse_OLS)
    drop = np.argmax(pvalues)
    X = np.delete(X, drop, 1) # last 1 = column, 0 = row
    del(FeaturesReduced[drop])
    nonzero_count = np.count_nonzero(ols_coef)
    rmse_OLS_list.append(rmse_OLS)
    rsquared_list.append(rsquared*100)
    nonzero_count_list.append(nonzero_count)
    drop_order.append(features_index[drop])
    features_index = scipy.delete(features_index, drop, 0) # last 1 = column, 0 = row
    print(nonzero_count)

plt.figure(1, figsize = (19, 10))
plt.plot(nonzero_count_list, rmse_OLS_list, ':')
plt.xlabel('Active coefficients')
plt.ylabel('Root of mean square error')
plt.title('Backward elimination. Root of mean square error vs Active coefficiants')
plt.axis('tight')
plt.savefig(F_img1, bbox_inches='tight')

plt.figure(2, figsize = (19, 10))
plt.plot(nonzero_count_list, rsquared_list, ':')
plt.xlabel('Active coefficients')
plt.ylabel('R2')
plt.title('Backward elimination. R2 vs Active coefficiants')
plt.axis('tight')
plt.savefig(F_img2, bbox_inches='tight')

Results = pd.DataFrame(np.zeros(shape = (len(nonzero_count_list), 1)).astype(float), columns=['Empty'], dtype=float)
Results.insert(1, 'N Non-zero coef', nonzero_count_list)
Results.insert(2, 'RMSE', rmse_OLS_list)
Results.insert(3, 'R2', rsquared_list)
Results.insert(4, 'Drop order', drop_order)
del Results['Empty']
writeResults = pd.ExcelWriter(F_xlsx)
Results.to_excel(writeResults,'Summary')
writeResults.save()

print("DONE")