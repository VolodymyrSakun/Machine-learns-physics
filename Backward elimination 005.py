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
from structure import class1

F_features = 'Features and energy two distances reduced.csv' # input csv file with combined features and energy
F_structure_FeaturesReduced = 'FeaturesReduced.dat' # output data structure which contains combined features
F_structure_FeaturesAll = 'FeaturesAll.dat' # output data structure which contains all features
F_img1 = 'RMSE_OLS'
F_img2 = 'R2_OLS'
F_xlsx = 'Backward selection 005.xlsx'
drop_criterion = 0.05
detailed_analysys = 20 # number of active coefficient for detailed analysys

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
aIC_list = []
while (max(pvalues) != 0) & (len(ols_coef) > 2):
    drop = []
    ols = sm.OLS(endog = y, exog = X, hasconst = False).fit()
    y_pred = ols.predict(X)
    ols_coef = ols.params
    pvalues = ols.pvalues
    rsquared = ols.rsquared
    mse_OLS = skm.mean_squared_error(y, y_pred)
    rmse_OLS = np.sqrt(mse_OLS)
    nonzero_count = np.count_nonzero(ols_coef)
    rmse_OLS_list.append(rmse_OLS)
    rsquared_list.append(rsquared*100)
    nonzero_count_list.append(nonzero_count)
    rSS = 0
    for i in range(0, Size, 1):
        rSS += (y_pred[i] - y[i])**2
    aIC = 2 * nonzero_count + Size * np.log(rSS)
    aIC_list.append(aIC)
    print(nonzero_count)
# remove coefficients with p-value > drop_criterion (usually 0.05)   
    for i in range(0, len(features_index), 1):
        if pvalues[i] > drop_criterion:
            drop.append(i)
            drop_order.append(i)
        else:
            drop.append(np.argmax(pvalues))
    X = np.delete(X, drop, 1) # last 1 = column, 0 = row
    features_index = np.delete(features_index, drop, 0) # last 1 = column, 0 = row
    length = len(drop)
    for i in range(0, length, 1):
        FeaturesReduced[drop[i]] = FeaturesReduced[drop[i]]._replace(nDistances=0)
    i = 0
    while i < len(FeaturesReduced):
        if FeaturesReduced[i].nDistances == 0:
            del(FeaturesReduced[i])
            i -= 1
        i += 1

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
Results.insert(4, 'AIC', aIC_list)
del Results['Empty']
writeResults = pd.ExcelWriter(F_xlsx)
Results.to_excel(writeResults,'Summary')
writeResults.save()

print("DONE")