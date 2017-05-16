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
F_xlsx = 'Backward elimination.xlsx'
writeResults = pd.ExcelWriter(F_xlsx)
min_nonzero = 20 # number of active coefficient for detailed analysys
min_r2 = 0.92 # R2 min for detailed analysys

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
rsquared = 1
while (max(pvalues) != 0) & (len(ols_coef) > 2):
    ols = sm.OLS(endog = y, exog = X, hasconst = False).fit()
    y_pred = ols.predict(X)
    ols_coef = ols.params
    nonzero_count = np.count_nonzero(ols_coef)
    print(nonzero_count)
    pvalues = ols.pvalues
    rsquared = ols.rsquared
    mse_OLS = skm.mean_squared_error(y, y_pred)
    rmse_OLS = np.sqrt(mse_OLS)
    rmse_OLS_list.append(rmse_OLS)
    rsquared_list.append(rsquared*100)
    nonzero_count_list.append(nonzero_count)
    drop = np.argmax(pvalues)
    drop_order.append(features_index[drop])
    # add results in detailed table if number of active coefficients less then min_nonzero or if R2 greater then min_r2
    if (len(ols_coef) < min_nonzero) | (rsquared > min_r2):
        nonzero_count_str = str(nonzero_count)
        Table = pd.DataFrame(np.zeros(shape = (nonzero_count, 13)).astype(float), columns=['Bond 1','Power 1','Intermolecular 1','Bond 2','Power 2','Intermolecular 2','Bond 3','Power 3','Intermolecular 3', 'Number of distances in feature','Coefficient','RMSE','R2'], dtype=float)
        max_distances_in_feature = 1
        for i in range(0, nonzero_count, 1):
            Table.iloc[i, 0] = FeaturesReduced[i].distances[0].Atom1.Symbol + '-' + FeaturesReduced[i].distances[0].Atom2.Symbol
            Table.iloc[i, 1] = FeaturesReduced[i].powers[0]
            if FeaturesReduced[i].distances[0].isIntermolecular != 0: # 1 = Yes 0 = No
                Table.iloc[i, 2] = 'Yes'
            else:
                Table.iloc[i, 2] = 'No'
            if FeaturesReduced[i].nDistances >= 2:
                Table.iloc[i, 3] = FeaturesReduced[i].distances[1].Atom1.Symbol + '-' + FeaturesReduced[i].distances[1].Atom2.Symbol
                Table.iloc[i, 4] = FeaturesReduced[i].powers[1]
                if max_distances_in_feature < 2:
                    max_distances_in_feature = 2
                if FeaturesReduced[i].distances[1].isIntermolecular != 0: # 1 = Yes 0 = No
                    Table.iloc[i, 5] = 'Yes'
                else:
                    Table.iloc[i, 5] = 'No'
            else:
                Table.iloc[i, 3] = ' '
                Table.iloc[i, 4] = ' '
                Table.iloc[i, 5] = ' '
            if FeaturesReduced[i].nDistances == 3:
                if max_distances_in_feature < 3:
                    max_distances_in_feature = 3
                Table.iloc[i, 6] = FeaturesReduced[i].distances[2].Atom1.Symbol + '-' + FeaturesReduced[i].distances[2].Atom2.Symbol
                Table.iloc[i, 7] = FeaturesReduced[i].powers[2]
                if FeaturesReduced[i].distances[2].isIntermolecular != 0: # 1 = Yes 0 = No
                    Table.iloc[i, 8] = 'Yes'
                else:
                    Table.iloc[i, 8] = 'No'
            else:
                Table.iloc[i, 6] = ' '
                Table.iloc[i, 7] = ' '
                Table.iloc[i, 8] = ' '
            counter = 0
            current_feature_type = FeaturesReduced[i].FeType
            for j in range(0, len(FeaturesAll), 1):
                if FeaturesAll[j].FeType == current_feature_type:
                    counter += 1
            Table.iloc[i, 9] = counter
            Table.iloc[i, 10] = ols_coef[i] 
        Table.iloc[0, 11] = rmse_OLS
        Table.iloc[0, 12] = rsquared
        if max_distances_in_feature <= 2:
            del(Table['Bond 3'])
            del(Table['Power 3'])
            del(Table['Intermolecular 3'])
        if max_distances_in_feature == 1:
            del(Table['Bond 2'])
            del(Table['Power 2'])
            del(Table['Intermolecular 2'])

        Table.to_excel(writeResults, nonzero_count_str)
    drop = np.argmax(pvalues)
    X = np.delete(X, drop, 1) # last 1 = column, 0 = row
    del(FeaturesReduced[drop])
    features_index = np.delete(features_index, drop, 0) # last 1 = column, 0 = row
    
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
Results.to_excel(writeResults,'Summary')
writeResults.save()

print("DONE")