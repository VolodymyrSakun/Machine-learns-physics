from project1 import structure
from project1 import library
import numpy as np
import pickle 
import re
import sklearn.metrics as skm

F = 'SystemDescriptor.' # file with info about system structure
nVariables = 9 # How many variables participate in fit

# read descriptor from file
with open(F) as f:
    lines = f.readlines()
f.close()
lines = [x.strip() for x in lines] # x is string

for i in range(0, len(lines), 1):
    x = lines[i]
    if len(x) == 0:
        continue
    if x[0] == '#':
        continue
    if (x.find('F_data') != -1):        
        s = re.split('F_data = |F_data=|', x)
        s = list(filter(bool, s))
        f_name = s
        F_data = f_name[0]
            
_, _, FeaturesAll, FeaturesReduced, system, _ = library.ReadFeatures(F_features='Harmonic Features.csv', \
    F_FeaturesAll='HarmonicFeaturesAll.dat', F_FeaturesReduced='HarmonicFeaturesReduced.dat',\
    F_System='system.dat', F_Records=None, verbose=False)

# load coefficints
f = open('Coefficients.dat', "rb")
Coef_tuple = pickle.load(f)
f.close()

# Read coordinates from file
f = open(F_data, "r")
data0 = f.readlines()
f.close()
data1 = []
for i in range(0, len(data0), 1):
    data1.append(data0[i].rstrip())
del(data0)
    
i = 0 # counts lines in textdata
j = 0 # counts atom records for each energy value
atoms_list = [] # temporary list
record_list = []
new_rec = False
while i < len(data1):
    s = data1[i].split() # line of text separated in list
    if len(s) < 4: # not record
        i += 1
        j = 0
        if new_rec:
            rec = structure.Record(0, atoms_list)
            record_list.append(rec)
            atoms_list = []
        new_rec = False
        if (len(s) == 1) and library.isfloat(s[0]):
            record_list[-1].e = float(s[0])
        continue
# record for energy value
    elif (len(s) == 4): 
        new_rec = True
        x = float(s[1])
        y = float(s[2])
        z = float(s[3])
        atoms_list.append(structure.AtomCoordinates(system.Atoms[j], x, y, z))
        j += 1
    i += 1
    
Size = len(record_list)
E0 = np.zeros(shape=(Size,), dtype=float)
E = np.zeros(shape=(Size,), dtype=float)
for i in range(0, Size, 1):
    if i % 100 == 0:
        print(i)
    E[i] = library.get_energy(Coef_tuple, FeaturesAll, FeaturesReduced, record_list, i, nVariables)
    E0[i] = record_list[i].e

mse_lr = skm.mean_squared_error(E0, E)
rmse = np.sqrt(mse_lr)
r2 = skm.r2_score(E0, E)

print('MSE = ', mse_lr)
print('RMSE = ', rmse)
print('R2 = ', r2)



