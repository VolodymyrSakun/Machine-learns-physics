from structure import class2
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
            
# load reduced features and energy from file
f = open('HarmonicFeaturesReduced.dat', "rb")
FeaturesReduced = pickle.load(f)
f.close()
# load list FeaturesAll from file
f = open('HarmonicFeaturesAll.dat', "rb")
FeaturesAll = pickle.load(f)
f.close()
# load system object from file
f = open('system.dat', "rb")
system = pickle.load(f)
f.close() 
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
            rec = class2.record(0, atoms_list)
            record_list.append(rec)
            atoms_list = []
        new_rec = False
        if (len(s) == 1) and class2.isfloat(s[0]):
            record_list[-1].e = float(s[0])
        continue
# record for energy value
    elif (len(s) == 4): 
        new_rec = True
        x = float(s[1])
        y = float(s[2])
        z = float(s[3])
        atoms_list.append(class2.AtomCoordinates(system.Atoms[j], x, y, z))
        j += 1
    i += 1

def get_energy(Coef_tuple, FeaturesAll, FeaturesReduced, record_list, Record, nVariables):
    tuple_number = 0
    Found = False
    for i in range(0, len(Coef_tuple), 1):
        if Coef_tuple[i][0] == nVariables:
            tuple_number = i   
            Found = True
    if not Found:
        print('Reord with ', nVariables, ' variables does not exist')
        return
    features_set = Coef_tuple[tuple_number]
    k = Record # number of record
    E = 0
    for i in range(0, len(features_set[1]), 1): # for each variable
        current_feature_idx = features_set[1][i]
        current_feature_type = FeaturesReduced[current_feature_idx].FeType
        variable = 0
        for j in range(0, len(FeaturesAll), 1): # for each combined feature
            if FeaturesAll[j].FeType == current_feature_type:
                if FeaturesAll[j].nDistances == 1:
                    atom1_index = FeaturesAll[j].DtP1.Distance.Atom1.Index # first atom number
                    atom2_index = FeaturesAll[j].DtP1.Distance.Atom2.Index # second atom number
                    d = np.sqrt((record_list[k].atoms[atom1_index].x - record_list[k].atoms[atom2_index].x)**2 +\
                                (record_list[k].atoms[atom1_index].y - record_list[k].atoms[atom2_index].y)**2 +\
                                (record_list[k].atoms[atom1_index].z - record_list[k].atoms[atom2_index].z)**2)            
                    r = d**FeaturesAll[j].DtP1.Power # distance to correcponding power
                if FeaturesAll[j].nDistances == 2:
                    atom11_index = FeaturesAll[j].DtP1.Distance.Atom1.Index
                    atom12_index = FeaturesAll[j].DtP1.Distance.Atom2.Index
                    atom21_index = FeaturesAll[j].DtP2.Distance.Atom1.Index
                    atom22_index = FeaturesAll[j].DtP2.Distance.Atom2.Index
                    d1 = np.sqrt((record_list[k].atoms[atom11_index].x - record_list[k].atoms[atom12_index].x)**2 +\
                                 (record_list[k].atoms[atom11_index].y - record_list[k].atoms[atom12_index].y)**2 +\
                                 (record_list[k].atoms[atom11_index].z - record_list[k].atoms[atom12_index].z)**2)            
                    r1 = d1**FeaturesAll[j].DtP1.Power # distance to correcponding power
                    d2 = np.sqrt((record_list[k].atoms[atom21_index].x - record_list[k].atoms[atom22_index].x)**2 +\
                                 (record_list[k].atoms[atom21_index].y - record_list[k].atoms[atom22_index].y)**2 +\
                                 (record_list[k].atoms[atom21_index].z - record_list[k].atoms[atom22_index].z)**2)            
                    r2 = d2**FeaturesAll[j].DtP2.Power # distance to correcponding power
                    r = r1 * r2      
                variable += r
        E += variable * features_set[2][i] # combined features by coefficient
    return E
    
Size = len(record_list)
E0 = np.zeros(shape=(Size,), dtype=float)
E = np.zeros(shape=(Size,), dtype=float)
for i in range(0, Size, 1):
    if i % 100 == 0:
        print(i)
    E[i] = get_energy(Coef_tuple, FeaturesAll, FeaturesReduced, record_list, i, nVariables)
    E0[i] = record_list[i].e

mse_lr = skm.mean_squared_error(E0, E)
rmse = np.sqrt(mse_lr)
r2 = skm.r2_score(E0, E)

print('MSE = ', mse_lr)
print('RMSE = ', rmse)
print('R2 = ', r2)



