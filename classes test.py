import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LinearRegression


from collections import namedtuple
F = 'SystemDescriptor' # file with info about system structure

class Atom(namedtuple('Atom', ['Symbol', 'Index', 'Type', 'MolecularIndex'], verbose=False)):
    __slots__ = ()

class AtomCoordinates(namedtuple('AtomCoordinates', ['Atom', 'x', 'y', 'z'], verbose=False)):
    __slots__ = ()

class Distance(namedtuple('Distance', ['Atom1', 'Atom2', 'isIntermolecular', 'DiType'], verbose=False)):
    __slots__ = ()
    def __new__(_cls, Atom1, Atom2, isIntermolecular, DiType):
        'Create new instance of Distance(Atom1, Atom2, isIntermolecular, DiType)'
        i = isIntermolecular
        if (Atom1.MolecularIndex == Atom2.MolecularIndex):
            i = 0
        else:
            i = 1
# compute a UNIQUE identification number for a distance type        
        d = nAtTypes*max(Atom1.Type, Atom2.Type) + min(Atom1.Type, Atom2.Type)+\
            nAtTypes**2 * i
        return tuple.__new__(_cls, (Atom1, Atom2, i, d))

class Angle(namedtuple('Angle', ['Atom1', 'Atom2', 'Atom3', 'isIntermolecular12', 'isIntermolecular13', 'isIntermolecular23', 'AnType'], verbose=False)):
    __slots__ = ()
    
class System(namedtuple('System', ['atoms', 'nAtoms', 'nAtType', 'distances', 'nDistances', 'nDiTypes'], verbose=False)):
    __slots__ = ()

class InvPowDistancesFeature(namedtuple('InvPowDistancesFeature', ['nDistances', 'distances', 'powers', 'FeType'], verbose=False)):
    __slots__ = ()    
    def __new__(_cls, nDistances, distances, powers, FeType):
        'Create new instance of InvPowDistancesFeature'
        if nDistances == 1:
            dist = (distances, 0)
            power = (powers, 0)
        else:
            print("Wrong constructor usage")
            return
        return tuple.__new__(_cls, (nDistances, dist, power, FeType))


def AreTwoFeaturesEquivalent(Feature1, Feature2):
# check number of distances in features (equal or not)
    if (Feature1.nDistances != Feature2.nDistances):
        return False
    else:
        ipow = 0
        while ipow < Feature1.nDistances:
            if (Feature1.distances[ipow].DiType != Feature2.distances[ipow].DiType):
                return False
            ipow += 1
        for j in range(0, Feature1.nDistances, 1):
            if (Feature1.powers[j] != Feature2.powers[j]):
                return False   
    return True

class record:
    def __init__(self, e, atoms):
        self.e = e
        self.atoms = atoms
       
def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False

# main()
with open(F) as f:
    lines = f.readlines()
lines = [x.strip() for x in lines] # x is string
# find '&Features' part
if ('&FEATURES' in lines):
    for i in range(0, len(lines), 1):
        x = lines[i]
        if (x.find('MaxPower') != -1):
            MaxPowerStr = x.split(',', -1)
MaxPower = int(MaxPowerStr[1])            
    
Atoms = [] # create list of atom structures from file
j = 0 # order in the system 
if ('&SYSTEM' in lines):
    i = 0
    while ((lines[i].find('&SYSTEM') == -1) & (i < len(lines))):
        i += 1
    i += 1
    while ((lines[i].find('&END') == -1) & (i < len(lines))):
        if (lines[i][0] == '#'):
            i += 1
            continue
        else:
            Str = lines[i].split(',', -1)
            Atoms.append(Atom(Str[0], j, int(Str[1]), int(Str[2])))
            j += 1
            i += 1
# determine n of atom types from list
nAtTypes = 0        
for i in range(0, len(Atoms), 1):
    if (Atoms[i].Type > nAtTypes):
        nAtTypes = Atoms[i].Type
nAtTypes += 1        

nAtoms = len(Atoms)
Distances = [] # create list of distances from list of atoms
for i in range(0, nAtoms, 1):
    for j in range(i+1, nAtoms, 1):
        Distances.append(Distance(Atoms[i], Atoms[j], None, None))

nDiTypes = 1     
DiTypesList = [] # store unique distance types in the list
DiTypesList.append(Distances[0].DiType)
for i in range(1, len(Distances), 1):
    if (Distances[i].DiType in DiTypesList):
        continue
    else:
        DiTypesList.append(Distances[i].DiType)
        nDiTypes += 1
    
sys = System(Atoms, nAtoms, nAtTypes, Distances, len(Distances), nDiTypes)

FeaturesAll = [] # list of all features
for i in range(1, MaxPower+1, 1):
    for j in range(0, sys.nDistances, 1):
        FeaturesAll.append(InvPowDistancesFeature(1, Distances[j], -i, -1))

NofFeatures = len(FeaturesAll) # Total number of features
currentFeatureType = 0   
FeaturesAll[0] = FeaturesAll[0]._replace(FeType=0)
FeaturesReduced = []

for i in range(0, NofFeatures, 1):
    if FeaturesAll[i].FeType == -1:
        currentFeatureType += 1
        FeaturesAll[i] = FeaturesAll[i]._replace(FeType=currentFeatureType)
        FeaturesReduced.append(FeaturesAll[i])
        for j in range(i, NofFeatures, 1):
            if AreTwoFeaturesEquivalent(FeaturesAll[i], FeaturesAll[j]):
                FeaturesAll[j] = FeaturesAll[j]._replace(FeType=FeaturesAll[i].FeType)

NofFeaturesReduced = len(FeaturesReduced)
# Read coordinates from file
#f = open("datafile2 10 02 2017.x", "r")
f = open("datafile1 from github gaussian process.x", "r")
data0 = f.readlines()
data1 = []
for i in range(0, len(data0), 1):
    data1.append(data0[i].rstrip())
    
# Rearrange data in structure
i = 0 # counts lines in textdata
j = 0 # counts atom records for each energy value
atoms_list = []
record_list = []
while i < len(data1):
    s = data1[i].split() # line of text separated in list
    if len(s) == 0: # empty line
        j = 0
        atoms_list = []
# record for energy value
    elif (len(s) == 1) and isfloat(s[0]): 
        e = float(s[0])
        rec = record(e, atoms_list)
        record_list.append(rec)
    elif (len(s) == 4): 
        atom_symbol = s[0]
        x = float(s[1])
        y = float(s[2])
        z = float(s[3])
        atoms_list.append(AtomCoordinates(Atoms[j], x, y, z))
        j += 1
    i += 1

Size = len(record_list) # N of observations

# How to access to structured data:    
    
# record_list[Number of observation].atoms[Number of atom].Atom.Symbol
# record_list[Number of observation].atoms[Number of atom].Atom.Index
# record_list[Number of observation].atoms[Number of atom].Atom.Type
# record_list[Number of observation].atoms[Number of atom].Atom.MolecularIndex
# record_list[Number of observation].atoms[Number of atom].x
# record_list[Number of observation].atoms[Number of atom].y
# record_list[Number of observation].atoms[Number of atom].z
# record_list[Number of observation].e
             
# FeaturesAll[Number of feature].distances.Atom1.Symbol
# FeaturesAll[Number of feature].distances.Atom1.Index
# FeaturesAll[Number of feature].distances.Atom1.Type
# FeaturesAll[Number of feature].distances.Atom1.MolecularIndex
# FeaturesAll[Number of feature].distances.Atom2.Symbol
# FeaturesAll[Number of feature].distances.Atom2.Index
# FeaturesAll[Number of feature].distances.Atom2.Type
# FeaturesAll[Number of feature].distances.Atom2.MolecularIndex
# FeaturesAll[100].powers
# FeaturesAll[100].FeType

# creating the array of all features
i = 0 # N of observation
j = 0 # N of feature
features_array = np.zeros(shape=(Size, NofFeatures), dtype=float)
energy = np.zeros(shape=(Size, 1), dtype=float)
for j in range(0, len(FeaturesAll), 1):
    for i in range(0, len(record_list), 1):
        atom1_index = FeaturesAll[j].distances[0].Atom1.Index # first atom number
        atom2_index = FeaturesAll[j].distances[0].Atom2.Index # second atom number
# calculating disnances in 3D space                                 
        d = np.sqrt((record_list[i].atoms[atom1_index].x - record_list[i].atoms[atom2_index].x)**2 +\
            (record_list[i].atoms[atom1_index].y - record_list[i].atoms[atom2_index].y)**2 +\
            (record_list[i].atoms[atom1_index].z - record_list[i].atoms[atom2_index].z)**2)            
        r = d**FeaturesAll[j].powers[0] # distance to correcponding power
        features_array[i, j] = r # store to array
        energy[i, 0] = record_list[i].e # energy

# create combined features
features_array_reduced = np.zeros(shape=(Size, NofFeaturesReduced), dtype=float)
# if features have the same FeType - sum them and store as one feature
for i in range(0, Size, 1): # observations
    for k in range(0, NofFeaturesReduced, 1):
        for j in range(0, NofFeatures, 1):
            if FeaturesAll[j].FeType == k:
                features_array_reduced[i, k] += features_array[i, j]
                              

x = features_array_reduced
y = energy
##############################################################################
# LassoLarsCV: least angle regression

# Compute paths
print("Computing regularization path using the Lars lasso...")
lassolars = LassoLarsCV(cv=30, max_iter=1000000, fit_intercept=False, normalize=True)
lassolars.fit(x, y)
alpha_lassolars = lassolars.alpha_
coef_lassolars = lassolars.coef_
nonzero_lassolars = np.count_nonzero(coef_lassolars)
y_pred = lassolars.predict(x)
r2_lassolars = skm.r2_score(y, y_pred)
mse_lassolars = skm.mean_squared_error(y, y_pred)
rmse_lassolars = np.sqrt(mse_lassolars)

resultsLassoLarsCV = []
for i in range(0, NofFeaturesReduced, 1):
    if coef_lassolars[i] != 0:
        if FeaturesReduced[i].distances[0].isIntermolecular == 0:
            description = ' Intermolecular '
        else:
            description = ' Not Intermolecular '
        resultsLassoLarsCV.append('Coefficient: ' + "{:.9f}".format(coef_lassolars[i]) + '\t' +\
            FeaturesReduced[i].distances[0].Atom1.Symbol + '-' +\
            FeaturesReduced[i].distances[0].Atom2.Symbol + '\t' +\
            str(FeaturesReduced[i].powers[0]) + description)

##############################################################################
# LassoLarsIC: least angle regression

# Compute paths
print("Computing regularization path using the Lars lasso...")
lassolarsIC = LassoLarsIC(criterion='aic', max_iter=1000000, fit_intercept=False, normalize=True)
lassolarsIC.fit(x, y)
alpha_lassolarsIC = lassolarsIC.alpha_
coef_lassolarsIC = lassolarsIC.coef_
nonzero_lassolarsIC = np.count_nonzero(coef_lassolarsIC)
y_pred = lassolarsIC.predict(x)
r2_lassolarsIC = skm.r2_score(y, y_pred)
mse_lassolarsIC = skm.mean_squared_error(y, y_pred)
rmse_lassolarsIC = np.sqrt(mse_lassolarsIC)

resultsLassoLarsIC = []
for i in range(0, NofFeaturesReduced, 1):
    if coef_lassolarsIC[i] != 0:
        if FeaturesReduced[i].distances[0].isIntermolecular == 0:
            description = ' Intermolecular '
        else:
            description = ' Not Intermolecular '
        resultsLassoLarsIC.append('Coefficient: ' + "{:.9f}".format(coef_lassolarsIC[i]) + '\t' +\
            FeaturesReduced[i].distances[0].Atom1.Symbol + '-' +\
            FeaturesReduced[i].distances[0].Atom2.Symbol + '\t' +\
            str(FeaturesReduced[i].powers[0]) + description)
                
##############################################################################
# LassoCV: 

# Compute paths
print("Computing regularization path using the Lars lasso...")
lasso = LassoCV(cv=30, max_iter=1000000, fit_intercept=False, normalize=True)
lasso.fit(x, y)
alpha_lasso = lasso.alpha_
coef_lasso = lasso.coef_
nonzero_lasso = np.count_nonzero(coef_lasso)
y_pred = lasso.predict(x)
r2_lasso = skm.r2_score(y, y_pred)
mse_lasso = skm.mean_squared_error(y, y_pred)
rmse_lasso = np.sqrt(mse_lasso)

resultsLassoCV = []
for i in range(0, NofFeaturesReduced, 1):
    if coef_lasso[i] != 0:
        if FeaturesReduced[i].distances[0].isIntermolecular == 0:
            description = ' Intermolecular '
        else:
            description = ' Not Intermolecular '
        resultsLassoCV.append('Coefficient: ' + "{:.9f}".format(coef_lasso[i]) + '\t' +\
            FeaturesReduced[i].distances[0].Atom1.Symbol + '-' +\
            FeaturesReduced[i].distances[0].Atom2.Symbol + '\t' +\
            str(FeaturesReduced[i].powers[0]) + description)
        
##############################################################################
# ElasticNetCV: 

# Compute paths
print("Computing regularization path using the Lars lasso...")
elastic_net = ElasticNetCV(l1_ratio=0.5, cv=30, max_iter=1000000, fit_intercept=False, normalize=True)
elastic_net.fit(x, y)
alpha_elastic_net = elastic_net.alpha_
coef_elastic_net = elastic_net.coef_
nonzero_elastic_net = np.count_nonzero(coef_elastic_net)
y_pred = elastic_net.predict(x)
r2_elastic_net = skm.r2_score(y, y_pred)
mse_elastic_net = skm.mean_squared_error(y, y_pred)
rmse_elastic_net = np.sqrt(mse_elastic_net)

resultsElasticNetCV = []
for i in range(0, NofFeaturesReduced, 1):
    if coef_elastic_net[i] != 0:
        if FeaturesReduced[i].distances[0].isIntermolecular == 0:
            description = ' Intermolecular '
        else:
            description = ' Not Intermolecular '
        resultsElasticNetCV.append('Coefficient: ' + "{:.9f}".format(coef_elastic_net[i]) + '\t' +\
            FeaturesReduced[i].distances[0].Atom1.Symbol + '-' +\
            FeaturesReduced[i].distances[0].Atom2.Symbol + '\t' +\
            str(FeaturesReduced[i].powers[0]) + description) 
       
lr = LinearRegression(fit_intercept=False, normalize=True)   
lr.fit(x, y)
coef_lr = lr.coef_     
nonzero_lr = np.count_nonzero(coef_lr)
y_pred = lr.predict(x)
r2_lr = skm.r2_score(y, y_pred)
mse_lr = skm.mean_squared_error(y, y_pred)
rmse_lr = np.sqrt(mse_lr)

print("DONE")