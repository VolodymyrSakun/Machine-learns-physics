# generates coordinates for 2 water molecules
# First molecule is prototype aligned at 0,0,0
# Requires prototype MoleculesDescriptor.

from project1 import spherical
import random
from project1 import IOfunctions
from project1 import structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import os
import shutil

# Global variables
MoleculePrototype = 'MolDescriptor-2H2O'
FileName = 'molecule.init' # file name
ParentDir = 'WaterDimer-02-15-0.2-100' # main folder
SubDirStartWith = 'Data' # subdirectories start with
RandomSeed = None # if none use seed()
DMin = 2.0 # minimal distance between center of mass of 2 water
DMax = 15.0 # max distance between center of mass of 2 water
Inc = 0.2 # length of interval
nRecords_per_interval = 1 # number of records per interval
MoleculeName = 'Water' # name of molecule in prototype

# read molecules from descriptor file
Prototypes = IOfunctions.ReadMoleculeDescription(F=MoleculePrototype) 
# align molecules that its center of mass it in 0,0,0 and principal axis aligned along x, y, z
for molecule in Prototypes:
    if molecule.Name == MoleculeName:
        prototype = copy.deepcopy(molecule)
        break
# Create intervals
Intervals = []
i = DMin
while i+Inc <= DMax:
    Intervals.append((round(i,2), round(i+Inc,2)))
    i += Inc
    
if RandomSeed is not None:
    random.seed(RandomSeed)
else:
    random.seed()
# Generate records
Records = []
for interval in Intervals:
    for i in range(0, nRecords_per_interval, 1):
        molecule = spherical.generate_random_molecules2(prototype, DMin=interval[0], DMax=interval[1], max_trials=1000)
        if molecule is None:
            break
        else:
            Molecules = [prototype, molecule]
            rec = structure.Rec(Molecules)
            Records.append(rec)

print('Number of records =', len(Records))
print('Records per interval =', nRecords_per_interval)
print('Intervals: ', Intervals)

InitialDir = os.getcwd()
ParentDir = os.path.join(InitialDir,ParentDir)
if os.path.exists(ParentDir):
    shutil.rmtree(ParentDir)
os.mkdir(ParentDir)
# Store records in files
records = []
for i in range(0, len(Records), 1):
    record = []
    record.append('$molecule\n')
    record.append('0 1\n')
    for j in Records[i].Molecules:
        for k in j.Atoms:
            S = k.Atom.Symbol
            line = "%3s%20.10f%20.10f%20.10f\n" % (S,k.x,k.y,k.z)
            record.append(line)
    record.append('$end\n')
    CurentDir = str(i)
    CurentDir = CurentDir.zfill(10)
    DirName = SubDirStartWith + CurentDir
    
    DatapointDir = os.path.join(ParentDir,DirName)
    os.mkdir(DatapointDir)
    os.chdir(DatapointDir)
    f = open(FileName, "w")
    f.writelines(record)
    f.close()   
    os.chdir(ParentDir)
    
os.chdir(InitialDir)
print("DONE")
