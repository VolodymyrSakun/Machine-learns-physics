# generates coordinates for molecules
# If number of molecules = 2, center of mass of first molecule aligned at 0,0,0
# Requires prototype MoleculesDescriptor.

from project1 import spherical
import random
from project1 import IOfunctions
from project1 import structure
import copy
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# return list of strings to stored in text file later
def RecordOut(RecordObject, idx=None):
    if idx is None: # proceed all molecules
        idx = list(range(0, len(RecordObject.Molecules), 1))
    record = []
    record.append('$molecule\n')
    record.append('0 1\n')
    for i in idx:
        molecule = RecordObject.Molecules[i]
        for k in molecule.Atoms:
            S = k.Atom.Symbol
            line = "%3s%20.10f%20.10f%20.10f\n" % (S,k.x,k.y,k.z)
            record.append(line)
    record.append('$end\n')            
    return record

# Global variables
MoleculePrototype = 'MolDescriptor-2H2O'
FileName = 'molecule.init' # file name
ParentDir = 'WaterDimer-02-15-0.2-100' # main folder
SubDirStartWith = 'Data' # subdirectories start with
RandomSeed = None # if none use seed()
DMin = 2.0 # minimal distance between center of mass of 2 water
# in case of 3 and more molecules - radius of sphere where there will be no center of mass of any molecule
DMax = 3.0 # max distance between center of mass of 2 water
# in case of 3 and more molecules - radius of sphere where centers of masses of molecules will be placed
Inc = 0.2 # length of bin. for 2 molecules only
nRecords_per_interval = 10 # number of records per bin to be generated for 2 molecules
MoleculeName = 'Water' # name of molecule in prototype
nMolecules = 3 # number of molecules per record
nRecords = 10 # number of records to be generated for 3 and more molecules
Idx = [[0,1,2], [0,1], [0,2], [1,2], [0], [1], [2]] # for 3 water molecules

# read molecules from descriptor file
Prototypes = IOfunctions.ReadMoleculeDescription(F=MoleculePrototype) 
# find required molecule in prototypes
for molecule in Prototypes:
    if molecule.Name == MoleculeName:
        prototype = copy.deepcopy(molecule)
        break

if RandomSeed is not None:
        random.seed(RandomSeed)
else:
    random.seed()  
    
# align molecules that its center of mass it in 0,0,0 and principal axis aligned along x, y, z
prototype = spherical.align_molecule(prototype)

if nMolecules == 2:
    # Create intervals
    Intervals = []
    i = DMin
    while i+Inc <= DMax:
        Intervals.append((round(i,2), round(i+Inc,2)))
        i += Inc
        
# Generate records    
    Records = []
    for interval in Intervals:
        for i in range(0, nRecords_per_interval, 1):
            molecule = spherical.generate_random_molecule2(prototype, DMin=interval[0], DMax=interval[1], max_trials=1000)
            if molecule is None:
                break
            else:
                Molecules = [prototype, molecule]
                rec = structure.Rec(Molecules)
                Records.append(rec)
    
    print('Number of records =', len(Records))
    print('Records per interval =', nRecords_per_interval)
    print('Intervals: ', Intervals)

else:
    i = 0
    Records = []
    while i < nRecords:
        Molecules = spherical.generate_molecule_coordinates_list(prototype,\
            nMolecules=nMolecules, nRecords=None, InnerRadius=DMin,\
            OuterRadius=DMax, additional_gap=0, PrototypeFirst=False, max_gen_trials=100,\
            max_inner_trials=100, max_outer_trials=10, verbose=False, n_verbose=10)
        if Molecules is None:
            break
        rec = structure.Rec(Molecules)
        Records.append(rec)  
        i += 1
    print('Number of records =', len(Records))
    
InitialDir = os.getcwd()
ParentDir = os.path.join(InitialDir,ParentDir)
if os.path.exists(ParentDir):
    shutil.rmtree(ParentDir)
os.mkdir(ParentDir)

# Store records in files
records = []
for i in range(0, len(Records), 1):    
    CurentDir = str(i)
    CurentDir = CurentDir.zfill(10)
    DirName = SubDirStartWith + CurentDir # 'Data000000000i'
    DatapointDir = os.path.join(ParentDir, DirName)
    os.mkdir(DatapointDir)
    os.chdir(DatapointDir)# go inside 'Data000000000i'
    if nMolecules == 2:
        record = RecordOut(Records[i], idx=None)
        f = open(FileName, "w")
        f.writelines(record)
        f.close()           
    elif nMolecules == 3:
        for j in Idx:
            record = RecordOut(Records[i], idx=j)
            MoleculeSubdir = ''
            for k in j:
                MoleculeSubdir += str(k) 
            MoleculeDir = os.path.join(DatapointDir, MoleculeSubdir)
            os.mkdir(MoleculeDir)
            os.chdir(MoleculeDir)# go inside 'Data000000000i\j'
            f = open(FileName, "w")
            f.writelines(record)
            f.close()        
            os.chdir(DatapointDir)
    os.chdir(ParentDir)
    
os.chdir(InitialDir)
print("DONE")

# plot 5 randomly choosen records
#for i in range(0, 5, 1):
#    rand = random.randrange(0, len(Records), 1)
#    fig1 = plt.figure(i, figsize=(19,10))
#    cx = fig1.gca(projection='3d')
#    cx.set_xlabel('X Label')
#    cx.set_ylabel('Y Label')
#    cx.set_zlabel('Z Label')
#    spherical.plot_molecules(cx, Records[rand].Molecules)



