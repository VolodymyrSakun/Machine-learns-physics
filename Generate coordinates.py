# generates coordinates for 2 water molecules
# First molecule is prototype aligned at 0,0,0

from project1 import spherical
import random
from project1 import IOfunctions
from project1 import structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# read molecules from descriptor file
# Molecules = spherical.ReadMolecules(F='MoleculesDescriptor.')  
Molecules = IOfunctions.ReadMoleculeDescription(F='MoleculesDescriptor.') 
# align molecules that its center of mass it in 0,0,0 and principal axis aligned along x, y, z
Water = spherical.align_molecule(Molecules[0])
prototype = Water # which molecule will be used
# generate coordinates
RandomSeed = None # if none use seed()
DMin = 2.0
DMax = 15.0
Inc = 0.10
nRecords_per_interval = 30 # number of records
Intervals = []
i = DMin
while i+Inc < DMax:
    Intervals.append((round(i,2), round(i+Inc,2)))
    i += Inc
    
# if we need to generate molecules far from each others, we can specify minimal
# distance between atoms by additional_gap variable
additional_gap = 0 # to be added to r1 + r2
if RandomSeed is not None:
    random.seed(RandomSeed)
else:
    random.seed()
   
Records = []
for interval in Intervals:
    for i in range(0, nRecords_per_interval, 1):
        molecule = spherical.generate_random_molecules2(prototype, DMin=interval[0], DMax=interval[1], max_trials=1000)
        if molecule is None:
            break
        Molecules = [prototype, molecule]
        rec = structure.Rec(Molecules)
        Records.append(rec)

print(len(Records))

records = []
for i in Records:
    record = []
    for j in i.Molecules:
        for k in j.Atoms:
            S = k.Atom.Symbol
            x = str(k.x)
            y = str(k.y)
            z = str(k.z)
            line = S + ' ' + x + '\t' + y + '\t' + z + '\n'
            record.append(line)
    record.append('\n')
    for line in record:
        records.append(line)

if type(records) is list:
    if additional_gap == 0:
        str1 = 'No additional gap'
    else:
        str1 = 'Additional gap = ' + str(additional_gap)
    f = open('Coordinates. 2 ' + prototype.Name + ' Molecules. ' + str(len(Records)) + ' Records. ' + str1 + '.x', "w")
    f.writelines(records)
    f.close()
    
# plot 5 randomly choosen records
for i in range(0, 5, 1):
    rand = random.randrange(0, nRecords_per_interval, 1)
    fig1 = plt.figure(i, figsize=(19,10))
    cx = fig1.gca(projection='3d')
    cx.set_xlabel('X Label')
    cx.set_ylabel('Y Label')
    cx.set_zlabel('Z Label')
    spherical.plot_molecules(cx, Records[rand].Molecules)
