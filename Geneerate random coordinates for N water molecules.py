# generate three water molecules coordinate file

import random

Size = 100 # number of records
NofMolecules = 3 # number of molecules
F_data = 'water 3 molecules coordinates.x'
F = open('water 3 molecules coordinates.x', "w")
j = 0
while j < Size:
    for i in range(0, NofMolecules, 1):
        Ox = random.random() * 10
        Oy = random.random() * 10
        Oz = random.random() * 10
        H1x = Ox + random.random()
        H1y = Oy + random.random()
        H1z = Oz + random.random()
        H2x = Ox + random.random()
        H2y = Oy + random.random()
        H2z = Oz + random.random()
        rO = 'O: ' + str(Ox) + "\t" + str(Oy) + "\t" + str(Oz) + "\n"
        rH1 = 'H: ' + str(H1x) + "\t" + str(H1y) + "\t" + str(H1z) + "\n"
        rH2 = 'H: ' + str(H2x) + "\t" + str(H2y) + "\t" + str(H2z) + "\n"
        F.write(rO)
        F.write(rH1)
        F.write(rH2)
    e = random.random() * 0.001
    energy = str(e) + "\n"
    F.write(energy)
    F.write("\n")
    j += 1
    
    
F.close()
print("DONE")
