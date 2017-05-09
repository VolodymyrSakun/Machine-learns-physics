# generate coordinate file for H2O, CH4, CO2

import random

Size = 1000 # number of records
F_data = 'H2O CH4 NO2.x'
F = open(F_data, "w")
j = 0
while j < Size:
# H2O
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
# CH4
    Cx = random.random() * 10
    Cy = random.random() * 10
    Cz = random.random() * 10
    H1x = Cx + random.random()
    H1y = Cy + random.random()
    H1z = Cz + random.random()
    H2x = Cx + random.random()
    H2y = Cy + random.random()
    H2z = Cz + random.random()
    H3x = Cx + random.random()
    H3y = Cy + random.random()
    H3z = Cz + random.random()
    H4x = Cx + random.random()
    H4y = Cy + random.random()
    H4z = Cz + random.random()
    rC = 'C: ' + str(Cx) + "\t" + str(Cy) + "\t" + str(Cz) + "\n"
    rH1 = 'H: ' + str(H1x) + "\t" + str(H1y) + "\t" + str(H1z) + "\n"
    rH2 = 'H: ' + str(H2x) + "\t" + str(H2y) + "\t" + str(H2z) + "\n"
    rH3 = 'H: ' + str(H3x) + "\t" + str(H3y) + "\t" + str(H3z) + "\n"
    rH4 = 'H: ' + str(H4x) + "\t" + str(H4y) + "\t" + str(H4z) + "\n"
    F.write(rC)
    F.write(rH1)
    F.write(rH2)
    F.write(rH3)
    F.write(rH4)
# NO2
    Nx = random.random() * 10
    Ny = random.random() * 10
    Nz = random.random() * 10
    O1x = Cx + random.random()
    O1y = Cy + random.random()
    O1z = Cz + random.random()
    O2x = Cx + random.random()
    O2y = Cy + random.random()
    O2z = Cz + random.random()
    rN = 'N: ' + str(Nx) + "\t" + str(Ny) + "\t" + str(Nz) + "\n"
    rO1 = 'O: ' + str(O1x) + "\t" + str(O1y) + "\t" + str(O1z) + "\n"
    rO2 = 'O: ' + str(O2x) + "\t" + str(O2y) + "\t" + str(O2z) + "\n"
    F.write(rN)
    F.write(rO1)
    F.write(rO2)
    
    e = random.random() * 0.001
    energy = str(e) + "\n"
    F.write(energy)
    F.write("\n")
    j += 1
    
F.close()
print("DONE")