from project1 import library
from project1 import structure
from project1 import spherical
from project1 import IOfunctions

if __name__ == '__main__':
# be sure that system.dat is correct and contains structure that corresponds to dataset 
    _, _, _, _, _, system, _ = IOfunctions.ReadFeatures(\
        F_Nonlinear='Distances.csv', F_linear='LinearFeatures.csv', F_Response='Response.csv',\
        F_FeaturesAll='LinearFeaturesAll.dat', F_FeaturesReduced='LinearFeaturesReduced.dat',\
        F_System='system.dat', F_Records=None, verbose=False)
    # read molecules from descriptor file
    MoleculePrototype = IOfunctions.ReadMoleculeDescription(F='MoleculesDescriptor.')
    
    F = 'datafile3 2 water molecules.x'
    RMin=0
    RMax=3.6
    CheckCenterOfMass = False
    nDistances = 3
    # Read coordinates from file
    f = open(F, "r")
    data0 = f.readlines()
    f.close()
    data1 = []
    for i in range(0, len(data0), 1):
        data1.append(data0[i].rstrip())
    del(data0)
    
    # Rearrange data in structure
    i = 0 # counts lines in textdata
    j = 0 # counts atom records for each energy value
    atoms_list = [] # temporary list
    record_list = []
    while i < len(data1):
        s = data1[i].split() # line of text separated in list
        if len(s) == 0: # empty line
            i += 1
            continue
    # record for energy value
        elif (len(s) == 1) and library.isfloat(s[0]): 
            e = float(s[0])
            rec = structure.Record(e, atoms_list)

            # molecules
            l = 0
            molecules = []
            while (l < len(rec.atoms)):
                one_molecule_atoms = []
                molecule_index = rec.atoms[l].Atom.MolecularIndex
    
                while (l < len(rec.atoms)) and (rec.atoms[l].Atom.MolecularIndex == molecule_index):
                    one_molecule_atoms.append(atoms_list[l])
                    l += 1
                for k in range(0, len(one_molecule_atoms), 1):
                    one_molecule_atoms[k].Atom.Mass = MoleculePrototype[0].Atoms[k].Atom.Mass
                    one_molecule_atoms[k].Atom.Radius = MoleculePrototype[0].Atoms[k].Atom.Radius
                molecules.append(structure.Molecule(one_molecule_atoms, MoleculePrototype[0].Name))
            if spherical.check_molecules(molecules, RMin=RMin, RMax=RMax, CheckCenterOfMass=CheckCenterOfMass, nDistances=nDistances):
                record_list.append(rec)
            j = 0
            atoms_list = []
        elif (len(s) == 4): 
            x = float(s[1])
            y = float(s[2])
            z = float(s[3])
            atoms_list.append(structure.AtomCoordinates(system.Atoms[j], x, y, z))
            j += 1
        i += 1
        
    print(len(record_list))    
    records = []
    for i in range(0, len(record_list), 1):
        for j in range(0, len(record_list[i].atoms), 1):
            S = record_list[i].atoms[j].Atom.Symbol
            x = str(record_list[i].atoms[j].x)
            y = str(record_list[i].atoms[j].y)
            z = str(record_list[i].atoms[j].z)
            line = S + ': ' + x + '\t' + y + '\t' + z + '\n'
            records.append(line)
        line = str(record_list[i].e) + '\n'
        records.append(line)
        records.append('\n')

    f = open('Reduced set 1.x', "w")
    f.writelines(records)
    f.close()



