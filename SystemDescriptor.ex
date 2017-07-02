#F_data = datafile short.x
#F_data = short.x
#F_data = short three water molecules.x
#F_data = datafile1 from github gaussian process.x
#F_data = datafile2.x
#F_data = datafile3 2 water molecules.x
F_data = datafile4 3 water molecules small.x
#F_data = datafile5 3 water molecules big.x

&FEATURES
&SingleDistances
SingleDistancesInclude = True # Include this class or not
&SingleDistancesDescription
O,O,intermolecular: -1,-2,-3,-4,-5,-6
H,H,intermolecular: -1,-2,-3,-4,-5,-6
O,H,intramolecular: 
H,H,intramolecular:
&endSingleDistancesDescription
&DefaultSingleDistances
SinglePowers: -1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15
&endSingleDistances

&DoubleDistances
DoubleDistancesInclude=True # Include this class or not
&DoubleDistancesDescription
O,O,intermolecular: -1,-2,-3,-4,-5,-6
O,H,intermolecular: -1,-2,-3,-4,-5,-6
H,H,intermolecular: -1,-2,-3,-4,-5,-6
O,H,intramolecular: -1,-2,-3,-4,-5,-6
H,H,intramolecular: -1,-2,-3,-4,-5,-6
&endDoubleDistancesDescription
&DefaultDoubleDistances
DoublePowers: -1,-2,-3,-4,-5
IncludeSameType=True # Include distances of the same type to pairs or not
IncludeAllExcept=False # if true all combinations of distances except below list will be included in the set
ExcludeAllExcept=True # if true all combinations of distances except below list will be excluded from the set
&IncludeExcludeList
O,H,intermolecular; O,H,intermolecular
&endIncludeExcludeList
&endDoubleDistances

&Harmonics
HarmonicsInclude=False
Order: -2,-1,0,1,2
Degree: 0,1,2
# symbol of atom to be a center of coordinate system to calculate harmonics
Center: O
Atoms: O,H
&endHarmonics
&endFEATURES

&SYSTEM
# Atom symbol (string), Atom type(integer), Molecule number (integer)
# water 1
O,1,0
H,0,0    
H,0,0
# water 2
O,1,1
H,0,1    
H,0,1
# water 3
O,1,2
H,0,2    
H,0,2
&END
 