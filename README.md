# Working with feature selection

* Data files:

SystemDescriptor - descriptor for Generate harmonic features (MP).py

datafile1 from github gaussian process.x (from quantum mechanics)

datafile2.x  (from rotation of molecules)

datafile3.x (two water molecules)

datafile4.x (three water molecules small)

datafile5.x (three water molecules huge)

* Libraries:

class2.py - mostly structures that describe system

library2.py - functions for working with features

genetic.py - functions that are used for genetic algorithm

spherical.py - functions that sre used for constructing spherical harmonics features

* Programs that generate data:

Geneerate random coordinates for N water molecules.py

Generate random coordinates H2O CH4 CO2.py

* Programs that genarate features:

Generate harmonic features.py - generates single, double and harmonic features

Generate harmonic features MP.py - same but uses multiprocessing

* Feature selection / elimination algorithms:

Abbreviation:

MP - code uses multiprocessing and run functions in parallel using 100% of CPU

ENet - Elastic Net algorithm from sklearn

GA - Genetic algorithm (structure.genetic)

BS - Backward sequential selection (elimination) algorithm (structure.library2)

FS - Forward sequential selection algorithm (structure.library2)

BF - Best Fit algorithm (structure.library2)
