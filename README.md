# Working with feature selection

* Data files:

SystemDescriptor

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

GetFit2.py - chain of algorithms: ElasticNEt, Backward sequential, Best Subset. Now can work in multiprocessing mode.

GetFitEnetGenetic.py - chain of algorithms: ElasticNEt, Genetic, Backward sequential, Best Subset (useful if I have too many features after elastic net)

