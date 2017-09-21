# Working with feature selection

* Data files:

SystemDescriptor - description of system

MoleculeDescriptor - description molecules in system

datafile1 from github gaussian process.x (from quantum mechanics)

datafile2.x  (from rotation of molecules)

datafile3.x (two water molecules)

datafile4.x (three water molecules small)

datafile5.x (three water molecules huge)

* Libraries:

regression.py - objects for linear and non-linear fit

library.py - functions for working with features

genetic.py - functions that are used for genetic algorithm

spherical.py - functions that sre used for constructing spherical harmonics features

IOfunctions.py - Input / output functions

1. Filter data.py - Prepares data for feature generation using filters

2. Generate features.py - Generates linear and exponential features for fitting

3. Fit.py - Fits the dataset, stores results and model and plots fitting path

4. Final Plot.py - Plots additional graphs
