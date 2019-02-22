from project1 import library
from project1 import IOfunctions
import numpy as np
import matplotlib.pyplot as plt
from project1 import structure
from sklearn.metrics import mean_squared_error
import pandas as pd

Files = structure.Files
Data = structure.Data
# load trained model from Genetic algorithm
ga = IOfunctions.LoadObject('ga.dat')
# Load trained model from Gaussian process
gaList = [IOfunctions.LoadObject('ga.dat')]

# Data preprocessing for forecasting dataset
MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=Files['System descriptor'])
setType = IOfunctions.getSetType(Files['Forecast'])  
print(setType)
if setType == 'Old':
    RecordsForecast = IOfunctions.ReadRecordMoleculesOld(Files['Forecast'], MoleculePrototypes) # Read records
else:
    RecordsForecast = IOfunctions.ReadRecordMoleculesNew(Files['Forecast'], MoleculePrototypes) # Read records      
IOfunctions.store_records(Files['Forecast set'], RecordsForecast) # store trained set
IOfunctions.store_average_distances(Files['COM forecast'], RecordsForecast)
# create features for forecasting
FeaturesDict = library.GenerateFeatures(Files, Forecast=True)
# read features ready for forecasting
featuresDict = IOfunctions.ReadFeatures(Files, FeaturesDict, Forecast=True) 
# center of mass
COM_forecast = IOfunctions.ReadCSV(Files['COM forecast']) 
# show results and compare predicted and true values of energy
yTrue = featuresDict['Y_forecast']
for i in range(0, len(ga.DecreasingChromosomes), 1):
    # ga.DecreasingChromosomes contains a list of trained models
    # each model is the best that GA could find for particular number of predictors
    # Number of predictors is number of Genes: len(ga.DecreasingChromosomes[i].Genes)
    yPred = ga.DecreasingChromosomes[i].predict(x_lin=featuresDict['X_Linear_forecast'])
    mse = mean_squared_error(yTrue, yPred)
    print('N predictors: ', len(ga.DecreasingChromosomes[i].Genes), 'MSE: ', mse)

chromosomeNumber = 3
yPred = ga.DecreasingChromosomes[chromosomeNumber].predict(x_lin=featuresDict['X_Linear_forecast'])

toPlotFull = pd.DataFrame(columns=['COM', 'Real', 'Predicted'], dtype=float)
toPlotFull['COM'] = COM_forecast
toPlotFull['Real'] = yTrue
toPlotFull['Predicted'] = yPred
toPlotFull = toPlotFull.reindex(np.random.permutation(toPlotFull.index)) # shuffle 
toPlotFull.reset_index(drop=True, inplace=True) # reset index  

nSamplesPlot = 100
toPlot = toPlotFull[toPlotFull.index < nSamplesPlot].copy(deep=True)

toPlot.sort_values(['COM'], ascending=True, na_position='first', inplace=True, kind='quicksort')
toPlot.reset_index(drop=True, inplace=True) # reset index  

fig = plt.figure(1, figsize=(19, 10))
#plt.scatter(toPlot.COM.loc[0 : nSamplesPlot].values, toPlot.Predicted.loc[0 : nSamplesPlot].values, s=1, c='g', label='Predicted')
#plt.scatter(toPlot.COM.loc[0 : nSamplesPlot].values, toPlot.Real.loc[0 : nSamplesPlot].values, s=1, c='r', label='True')
plt.plot(toPlot.COM.loc[0 : nSamplesPlot].values, toPlot.Predicted.loc[0 : nSamplesPlot].values, c='g', label='Predicted')
plt.plot(toPlot.COM.loc[0 : nSamplesPlot].values, toPlot.Real.loc[0 : nSamplesPlot].values, c='r', label='True')
plt.xlabel('Distance between center of masses')
plt.ylabel('Energy')
plt.title('{} {} {} {} {}'.format('Plot', nSamplesPlot, 'random observations',\
          'Number of predictors=', len(ga.DecreasingChromosomes[chromosomeNumber].Genes)))
plt.legend()
plt.show()





