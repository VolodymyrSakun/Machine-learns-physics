from project1 import library
from project1 import IOfunctions
import numpy as np
import matplotlib.pyplot as plt
from project1 import structure
from sklearn.metrics import mean_squared_error
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import pandas as pd
from sklearn.metrics import mean_squared_error


Files = structure.Files
Data = structure.Data
# load trained model from Genetic algorithm
#ga = IOfunctions.LoadObject('ga.dat')
# Load trained model from Gaussian process
#gaList = [IOfunctions.LoadObject('ga.dat')]

# Data preprocessing for forecasting dataset
MoleculePrototypes = IOfunctions.ReadMoleculeDescription(F=Files['System descriptor'])
setType = IOfunctions.getSetType(Files['Set'])  
print(setType)
if setType == 'Old':
    RecordsTrain = IOfunctions.ReadRecordMoleculesOld(Files['Set'], MoleculePrototypes) # Read records
else:
    RecordsTrain = IOfunctions.ReadRecordMoleculesNew(Files['Set'], MoleculePrototypes) # Read records 

# Data preprocessing for forecasting dataset
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
featuresDictForecast = IOfunctions.ReadFeatures(Files, FeaturesDict, Forecast=True) 
# center of mass
COM_forecast = IOfunctions.ReadCSV(Files['COM forecast']) 

idxTest = list(np.random.choice(np.arange(len(RecordsTrain)), int(len(RecordsTrain)*0.2)))
RecordsTest = []
for i in idxTest:
    RecordsTest.append(RecordsTrain[i])
     
for i in sorted(idxTest, reverse=True): 
    del(RecordsTrain[i])

IOfunctions.store_records(Files['Training set'], RecordsTrain) # store trained set
IOfunctions.store_records(Files['Test set'], RecordsTest) # store trained set

IOfunctions.store_average_distances(Files['COM train'], RecordsTrain)
IOfunctions.store_average_distances(Files['COM test'], RecordsTest)

# create features for forecasting
FeaturesDict = library.GenerateFeatures(Files, Forecast=False)

# read features ready for forecasting
featuresDictTrainTest = IOfunctions.ReadFeatures(Files, FeaturesDict, Forecast=False) 

Y_train = featuresDictTrainTest['Y_train']
Y_test = featuresDictTrainTest['Y_test']
X_LinearSingle_train = featuresDictTrainTest['X_LinearSingle_train']
X_LinearSingle_test = featuresDictTrainTest['X_LinearSingle_test']
X_Linear_train = featuresDictTrainTest['X_Linear_train']
X_Linear_test = featuresDictTrainTest['X_Linear_test']
FeaturesLinearAll = featuresDictTrainTest['FeaturesLinearAll']
FeaturesLinearReduced = featuresDictTrainTest['FeaturesLinearReduced']

Y_forecast_true = featuresDictForecast['Y_forecast']
X_LinearSingle_forecast = featuresDictForecast['X_LinearSingle_forecast']
X_Linear_forecast = featuresDictForecast['X_Linear_forecast']


# H2ODeepLearningEstimator Classifier H2O  

h2o.init(nthreads=6, max_mem_size=14)
    
nEpoch = 3000
h1 = X_Linear_train.shape[1]*2
h2 = int(h1/2)
if h2 % 2 != 0:
    h2 += 1
classifierH2O = H2ODeepLearningEstimator(hidden=[h1, h2], epochs=nEpoch)  
#classifierH2O = H2ODeepLearningEstimator(hidden=[h1], epochs=nEpoch)  
  
columns = list(np.arange(X_Linear_train.shape[1]))
columns = [str(i) for i in columns]

X_Linear_trainDF = pd.DataFrame(X_Linear_train, columns=columns, dtype=float)
X_Linear_trainDF['Response'] = Y_train
X_Linear_testDF = pd.DataFrame(X_Linear_test, columns=columns, dtype=float)
X_Linear_testDF['Response'] = Y_test
X_Linear_forecastDF = pd.DataFrame(X_Linear_forecast, columns=columns, dtype=float)
#X_Linear_forecastDF['Response'] = Y_forecast_true

trainH2O = h2o.H2OFrame(X_Linear_trainDF)
testH2O = h2o.H2OFrame(X_Linear_testDF)
forecastH2O = h2o.H2OFrame(X_Linear_forecastDF)

classifierH2O.train(x = columns, y = 'Response', training_frame = trainH2O, validation_frame=testH2O)
                        
yForecastH2O = classifierH2O.predict(forecastH2O[columns])
yForecastH2ODF = h2o.as_list(yForecastH2O)
Y_forecast_predict = yForecastH2ODF.values
h2o.cluster().shutdown()

mse = mean_squared_error(Y_forecast_true, Y_forecast_predict)
print(mse)

toPlotFull = pd.DataFrame(columns=['COM', 'Real', 'Predicted'], dtype=float)
toPlotFull['COM'] = COM_forecast
toPlotFull['Real'] = Y_forecast_true
toPlotFull['Predicted'] = Y_forecast_predict
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
plt.title('{} {} {}'.format('Plot', nSamplesPlot, 'random observations'))
plt.legend()
plt.show()


# Keras ANN    
#Hidden1 = X_Linear_train.shape[1]*2
#Hidden2 = X_Linear_train.shape[1]
#if Hidden2 % 2 != 0:
#    Hidden2 += 1
#nEpochs = 1500    
#BatchSize = round(X_Linear_train.shape[0] / 10)
#if BatchSize % 2 != 0:
#    BatchSize += 1    
#print("Keras Artificial neural network")
## Initialising the ANN
#from keras.models import Sequential
#from keras.layers import Dense
#classifier = Sequential()
#    
## Adding the input layer and the first hidden layer
#classifier.add(Dense(Hidden1, input_dim = X_Linear_train.shape[1], activation = 'relu'))   
#classifier.add(Dense(Hidden2, kernel_initializer='normal', activation='relu'))
#classifier.add(Dense(1, kernel_initializer='normal'))
#classifier.compile(loss='mean_squared_error', optimizer='adam')
#classifier.fit(X_Linear_train, Y_train, batch_size = BatchSize, epochs = nEpochs,\
#    shuffle=True, validation_data=(X_Linear_test, Y_test))
    


# Predicting the Test set results
#    y_pred_test = classifier.predict(X_test)
#    y_pred_test = (y_pred_test > 0.5)
#    y_new_pred_ANN = classifier.predict(X_new)
#    










