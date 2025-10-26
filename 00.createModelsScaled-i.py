# multivariate lstm example
import os

from keras.layers import ConvLSTM2D
from numpy import hstack
from keras.layers import LSTM

import pandas as pd
import numpy as np
import random
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.layers import Input
#from keras.layers.merge import concatenate
from keras.layers import Concatenate

rutaApaper = "/home/aurorax/Git_repos/postdoc/estanciaERAU/paper"
rutaApaper = os.getcwd()


### FUNCTIONS

def myround(x, base=5):
    return int(base * round(float(x)/base))



# split a univariate dataset into train/test sets
def split_dataset(data, factor = 7, n_out = 7):  # Here n_out help us not to run out of numbers
    # split into standard weeks
    trainsize = myround(0.77*len(data),factor) # 77% of the data is train data
    testsize = myround(len(data)-trainsize-n_out,factor)
    train, test = data[0:trainsize], data[(trainsize+1):(trainsize+1+testsize)]
    # restructure into windows of weekly data
    train = array(split(train, len(train)/factor))
    test = array(split(test, len(test)/factor))
    return train, test

def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def FitForecast(X_train, y_train, X_test, N_OUT,
                #hidden_units, dropout, val_split, learning_rate, 
                n_seq,n_substeps, n_features, epochs, trained_model):   
    X_train = X_train.reshape((X_train.shape[0],n_seq, 1, n_substeps, n_features)) 
    #Añadir parada temprana sobreentrenamiento
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_substeps, n_features)))  # here SHAPE is important
    model.add(Flatten())
    model.add(Dense(N_OUT))
    model.compile(optimizer='adam', loss='mse')
    if trained_model is not None:
        model.set_weights(weights = trained_model.get_weights())
    model.fit(X_train, y_train, epochs, verbose=0)
    predictions = list()
    for i in range(len(test_x)):
        # predict the 24h
        yhat_sequence = model.predict(test_x[i].reshape((1, n_seq,1,n_substeps, n_features)), verbose=0)
        # store the predictionsmse.apend(mse0)
        predictions.append(yhat_sequence)
    predictions = array(predictions)
    return model, predictions


## USAGE 

import git # pip install gitpython
def lsremote(url):
    remote_refs = {}
    g = git.cmd.Git()
    for ref in g.ls_remote(url).split('\n'):
        hash_ref_list = ref.split('\t')
        remote_refs[hash_ref_list[1]] = hash_ref_list[0]
    return remote_refs


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def compute_metrics_fn(y_valid_resc, y_hat_resc):
    ## actual train and test values
    mae_ = mean_absolute_error(y_valid_resc, y_hat_resc)
    mse_ = mean_squared_error(y_valid_resc, y_hat_resc)
    rmse_ = mean_squared_error(y_valid_resc, y_hat_resc, squared = False)
    cvrmse_ = rmse_/np.mean(y_valid_resc)*100 # it is a percentage
    mape_ = mape(y_valid_resc, y_hat_resc)
    return mae_, mse_, rmse_, cvrmse_, mape_


labelsAndDist = pd.read_csv(rutaApaper+'/k-prod/kprototypes-labels-dist-iii15.csv', sep=";")

### Crear los k modelos
k = max(labelsAndDist[labelsAndDist.columns[1]])+1
# this is only for printing the name of the main buildings that are selected as centroids
for i in range(0,k):
    closerToThePrototype = labelsAndDist[labelsAndDist[labelsAndDist.columns[1]] == i][labelsAndDist.columns[0]].idxmin()
    refs = lsremote('https://bitbucket.org/aurorax/datangi')
    lastCommit = refs['HEAD']
    infoBuildings = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-energybuildingsEnergyInfo.csv')
    names = infoBuildings['uid']
    building = names[closerToThePrototype]
    print(building)
    
for i in range(0,k):
    closerToThePrototype = labelsAndDist[labelsAndDist[labelsAndDist.columns[1]] == i][labelsAndDist.columns[0]].idxmin()
    refs = lsremote('https://bitbucket.org/aurorax/datangi')
    lastCommit = refs['HEAD']
    infoBuildings = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-energybuildingsEnergyInfo.csv')
    names = infoBuildings['uid']
    building = names[closerToThePrototype]
    print(building)
    cons = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-energy/'+building+'-tsCons.csv', sep=';', index_col=0)
    temp = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-temperature/t'+building+'-ts.csv', sep=';', index_col=0)
    merge=pd.merge(cons,temp, how='outer', left_index=True, right_index=True)
    dataset= merge
    dataset = dataset.dropna()
    fooc = dataset.iloc[:,0].values
    foot = dataset.iloc[:,1].values
    N_OUT =24
    data = dataset.values
    factor = N_OUT
    n_out = N_OUT
    trainsize = myround(0.77*len(data),factor) # 77% of the data is train data
    testsize = myround(len(data)-trainsize-n_out,factor)
    train, test = data[0:trainsize], data[(trainsize+1):(trainsize+1+testsize)]
    scaler_cons = StandardScaler()
    scaler_temp = StandardScaler()
    consScaledtr = scaler_cons.fit_transform(train[:,0].reshape(-1,1))
    tempScaledtr = scaler_temp.fit_transform(train[:,1].reshape(-1,1))
    consScaledte = scaler_cons.transform(test[:,0].reshape(-1,1))
    tempScaledte = scaler_temp.transform(test[:,1].reshape(-1,1))
    train = np.column_stack((consScaledtr, tempScaledtr))
    test = np.column_stack((consScaledte, tempScaledte))
    train = array(split(train, len(train)/factor))
    test = array(split(test, len(test)/factor))
    n_input=N_OUT*7
    train_x, train_y = to_supervised(train, n_input, n_out=N_OUT)
    X = train_x
    y = train_y
    n_features = X.shape[2]
    test_x, test_y = to_supervised(test, n_input, n_out=N_OUT)
    n_seq = 7
    n_epochs=5
    mod, pred = FitForecast(X, y, test_x, 24, n_seq,24,n_features,n_epochs, None)

    mod_json = mod.to_json()
    test_y_resc = scaler_cons.inverse_transform(test_y)
    print("lo hizo")
    pred = pred.reshape(pred.shape[0],pred.shape[2])
    pred_resc = scaler_cons.inverse_transform(pred)
    mae_, mse_, rmse_, cvrmse_, mape_ = compute_metrics_fn(test_y_resc.reshape(-1), pred_resc.reshape(-1))
    with open(rutaApaper+"/transfer-learning/modelsProtoScaled5/mod"+str(i)+".json", "w") as json_file:
        json_file.write(mod_json)
    # serialize weights to HDF5
    mod.save_weights(rutaApaper+"/transfer-learning/modelsProtoScaled5/mod"+str(i)+".weights.h5")
    print("Saved model"+str(i)+" to disk")
    pd.DataFrame([mae_, mse_, rmse_, cvrmse_, mape_ ]).to_csv(rutaApaper+'/transfer-learning/modelsProtoScaled5/mod'+str(i)+'_metrics.csv', header=None, index=None)

