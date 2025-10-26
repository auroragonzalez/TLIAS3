import math
import os
from typing import Optional, Tuple, Dict
import numpy
import flwr as fl
import matplotlib.pyplot as plt

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
from keras.models import Model
from keras.layers import Input
# from keras.layers.merge import concatenate

from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from numpy import savetxt
import statistics
from statistics import mode

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
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.layers import Input
from keras.models import model_from_json

from keras.optimizers import Adam

# from keras.layers.merge import concatenate

dir_base = os.getcwd()

lr = 0.001
alpha = 2.9

def set_alpha(alp):
    g = globals()
    g['alpha'] = alp

def set_lr(lr):
    g = globals()
    g['lr'] = lr

###### Funciones create models
def myround(x, base=5):
    return int(base * round(float(x) / base))


# split a univariate dataset into train/test sets
def split_dataset(data, factor=7, n_out=7):  # Here n_out help us not to run out of numbers
    # split into standard weeks
    trainsize = myround(0.77 * len(data), factor)  # 77% of the data is train data
    testsize = myround(len(data) - trainsize - n_out, factor)
    train, test = data[0:trainsize], data[(trainsize + 1):(trainsize + 1 + testsize)]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / factor))
    test = array(split(test, len(test) / factor))
    return train, test


def to_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
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
        if end_ix > len(sequences) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def FitForecast(X_train, y_train, X_test, N_OUT,
                # hidden_units, dropout, val_split, learning_rate,
                n_seq, n_substeps, n_features, epochs, trained_model):
    X_train = X_train.reshape((X_train.shape[0], n_seq, 1, n_substeps, n_features))
    # Añadir parada temprana sobreentrenamiento
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu',
                         input_shape=(n_seq, 1, n_substeps, n_features)))  # here SHAPE is important
    model.add(Flatten())
    model.add(Dense(N_OUT))
    model.compile(optimizer='adam', loss='mse')
    if trained_model is not None:
        model.set_weights(weights=trained_model.get_weights())
    model.fit(X_train, y_train, epochs, verbose=0)
    predictions = list()
    for i in range(len(X_test)):  # seria X_test?
        # predict the 24h
        yhat_sequence = model.predict(X_test[i].reshape((1, n_seq, 1, n_substeps, n_features)), verbose=0)
        # store the predictionsmse.apend(mse0)
        predictions.append(yhat_sequence)
    predictions = array(predictions)
    return model, predictions


def create_model(N_OUT, n_seq, n_substeps, n_features):
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu',
                         input_shape=(n_seq, 1, n_substeps, n_features)))  # here SHAPE is important
    model.add(Flatten())
    model.add(Dense(N_OUT))
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse')
    return model


def FitForecast_e(model, X_train, y_train, X_test, N_OUT,
                  # hidden_units, dropout, val_split, learning_rate,
                  n_seq, n_substeps, n_features, epochs,number, trained_model):
    X_train = X_train.reshape((X_train.shape[0], n_seq, 1, n_substeps, n_features))
    if trained_model is not None:
        model.set_weights(weights=trained_model.get_weights())

    if number == 1:
        model.fit(X_train, y_train, epochs=epochs, verbose=0)
    else:
        model.fit(X_train, y_train, epochs=epochs, verbose=0)
    predictions = list()

    for i in range(len(X_test)):  # seria X_test?
        # predict the 24h
        yhat_sequence = model.predict(X_test[i].reshape((1, n_seq, 1, n_substeps, n_features)), verbose=0)
        # store the predictionsmse.apend(mse0)
        predictions.append(yhat_sequence)
    predictions = array(predictions)
    return model, predictions


os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git  # pip install gitpython


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
    rmse_ = mean_squared_error(y_valid_resc, y_hat_resc, squared=False)
    cvrmse_ = rmse_ / np.mean(y_valid_resc) * 100  # it is a percentage
    mape_ = mape(y_valid_resc, y_hat_resc)
    return mae_, mse_, rmse_, cvrmse_, mape_


#####

##### Codigo K-prototypes
def kprot_missing(X, n_clusters, cat_index, centroids,
                  max_iter=10):  # cat_index has to be a vector: [5], or [3,4,5] , never an integer
    """Perform K-Means clustering on data with missing values.
    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.
    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """

    missing_c = ~np.isfinite(centroids)
    mu_c = np.nanmean(centroids, 0, keepdims=1)  # average values for each feature
    modes_c = np.apply_along_axis(mode, 0, centroids)  # mode of each feature
    mu_c = mu_c.reshape(
        modes_c.shape[0])  # this is for chaning the shape for later on (3 lines), but the values stays as they are
    num_index_c = list(
        set(range(0, centroids.shape[1])) ^ set(
            cat_index))  # the index of numerical features (complementary to cat_index)
    filling_c = mu_c
    filling_c[cat_index] = modes_c[cat_index]  # we create a vector with mean and mode of each feature
    centroids = np.where(missing_c, filling_c, centroids)

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)  # average values for each feature
    modes = np.apply_along_axis(mode, 0, X)  # mode of each feature
    mu = mu.reshape(
        modes.shape[0])  # this is for chaning the shape for later on (3 lines), but the values stays as they are
    num_index = list(
        set(range(0, X.shape[1])) ^ set(cat_index))  # the index of numerical features (complementary to cat_index)
    filling = mu
    filling[cat_index] = modes[cat_index]  # we create a vector with mean and mode of each feature
    X_hat = np.where(missing, filling, X)


    #init='Huang' random_state=np.random.seed(random_state)
    centroids_n, centroids_c = split_num_cat2(centroids, categorical=[1, 2, 3, 5, 7, 8, 10])
    centroids = [centroids_n, centroids_c]
    print(centroids)

    cls = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=2)
    labels = cls.fit_predict(X_hat, categorical=list(cat_index))
    centroids = cls._enc_cluster_centroids
    print(centroids)
    centroids_num = centroids[0]
    centroids_cat = centroids[1]
    centroids_all = np.empty([centroids_num.shape[0], centroids_num.shape[1] + centroids_cat.shape[1]])
    centroids_all[:, cat_index] = centroids_cat
    centroids_all[:, num_index] = centroids_num
    X_hat[missing] = centroids_all[labels][missing]
    gamma = cls.gamma

    # initialize KMeans with the previous set of centroids. this is much
    # faster and makes it easier to check convergence (since labels
    # won't be permuted on every iteration), but might be more prone to
    # getting stuck in local minima.
    """
    for i in range(max_iter):
        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat, categorical=list(cat_index))
        centroids = cls._enc_cluster_centroids
        #print(centroids)
        centroids_num = centroids[0]
        centroids_cat = centroids[1]
        centroids_all = np.empty([centroids_num.shape[0], centroids_num.shape[1] + centroids_cat.shape[1]])
        centroids_all[:, cat_index] = centroids_cat
        centroids_all[:, num_index] = centroids_num
        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids_all[labels][missing]
        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break
        prev_labels = labels
        prev_centroids = centroids
        gamma = cls.gamma
        print(prev_centroids)
        cls = KPrototypes(n_clusters=n_clusters, init=prev_centroids, verbose=2)
    """
    return labels, centroids_all, X_hat, gamma, cls


def split_num_cat2(X, categorical):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.
    :param X: Feature matrix
    :param categorical: Indices of categorical columns
    """
    Xnum = np.asanyarray(X[:, [ii for ii in range(X.shape[1])
                               if ii not in categorical]]).astype(np.float64)
    Xcat = np.asanyarray(X[:, categorical])
    return Xnum, Xcat


def labels_cost2(Xnum, Xcat, centroids, num_dissim, cat_dissim, gamma, labels, membship=None):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """
    n_points = Xnum.shape[0]
    # Xnum = check_array(Xnum)
    tot_costs = np.empty(n_points)
    num_index = [0, 4, 6, 9]
    cat_index = [1, 2, 3, 5, 7, 8, 10]
    for ipoint in range(n_points):
        # Numerical cost = sum of Euclidean distances
        num_costs = num_dissim(centroids[labels[ipoint]][num_index], Xnum[ipoint])
        cat_costs = cat_dissim(centroids[labels[ipoint]][cat_index], Xcat[ipoint], X=Xcat, membship=membship)
        # Gamma relates the categorical cost to the numerical cost.
        tot_costs[ipoint] = num_costs + gamma * cat_costs
    return tot_costs


gitrepo = "/home/aurorax/Git_repos/"


def make_Kpods():
    """
    study1 = pd.read_csv(
        gitrepo + 'postdoc/estanciaERAU/bibliografia/energy/study1/temporal-features-for-nonres-buildings-library-master/data/raw/meta_open.csv',
        sep=',', index_col=0)
    """
    study1 = pd.read_csv(dir_base + "/k-prod/meta_open.csv", sep=',', index_col=0)
    vbles = ["energystarscore", "heatingtype", "industry",
             "numberoffloors", "occupants", "rating",
             "sqm", "subindustry", "timezone", "yearbuilt", "primaryspaceuse_abbrev"]
    df = study1[vbles]
    # type: num, cat, cat
    # type: cat, num, cat
    # type: num, cat, cat, num, cat
    # cat_index = [1,2,3,5,7,8,10]
    """
    study2 = pd.read_csv(gitrepo + 'postdoc/estanciaERAU/bibliografia/energy/study4:energy+occupation/meta_open.csv',
                         sep=',', index_col=0) #Esta en i-blend
    """
    study2 = pd.read_csv(dir_base + "/00.data/I-BLEND/meta_open.csv", sep=',', index_col=0)
    df2 = study2[vbles]
    df3 = df.append(df2)
    df = df3

    def heatingtype_to_numeric(x):
        if x == 'Biomass':
            return 1
        if x == 'District heating':
            return 2
        if x == 'District Heating':
            return 2
        if x == 'Electric':
            return 3
        if x == 'Electricity':
            return 3
        if x == 'Gas':
            return 4
        if x == 'Heat network':
            return 5
        if x == 'Heat network and steam':
            return 6
        if x == 'Oil':
            return 7

    df['heatingtype'] = df['heatingtype'].apply(heatingtype_to_numeric)

    def industry_to_numeric(x):
        if x == 'Commercial Property':
            return 1
        if x == 'Education':
            return 2
        if x == 'Government':
            return 3

    df['industry'] = df['industry'].apply(industry_to_numeric)

    def rating_to_numeric(x):
        if x == 'B':
            return 1
        if x == 'C':
            return 2
        if x == 'D':
            return 3
        if x == 'E':
            return 4
        if x == 'F':
            return 5
        if x == 'G':
            return 6

    df['rating'] = df['rating'].apply(rating_to_numeric)

    def subindustry_to_numeric(x):
        if x == 'Bank/Financial Services':
            return 1
        if x == 'Business Services':
            return 2
        if x == 'College/University':
            return 3
        if x == 'Commercial Real Estate':
            return 4
        if x == 'Corporate Office':
            return 5
        if x == 'Other Government Buildings':
            return 6
        if x == 'Primary/Secondary School':
            return 7
        if x == 'Social Services':
            return 8

    df['subindustry'] = df['subindustry'].apply(subindustry_to_numeric)

    def timezone_to_numeric(x):
        if x == 'America/Chicago':
            return 1
        if x == 'America/Denver':
            return 2
        if x == 'America/Los_Angeles':
            return 3
        if x == 'America/New_York':
            return 4
        if x == 'America/Phoenix':
            return 5
        if x == 'Asia/Singapore':
            return 6
        if x == 'Europe/London':
            return 7
        if x == 'Europe/Zurich':
            return 8

    df['timezone'] = df['timezone'].apply(timezone_to_numeric)

    def yearbuilt_to_numeric(x):
        if x == '11th Century onwards':
            return 1200
        if x == '1862-1875':
            return 1868
        if x == '1888-1890':
            return 1889
        if x == '1898-1902':
            return 1900
        if x == '1903-1906':
            return 1904
        if x == '1913-1915':
            return 1914
        if x == '1919-1945':
            return 1932
        if x == '1945-1966':
            return 1955
        if x == '1967-1976':
            return 1971
        if x == 'post 1976':
            return 1982
        if x == 'Post 1976':
            return 1982
        if x == 'pre 1919':
            return 1910
        if x == 'Pre 1919':
            return 1910

    df['yearbuilt'] = df['yearbuilt'].apply(yearbuilt_to_numeric)

    def primaryspaceuse_abbrev_to_numeric(x):
        if x == 'Office':
            return 1
        if x == 'PrimClass':
            return 2
        if x == 'UnivClass':
            return 3
        if x == 'UnivDorm':
            return 4
        if x == 'UnivLab':
            return 5
        if x == 'UnivLib':
            return 6
        if x == 'UnivDining':
            return 7

    def euclidean_dissim(a, b, **_):
        """Euclidean distance dissimilarity function"""
        if np.isnan(a).any() or np.isnan(b).any():
            raise ValueError("Missing values detected in numerical columns.")
        return np.sum((a - b) ** 2)

    def matching_dissim(a, b, **_):
        """Simple matching dissimilarity function"""
        return np.sum(a != b)

    df['primaryspaceuse_abbrev'] = df['primaryspaceuse_abbrev'].apply(primaryspaceuse_abbrev_to_numeric)
    lista = ['PrimClass_Julianna', 'UnivClass_Anya', 'Office_Gustavo',
             'PrimClass_Jaylin', 'UnivDorm_Marquis', 'PrimClass_Ervin',
             'PrimClass_Jacquelyn', 'Office_Maximus', 'PrimClass_Johnathan',
             'UnivLab_Marie', 'Office_Erik', 'PrimClass_Johnathon',
             'UnivDorm_Alka', 'Office_Benthe', 'Office_Jude']
    df_centroids = df.loc[lista]
    df_centroids = df_centroids.to_numpy()
    df_centroids = np.array(df_centroids, dtype=float)

    df = df.to_numpy()
    df = np.array(df, dtype=float)
    labels, centroids, X_hat, gamma, modelF = kprot_missing(df, n_clusters=15, cat_index=[1, 2, 3, 5, 7, 8, 10],
                                                             centroids=df_centroids,max_iter=1000)
    Xnum, Xcat = split_num_cat2(X_hat, categorical=[1, 2, 3, 5, 7, 8, 10])
    num_dissim = euclidean_dissim
    cat_dissim = matching_dissim
    cat_index = [1, 2, 3, 5, 7, 8, 10]
    num_index = list(set(range(0, X_hat.shape[1])) ^ set(cat_index))
    # centroidsCat = centroids[cat_index]
    # distance between points and their cluster prototype
    distances = labels_cost2(Xnum, Xcat, centroids, num_dissim, cat_dissim, gamma, labels)

    df2 = pd.DataFrame(df)
    X_hat2 = pd.DataFrame(X_hat)
    labels2dist = pd.DataFrame(labels, distances)
    centroids2 = pd.DataFrame(centroids)

    df2.to_csv("kprototypes-df-iii15.csv", sep=';')
    X_hat2.to_csv("kprototypes-X_hat-iii15.csv", sep=';')
    labels2dist.to_csv(dir_base + '/k-prod/kprototypes-labels-dist-iii15.csv', sep=";")
    centroids2.to_csv("kprototypes-centroids-iii15.csv", sep=';')
    labelsAndDist = pd.read_csv(dir_base + '/k-prod/kprototypes-labels-dist-iii15.csv', sep=";")
    k = max(labelsAndDist[labelsAndDist.columns[1]]) + 1
    return k


def create_clients(kpods):
    print("leyendo clientes")
    #labelsAndDist = pd.read_csv(dir_base + '/k-prod/kprototypes-labels-dist-iii15.csv', sep=";")
    labelsAndDist = pd.read_csv(dir_base + '/01.k-prot-clustering/kprototypes-labels-dist-iii15.csv', sep=",")
    #print(labelsAndDist)

    k = kpods  # + 1
    clients = []
    lastCommits = []
    #print(len(labelsAndDist))
    for i in range(0, k):
        print(i)
        closerToThePrototype = labelsAndDist[labelsAndDist[labelsAndDist.columns[1]] == i][
            labelsAndDist.columns[0]].idxmin()

        refs = lsremote('https://bitbucket.org/aurorax/datangi')
        lastCommit = refs['HEAD']
        lastCommits.append(lastCommit)
        infoBuildings = pd.read_csv(
            'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-energybuildingsEnergyInfo.csv')
        names = infoBuildings['uid']
        #print(closerToThePrototype)
        #print(len(names))
        building = names[closerToThePrototype]
        #print(building)

        clients.append(building)

    #print(clients)
    return clients, lastCommits


def read_client(lista_clientes, lastCommits, client):
    building = lista_clientes[client - 1]
    lastCommit = lastCommits[client - 1]
    cons = pd.read_csv(
        'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-energy/' + building + '-tsCons.csv',
        sep=';', index_col=0)
    temp = pd.read_csv(
        'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-temperature/t' + building + '-ts.csv',
        sep=';', index_col=0)
    merge = pd.merge(cons, temp, how='outer', left_index=True, right_index=True)
    dataset = merge
    dataset = dataset.dropna()
    #cuidado quitar cuando no haga falta
    dataset=dataset[-2000:]
    #
    fooc = dataset.iloc[:, 0].values
    foot = dataset.iloc[:, 1].values
    N_OUT = 24
    data = dataset.values
    factor = N_OUT
    n_out = N_OUT
    trainsize = myround(0.77 * len(data), factor)  # 77% of the data is train data
    testsize = myround(len(data) - trainsize - n_out, factor)
    train, test = data[0:trainsize], data[(trainsize + 1):(trainsize + 1 + testsize)]
    scaler_cons = StandardScaler()
    scaler_temp = StandardScaler()
    consScaledtr = scaler_cons.fit_transform(train[:, 0].reshape(-1, 1))
    tempScaledtr = scaler_temp.fit_transform(train[:, 1].reshape(-1, 1))
    consScaledte = scaler_cons.transform(test[:, 0].reshape(-1, 1))
    tempScaledte = scaler_temp.transform(test[:, 1].reshape(-1, 1))
    train = np.column_stack((consScaledtr, tempScaledtr))
    test = np.column_stack((consScaledte, tempScaledte))
    train = array(split(train, len(train) / factor))
    test = array(split(test, len(test) / factor))
    n_input = N_OUT * 7
    train_x, train_y = to_supervised(train, n_input, n_out=N_OUT)
    X = train_x
    y = train_y
    n_features = X.shape[2]
    test_x, test_y = to_supervised(test, n_input, n_out=N_OUT)
    n_seq = 7
    model = create_model(N_OUT, n_seq, 24, n_features)
    return (train_x, train_y), (test_x, test_y), n_features, model, scaler_cons, building


def split_dataset(data, factor=7, n_out=7):  # Here n_out help us not to run out of numbers
    # split into standard weeks
    trainsize = myround(0.77 * len(data), factor)  # 77% of the data is train data
    testsize = myround(len(data) - trainsize - n_out, factor)
    train, test = data[0:trainsize], data[(trainsize + 1):(trainsize + 1 + testsize)]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / factor))
    test = array(split(test, len(test) / factor))
    return train, test


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger


def FitForecast2(model, X_train, y_train, X_test, N_OUT,
                 # hidden_units, dropout, val_split, learning_rate,
                 n_seq, n_substeps, n_features, epochs, trained_model):
    checkpoint = ModelCheckpoint("./models/InceptionV3-{epoch:02d}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy',
                                 mode='max', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=500, monitor='val_accuracy', mode='max', restore_best_weights=True)
    tensorboard = TensorBoard(log_dir="./results/", histogram_freq=0, write_graph=True, write_images=True)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=70, min_lr=6.25e-6)
    csv_logger = CSVLogger("./results/training_metrics.csv")
    X_train = X_train.reshape((X_train.shape[0], n_seq, 1, n_substeps, n_features))
    # Añadir parada temprana sobreentrenamiento

    if trained_model is not None:
        model.set_weights(weights=trained_model.get_weights())
    model.fit(X_train, y_train, epochs, verbose=0,
              callbacks=[checkpoint, early_stopping, tensorboard, reduce_lr, csv_logger])
    predictions = list()
    for i in range(len(X_test)):
        # predict the 24h
        yhat_sequence = model.predict(X_test[i].reshape((1, n_seq, 1, n_substeps, n_features)), verbose=0)
        # store the predictionsmse.apend(mse0)
        predictions.append(yhat_sequence)
    predictions = array(predictions)
    return model, predictions


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

def transfer_learning(cluster, lastCommits, resWO, resW, count, n_muestras):
    print("parte transfer learning cluster: " + str(cluster))

    i = cluster
    car_metricas = "metricas_TL"
    try:
        os.mkdir(car_metricas)
    except FileExistsError:
        aux = 0

    car_metricas = "metricas_TL/cluster_"+str(i)
    try:
        os.mkdir(car_metricas)
    except FileExistsError:
        aux = 0

    #labelsAndDist = pd.read_csv(dir_base + '/k-prod/kprototypes-labels-dist-iii15.csv', sep=";")
    labelsAndDist = pd.read_csv(dir_base + '/01.k-prot-clustering/kprototypes-labels-dist-iii15.csv', sep=";")
    lastCommit = lastCommits[cluster-1]
    infoBuildings = pd.read_csv(
        'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-energybuildingsEnergyInfo.csv')
    names = infoBuildings['uid']
    indx = labelsAndDist[labelsAndDist[labelsAndDist.columns[1]] == i].index  # index of the buildings that belong to cluster "i"
    indx = indx[indx < 508]  # momentaneo, arreglar

    json_file = open(dir_base + "/transfer-learning/modelsProtoScaled5/mod" + str(cluster) + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(dir_base + "/transfer-learning/modelsProtoScaled5/mod" + str(cluster) + ".h5")
    print("Loaded model " + str(i) + " from disk")

    for nm in indx:
        print(nm)
        building = names[nm]
        print(building)
        cons = pd.read_csv(
            'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-energy/' + building + '-tsCons.csv',
            sep=';', index_col=0)
        temp = pd.read_csv(
            'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-temperature/t' + building + '-ts.csv',
            sep=';', index_col=0)
        merge = pd.merge(cons, temp, how='outer', left_index=True, right_index=True)
        dataset = merge
        dataset = dataset.dropna()
        n_samples = n_muestras
        dataset = dataset[n_samples:]
        fooc = dataset.iloc[:, 0].values
        foot = dataset.iloc[:, 1].values
        N_OUT = 24
        data = dataset.values
        factor = N_OUT
        n_out = N_OUT
        trainsize = myround(0.77 * len(data), factor)  # 77% of the data is train data
        testsize = myround(len(data) - trainsize - n_out, factor)
        train, test = data[0:trainsize], data[(trainsize + 1):(trainsize + 1 + testsize)]
        scaler_cons = StandardScaler()
        scaler_temp = StandardScaler()
        consScaledtr = scaler_cons.fit_transform(train[:, 0].reshape(-1, 1))
        tempScaledtr = scaler_temp.fit_transform(train[:, 1].reshape(-1, 1))
        consScaledte = scaler_cons.transform(test[:, 0].reshape(-1, 1))
        tempScaledte = scaler_temp.transform(test[:, 1].reshape(-1, 1))
        train = np.column_stack((consScaledtr, tempScaledtr))
        test = np.column_stack((consScaledte, tempScaledte))
        train = array(split(train, len(train) / factor))
        test = array(split(test, len(test) / factor))
        n_input = N_OUT * 7
        train_x, train_y = to_supervised(train, n_input, n_out=N_OUT)
        X = train_x
        y = train_y
        n_features = X.shape[2]
        test_x, test_y = to_supervised(test, n_input, n_out=N_OUT)
        n_seq = 7
        n_epochs = 300
        model = create_model(N_OUT, n_seq, 24, n_features)
        mod, pred = FitForecast_e(X, y, test_x, 24, n_seq, 24, n_features, n_epochs,0, None)
        test_y_resc = scaler_cons.inverse_transform(test_y.reshape(-1, 1))
        pred_resc = scaler_cons.inverse_transform(pred.reshape(-1, 1))
        mae_, mse_, rmse_, cvrmse_, mape_ = compute_metrics_fn(test_y_resc.reshape(-1), pred_resc.reshape(-1))
        print(cvrmse_)
        resWO[count, :] = [i, nm, mae_, mse_, rmse_, cvrmse_, mape_]
        aux = []
        path = dir_base + "/" + car_metricas + "/edificio_" + str(nm) + ".csv"
        col_name = ['Cargado','MAE', 'MSE', 'RSME', 'CVRMSE', 'MAPE', "Muestras"]
        lista = {
            'Cargado': "Si",
            "MAE": mae_,
            "MSE": mse_,
            "RSME": rmse_,
            "CVRMSE": cvrmse_,
            "MAPE": mape_,
            "Muestras": n_samples
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

        model = create_model(N_OUT, n_seq, 24, n_features)
        mod, pred = FitForecast_e(model, X, y, test_x, 24, n_seq, 24, n_features, n_epochs,0, loaded_model)
        pred_resc = scaler_cons.inverse_transform(pred.reshape(-1, 1))
        mae_, mse_, rmse_, cvrmse_, mape_ = compute_metrics_fn(test_y_resc.reshape(-1), pred_resc.reshape(-1))
        print(cvrmse_)
        resW[count, :] = [i, nm, mae_, mse_, rmse_, cvrmse_, mape_]
        count = count + 1
        """
        # mod_json = mod.to_json()
        test_y_resc = scaler_cons.inverse_transform(test_y.reshape(-1, 1))
        #pred_resc = scaler_cons.inverse_transform(pred)
        pred_resc = scaler_cons.inverse_transform(pred.reshape(-1, 1))
        mae_, mse_, rmse_, cvrmse_, mape_ = compute_metrics_fn(test_y_resc.reshape(-1), pred_resc.reshape(-1))
        """
        aux = []
        path = dir_base + "/" +car_metricas+ "/edificio_"+str(nm)+".csv"
        col_name = ['Cargado','MAE', 'MSE', 'RSME', 'CVRMSE', 'MAPE', "Muestras"]
        lista = {
            'Cargado': "No",
            "MAE": mae_,
            "MSE": mse_,
            "RSME": rmse_,
            "CVRMSE": cvrmse_,
            "MAPE": mape_,
            "Muestras": n_samples
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

        print(cvrmse_)
        #resWO[count, :] = [i, nm, mae_, mse_, rmse_, cvrmse_, mape_]







class tfmlpClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, n_seq, n_features, scaler, dir_met, party_number):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_seq = n_seq
        self.n_features = n_features
        self.dir_met = dir_met
        self.scaler = scaler

        self.party_number = party_number

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        # self.model.set_weights(parameters)
        mean_weights = parameters

        # Get hyperparameters for this round
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        rnd = config["round"]
        steps = config["val_steps"]
        rondas = config["rounds"]
        """
        if rnd == 1:
            self.model.set_weights(parameters)
        """
        print("Ronda: " + str(rnd))
        # Train the model using hyperparameters from config
        if rnd > 1:
            for epoch in range(epochs):

                self.model, pred = FitForecast_e(self.model, self.x_train, self.y_train, self.x_test, 24, self.n_seq,
                                                 24, self.n_features, epochs, self.party_number, None)

                #theta = 0.998
                theta = 1/(1+(alpha*lr))
                #theta = alpha
                new_param = fedplus(self.model.get_weights(), mean_weights, theta)
                self.model.set_weights(new_param)
        else:
            # Train the model using hyperparameters from config
            self.model, pred = FitForecast_e(self.model, self.x_train, self.y_train, self.x_test, 24, self.n_seq, 24,
                                             self.n_features, epochs, self.party_number, None)

            new_param = self.model.get_weights()

        parameters_prime = new_param
        num_examples_train = len(self.x_train)
        results = {
            "loss": 0,
        }
        if rnd == rondas:
            mod_json = self.model.to_json()
            with open(dir_base + "/transfer-learning/modelsProtoScaled5/mod" + str(self.party_number) + ".json", "w") as json_file:
                json_file.write(mod_json)
            # serialize weights to HDF5
            self.model.save_weights(dir_base + "/transfer-learning/modelsProtoScaled5/mod" + str(self.party_number) + ".h5")
            print("Saved model" + str(self.party_number) + " to disk")

        scaler_cons = self.scaler
        test_y_resc = scaler_cons.inverse_transform(self.y_test)
        pred = pred.reshape(pred.shape[0], pred.shape[2])
        pred_resc = scaler_cons.inverse_transform(pred)
        mae_, mse_, rmse_, cvrmse_, mape_ = compute_metrics_fn(test_y_resc.reshape(-1), pred_resc.reshape(-1))

        # Guardar en csv externo
        aux = []
        path = self.dir_met
        col_name = ['MAE', 'MSE', 'RSME', 'CVRMSE', 'MAPE']
        lista = {
            "MAE": mae_,
            "MSE": mse_,
            "RSME": rmse_,
            "CVRMSE": cvrmse_,
            "MAPE": mape_
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        # loss, accuracy, *aux = self.model.evaluate(self.x_train, self.y_train, 32, steps=steps)

        """
        print("loss: " + str(loss))
        print("accuracy: " + str(accuracy))
        print("recall: " + str(recall))
        print("precision: " + str(precision))
        print("f1_sc: " + str(f1_sc))
        print("mcc: " + str(mcc))
        """
        num_examples_test = len(self.x_train)
        loss = 0
        # Guardar en csv externo
        aux = []
        path = self.dir_met
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient',
                    'Loss']
        lista = {
            "Accuracy": 0,
            "Recall": 0,
            "Precision": 0,
            "F1_score": 0,
            "Matthew_Correlation_coefficient": 0,
            "Loss": 0
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))

        return loss, num_examples_test, {"accuracy": 0}


def fedplus(weights, mean, theta):
    z = numpy.asarray(mean)
    weights = numpy.asarray(weights)

    fedp = theta * weights + (1 - theta) * z
    return fedp


def start_client_tl(client_n, names, cluster):
    print("Adelante party " + str(client_n))
    dir_base = os.getcwd()
    building = names[client_n]
    refs = lsremote('https://bitbucket.org/aurorax/datangi')
    lastCommit = refs['HEAD']
    """
    (x_train, y_train), (x_test, y_test), n_features, model, scaler_cons, building = read_client(lista_clientes=lista_clientes,
                                                                                       lastCommits=lastscommits,
                                                                                       client=client_n)
    """
    cons = pd.read_csv(
        'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-energy/' + building + '-tsCons.csv',
        sep=';', index_col=0)
    temp = pd.read_csv(
        'https://bitbucket.org/aurorax/datangi/raw/' + lastCommit + '/processed/study1-temperature/t' + building + '-ts.csv',
        sep=';', index_col=0)
    merge = pd.merge(cons, temp, how='outer', left_index=True, right_index=True)
    dataset = merge
    dataset = dataset.dropna()
    dataset = dataset[-2000:]
    fooc = dataset.iloc[:, 0].values
    foot = dataset.iloc[:, 1].values
    N_OUT = 24
    data = dataset.values
    factor = N_OUT
    n_out = N_OUT
    trainsize = myround(0.77 * len(data), factor)  # 77% of the data is train data
    testsize = myround(len(data) - trainsize - n_out, factor)
    train, test = data[0:trainsize], data[(trainsize + 1):(trainsize + 1 + testsize)]
    scaler_cons = StandardScaler()
    scaler_temp = StandardScaler()
    consScaledtr = scaler_cons.fit_transform(train[:, 0].reshape(-1, 1))
    tempScaledtr = scaler_temp.fit_transform(train[:, 1].reshape(-1, 1))
    consScaledte = scaler_cons.transform(test[:, 0].reshape(-1, 1))
    tempScaledte = scaler_temp.transform(test[:, 1].reshape(-1, 1))
    train = np.column_stack((consScaledtr, tempScaledtr))
    test = np.column_stack((consScaledte, tempScaledte))
    train = array(split(train, len(train) / factor))
    test = array(split(test, len(test) / factor))
    n_input = N_OUT * 7
    train_x, train_y = to_supervised(train, n_input, n_out=N_OUT)
    X = train_x
    y = train_y
    n_features = X.shape[2]
    print(n_features)
    test_x, test_y = to_supervised(test, n_input, n_out=N_OUT)
    n_seq = 7
    n_epochs = 300

    json_file = open(
        "/home/enrique/flower/TL_IAS/transfer-learning/modelsProtoScaled5/mod" + str(cluster + 1) + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(
        "/home/enrique/flower/TL_IAS/transfer-learning/modelsProtoScaled5/mod" + str(cluster + 1) + ".h5")

    model = create_model(N_OUT, n_seq, 24, n_features)
    model.set_weights(loaded_model.get_weights())

    #mod, pred = FitForecast(X, y, test_x, 24, n_seq, 24, n_features, n_epochs, model)
    # Start Flower client
    #metricas = dir_base + "/metricas/Metricas_cliente_" + str(client_n) + "_" + str(building) + ".csv"

    metricas = dir_base + "/metricas/TL/cluster "+str(cluster)+"/Metricas_cliente_" + str(client_n) +".csv"

    vacio = []
    path = metricas
    col_name = ['MAE', 'MSE', 'RSME', 'CVRMSE', 'MAPE']
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)

    """
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    """
    client = tfmlpClient(model, train_x, train_y, test_x, test_y, 7, n_features, scaler_cons, metricas, client_n)
    print(("party " + str(client_n) + " lista"))

    # IP lorien 155.54.95.95
    fl.client.start_numpy_client("[::]:8080", client=client)



def start_client(client_n, lista_clientes, lastscommits):
    print("Adelante party " + str(client_n))
    dir_base = os.getcwd()
    (x_train, y_train), (x_test, y_test), n_features, model, scaler_cons, building = read_client(lista_clientes=lista_clientes,
                                                                                       lastCommits=lastscommits,
                                                                                       client=client_n)
    # Start Flower client
    #metricas = dir_base + "/metricas/Metricas_cliente_" + str(client_n) + "_" + str(building) + ".csv"
    metricas = dir_base + "/metricas/Metricas_cliente_" + str(client_n) +".csv"
    vacio = []
    path = metricas
    col_name = ['MAE', 'MSE', 'RSME', 'CVRMSE', 'MAPE']
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)
    """
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    """
    client = tfmlpClient(model, x_train, y_train, x_test, y_test, 7, n_features, scaler_cons, metricas, client_n)
    print(("party " + str(client_n) + " lista"))

    # IP lorien 155.54.95.95
    fl.client.start_numpy_client("[::]:8080", client=client)


def start_server(parties, lista_parties, lastscommits, rounds, alp,lea):
    print(("Adelante Server"))

    dir_base = os.getcwd()
    _, _, _, model, scaler, aux = read_client(lista_parties, lastCommits=lastscommits, client=2)
    set_alpha(alp)
    set_lr(lea)

    # Create strategy

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=parties,
        min_evaluate_clients=parties,
        min_available_clients=parties,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights(),
    )
    # IP lorien 155.54.95.95
    # Start Flower server for four rounds of federated learning
    fl.server.start_server(config={"num_rounds": rounds}, strategy=strategy)

def start_server_tl(parties, rounds, alp, lea):
    print(("Adelante Server"))

    set_alpha(alp)
    set_lr(lea)
    N_OUT = 24
    n_seq = 7
    model = create_model(N_OUT, n_seq, 24, 2)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=parties,
        min_evaluate_clients=parties,
        min_available_clients=parties,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights(),
    )
    # IP lorien 155.54.95.95
    # Start Flower server for four rounds of federated learning
    fl.server.start_server(config={"num_rounds": rounds}, strategy=strategy)



from typing import Callable, Dict, Optional, Tuple

def get_eval_fn(model) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, Dict[str, fl.common.Scalar]]]]:
    """Return an evaluation function for server-side evaluation."""

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Convert parameters (list of ndarrays) to model weights
        model.set_weights(parameters)

        # Aquí deberías incluir tu evaluación real, por ejemplo:
        # loss, accuracy = model.evaluate(x_val, y_val)
        loss = 0.0
        accuracy = 0.0

        return loss, {"accuracy": accuracy}

    return evaluate



def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "local_epochs": 1,  # if rnd < 2 else 2,
        "round": rnd,
        "val_steps": 5,
        "rounds": 4
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}
