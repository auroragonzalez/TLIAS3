import pandas as pd
import numpy as np
import argparse
import os

mae_array = []
mse_array = []
rsme_array = []
cvrmse_array = []
mape_array = []

for i in range(15):
    csv_fed = pd.read_csv(os.getcwd()+"/metricas/Metricas_cliente_"+str(i+1)+".csv")
    col_name = ['MAE', 'MSE', 'RSME', 'CVRMSE', 'MAPE']
    MAE = np.asarray(csv_fed['MAE'])
    MSE = np.asarray(csv_fed['MSE'])
    RSME = np.asarray(csv_fed['RSME'])
    CVRMSE = np.asarray(csv_fed['CVRMSE'])
    MAPE = np.asarray(csv_fed['MAPE'])

    mae = MAE[-1]
    mse = MSE[-1]
    rsme = RSME[-1]
    cvrmse = CVRMSE[-1]
    mape = MAPE[-1]

    mae_array.append(mae)
    mse_array.append(mse)
    rsme_array.append(rsme)
    cvrmse_array.append(cvrmse)
    mape_array.append(mape)

    #print("Centroid_"+ str(i+1)+" = " + "["+str(mae)+","+str(mse)+","+str(rsme)+","+str(cvrmse)+","+str(mape)+"]")

print("mae = " + str(mae_array))
print("mse = " + str(mse_array))
print("rsme = " + str(rsme_array))
print("cvrmse = " + str(cvrmse_array))
print("mape = " + str(mape_array))



















