import pandas as pd
import numpy as np
import argparse
import os

parametros = argparse.ArgumentParser()
parametros.add_argument("clientes", help="cluster", type=int)
param = parametros.parse_args()
clientes = param.clientes


cvrmse_array = []


for i in range(clientes):
    csv_fed = pd.read_csv(os.getcwd()+"/metricas/Metricas_cliente_"+str(i+1)+".csv")
    col_name = ['MAE', 'MSE', 'RSME', 'CVRMSE', 'MAPE']

    CVRMSE = np.asarray(csv_fed['CVRMSE'])

    cvrmse = CVRMSE[-1]

    cvrmse_array.append(cvrmse)


print(np.mean(cvrmse_array))
