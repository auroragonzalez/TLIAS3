import numpy as np
import pandas as pd
import os
import argparse

n_parties = 45
dir_base = os.getcwd()
rondas = 2

cvrmse_total = []
rango = range(1, n_parties+1)
for i in rango:
    party = pd.read_csv(dir_base + "/metricas/TL/cluster 1/Metricas_cliente_" + str(i) + ".csv")
    cvrmse = np.array(party["CVRMSE"])
    cvrmse_total.append(cvrmse[-1])

print(np.mean(cvrmse_total))