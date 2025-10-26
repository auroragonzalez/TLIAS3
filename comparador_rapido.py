import pandas as pd
import numpy as np
import argparse
import os

contador = 0
restas = []
parametros = argparse.ArgumentParser()
parametros.add_argument("alpha", help="Alpha of Fed+", type=float)
parametros.add_argument("lr", help="Learning Rate", type=float)
param = parametros.parse_args()
alpha = param.alpha
lr = param.lr
for i in range(15):
    csv_dis = pd.read_csv("/home/enrique/flower/TL_IAS_Distribuido/transfer-learning/modelsProtoScaled15/mod"+str(i)+"_metrics.csv", header=None)
    csv_dis = np.array(csv_dis)
    csv_dis = csv_dis[3]
    csv_fed = pd.read_csv("/home/enrique/flower/TL_IAS/metricas/Metricas_cliente_"+str(i+1)+".csv")
    csv_fed = np.asarray(csv_fed['CVRMSE'])
    csv_fed = csv_fed[-1]
    datos = [csv_fed, str(csv_dis)]
    resta = csv_dis - csv_fed
    restas.append(resta)
    if csv_fed < csv_dis:
        contador = contador+1
        #resta = csv_dis - csv_fed
        #restas.append(resta)
    print("Centroid "+ str(i+1)+":" + str(datos))

# Guardar en csv externo
aux = []
path = "/home/enrique/flower/TL_IAS/tunning_fl_theta_lr.csv"
col_name = ['Alpha', 'lr', 'Proporcion', 'Media resta']
lista = {
    "Alpha": alpha,
    "lr": lr,
    "Proporcion": str(contador) + "/15",
    "Media resta": np.mean(restas)
}
aux.append(lista)
df1 = pd.DataFrame(aux, columns=col_name)
df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))
print(str(contador) + "/15")
print(np.mean(restas))
#print(np.std(restas))