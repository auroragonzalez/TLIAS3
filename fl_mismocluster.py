from FuncionesAux import *
import warnings
import os
import argparse

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
"""
import gc
gc.collect()
"""

warnings.filterwarnings("ignore")
import multiprocessing
import time

parametros = argparse.ArgumentParser()
parametros.add_argument("alpha", help="Alpha of Fed+", type=float)
parametros.add_argument("lr", help="Learning Rate", type=float)
param = parametros.parse_args()
alpha = param.alpha
lr = param.lr

refs = lsremote('https://bitbucket.org/aurorax/datangi')
lastCommit = refs['HEAD']

print("Number of cpu : ", multiprocessing.cpu_count())
labelsAndDist = pd.read_csv(os.getcwd()+'/01.k-prot-clustering/kprototypes-labels-dist-iii15.csv', sep=",")
infoBuildings = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-energybuildingsEnergyInfo.csv')
names = infoBuildings['uid']

cluster = 1

indx = labelsAndDist[labelsAndDist[labelsAndDist.columns[1]] == cluster].index # index of the buildings that belong to cluster "i"
indx[indx<508]  # momentaneo, arreglar
"""
json_file = open("/home/enrique/flower/TL_IAS/transfer-learning/modelsProtoScaled5/mod"+str(cluster+1)+".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/home/enrique/flower/TL_IAS/transfer-learning/modelsProtoScaled5/mod"+str(cluster+1)+".h5")
"""
"""
N_OUT = 24
n_seq = 7
model = create_model(N_OUT, n_seq, 24, 2)
"""
#model.set_weights(weights= loaded_model.get_weights())
#model = create_model(N_OUT, n_seq, 24, 2)
numero_clientes = len(indx)
print(numero_clientes)
rondas = 3
n_clientes_grid = [numero_clientes]
n_clientes = numero_clientes
acc_media_gb = []
# for i in n_clientes:
clients = []
"""
dir_base = os.getcwd()
df = pd.DataFrame(columns=["clientes", "accuracy", "tiempo"])
df.to_csv(dir_base+"/grid_acc_medias_fp_tiempo.csv",index=False)
"""
inicio = time.time()
server = multiprocessing.Process(target=start_server_tl, args=(n_clientes, rondas, alpha, lr))
server.start()
time.sleep(30)

for i in range(n_clientes):
    inx = i + 1
    p = multiprocessing.Process(target=start_client_tl, args=(inx, names, cluster))
    p.start()
    clients.append(p)

server.join()
for client in clients:
    client.join()
"""    
fin = time.time()
tiempo = fin - inicio
del fin, inicio
"""