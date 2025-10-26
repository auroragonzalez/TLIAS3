from FuncionesAux import *
import warnings
import os
import collections

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
"""
import gc
gc.collect()
"""
import warnings
import os
import argparse

warnings.filterwarnings("ignore")
import multiprocessing
import time

print("Number of cpu : ", multiprocessing.cpu_count())

# limpiar_csv(5)
rondas = 4
acc_media_gb = []
# for i in n_clientes:
clients = []
dir_base = os.getcwd()
"""
dir_base = os.getcwd()
df = pd.DataFrame(columns=["clientes", "accuracy", "tiempo"])
df.to_csv(dir_base+"/grid_acc_medias_fp_tiempo.csv",index=False)
"""
parametros = argparse.ArgumentParser()
parametros.add_argument("alpha", help="Alpha of Fed+", type=float)
parametros.add_argument("lr", help="Learning Rate", type=float)
param = parametros.parse_args()
alpha = param.alpha
lr = param.lr


k = 15
try:
    lista_clientes, lastCommits = create_clients(k)
except KeyError:
    print("Key error")
else:
    aux_27 = 0

lista = ['PrimClass_Julianna','UnivClass_Anya','Office_Gustavo',
         'PrimClass_Jaylin','UnivDorm_Marquis','PrimClass_Ervin',
         'PrimClass_Jacquelyn','Office_Maximus','PrimClass_Johnathan',
         'UnivLab_Marie','Office_Erik','PrimClass_Johnathon',
         'UnivDorm_Alka','Office_Benthe','Office_Jude']
if collections.Counter(lista_clientes) == collections.Counter(lista):
    print('bien')


n_clientes = 15

inicio = time.time()
server = multiprocessing.Process(target=start_server, args=(n_clientes, lista_clientes, lastCommits, rondas, alpha,lr))
server.start()
time.sleep(30)

for i in range(n_clientes):
    inx = i + 1
    p = multiprocessing.Process(target=start_client, args=(inx, lista_clientes, lastCommits))
    p.start()
    clients.append(p)

server.join()
for client in clients:
    client.join()
fin = time.time()
tiempo = fin - inicio
del fin, inicio

labelsAndDist = pd.read_csv(dir_base + '/01.k-prot-clustering/kprototypes-labels-dist-iii15.csv', sep=";")
resWO = np.zeros((labelsAndDist.shape[0], 7))
resW = np.zeros((labelsAndDist.shape[0], 7))
count = 0
import time
time.sleep(30)


"""
for i in range(0,15):
    transfer_learning(i,lastCommits,resWO, resW, count, -2000)
    count = count+1
"""
