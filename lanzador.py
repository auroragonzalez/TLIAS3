from FuncionesAux import *
import warnings
import os
import collections
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

print("Number of cpu : ", multiprocessing.cpu_count())

# Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Lanzador de Transfer Learning y Federated Learning')
parser.add_argument('--mode', type=str, choices=['TL', 'FL'], default='TL',
                    help='Modo de ejecución: TL (Transfer Learning) o FL (Federated Learning)')
parser.add_argument('--clusters', type=int, default=15,
                    help='Número de clusters a procesar (por defecto: 15)')
parser.add_argument('--rounds', type=int, default=2,
                    help='Número de rondas para FL (por defecto: 15)')
parser.add_argument('--samples', type=int, default=-2000,
                    help='Número de muestras a usar (negativo = últimas N muestras, por defecto: -2000)')
parser.add_argument('--alpha', type=float, default=2.9,
                    help='Parámetro alpha para Fed+ (por defecto: 2.9)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate (por defecto: 0.001)')

args = parser.parse_args()

# Configuración basada en argumentos
mode = args.mode
k = args.clusters
rondas = args.rounds
n_muestras = args.samples
alpha = args.alpha
lr = args.lr

print(f"\n{'='*60}")
print(f"MODO DE EJECUCIÓN: {mode}")
print(f"{'='*60}")
print(f"Clusters: {k}")
print(f"Rondas: {rondas}")
print(f"Muestras: {n_muestras}")
if mode == 'FL':
    print(f"Alpha (Fed+): {alpha}")
    print(f"Learning Rate: {lr}")
print(f"{'='*60}\n")

# limpiar_csv(5)
acc_media_gb = []
clients = []
dir_base = os.getcwd()

lista = ['PrimClass_Julianna','UnivClass_Anya','Office_Gustavo',
         'PrimClass_Jaylin','UnivDorm_Marquis','PrimClass_Ervin',
         'PrimClass_Jacquelyn','Office_Maximus','PrimClass_Johnathan',
         'UnivLab_Marie','Office_Erik','PrimClass_Johnathon',
         'UnivDorm_Alka','Office_Benthe','Office_Jude']

['UnivDorm_Patty', 'PrimClass_Jacquelyn', 'Office_Lane', 'Office_Leland', 'Office_Angelo', 'Office_Benthe', 'UnivClass_Caitlyn', 'UnivLab_Alisa', 'PrimClass_Johnathon', 'PrimClass_Julianna', 'UnivLab_Cindy', 'PrimClass_Jaylin', 'UnivClass_Anya', 'UnivClass_Adrienne', 'Office_Cecelia']

#for i in range(400,1000):

    #k = make_Kpods(i)
    #lista_clientes, lastCommits = create_clients(k)
#k = make_Kpods()
"""
try:
    k = make_Kpods()
except ValueError:
    print("Value error")
else:
    aux_27 = 0
"""
#lista_clientes, lastCommits = create_clients(k)
print("Creando clientes (edificios representativos)...")
try:
    lista_clientes, lastCommits = create_clients(k)
except KeyError:
    print("Key error")
else:
    aux_27 = 0


if collections.Counter(lista_clientes) == collections.Counter(lista):
    print('✓ Lista de clientes verificada correctamente')


n_clientes = k

# ==============================================================================
# MODO FEDERATED LEARNING
# ==============================================================================
if mode == 'FL':
    print(f"\n{'='*60}")
    print(f"INICIANDO FEDERATED LEARNING")
    print(f"{'='*60}")
    print(f"Número de clientes: {n_clientes}")
    print(f"Rondas de FL: {rondas}")
    print(f"Alpha (Fed+): {alpha}")
    print(f"Learning Rate: {lr}")
    print(f"{'='*60}\n")
    
    # Iniciar servidor FL

    if __name__ == '__main__':
        inicio = time.time()
        print("Iniciando servidor FL...")
        server = multiprocessing.Process(target=start_server, args=(n_clientes, lista_clientes, lastCommits, rondas, alpha, lr))
        server.start()
        time.sleep(30)  # Esperar a que el servidor esté listo
        
        # Iniciar clientes FL
        print(f"Iniciando {n_clientes} clientes FL...")
        for i in range(n_clientes):
            inx = i + 1
            p = multiprocessing.Process(target=start_client, args=(inx, lista_clientes, lastCommits))
            p.start()
            clients.append(p)
            print(f"  Cliente {inx}/{n_clientes} iniciado")
        
        # Esperar a que terminen todos los procesos
        print("\nEsperando finalización del servidor...")
        server.join()
        print("Esperando finalización de clientes...")
        for client in clients:
            client.join()
        
        fin = time.time()
        tiempo = fin - inicio
        
        print(f"\n{'='*60}")
        print(f"FEDERATED LEARNING COMPLETADO")
        print(f"Tiempo total: {tiempo:.2f} segundos ({tiempo/60:.2f} minutos)")
        print(f"Resultados guardados en: metricas/Metricas_cliente_*.csv")
        print(f"{'='*60}\n")
        
        del fin, inicio

# ==============================================================================
# MODO TRANSFER LEARNING
# ==============================================================================
elif mode == 'TL':
    print(f"\n{'='*60}")
    print(f"INICIANDO TRANSFER LEARNING")
    print(f"{'='*60}")
    print(f"Clusters a procesar: {k}")
    print(f"Muestras por edificio: {n_muestras}")
    print(f"{'='*60}\n")
    
    labelsAndDist = pd.read_csv(dir_base + '/01.k-prot-clustering/kprototypes-labels-dist-iii15.csv', sep=";")
    resWO = np.zeros((labelsAndDist.shape[0], 7))
    resW = np.zeros((labelsAndDist.shape[0], 7))
    count = 0
    
    inicio = time.time()
    
    for i in range(0, k):
        print(f"\nProcesando cluster {i+1}/{k}...")
        transfer_learning(i, lastCommits, resWO, resW, count, n_muestras)
        count = count + 1
    
    fin = time.time()
    tiempo = fin - inicio
    
    # Guardar resultados
    print("\nGuardando resultados...")
    res_resWO = pd.DataFrame(resWO)
    res_resWO.to_csv("resWO-15-i.csv", sep=';')
    print("  ✓ Resultados SIN Transfer Learning: resWO-15-i.csv")
    
    res_resW = pd.DataFrame(resW)
    res_resW.to_csv("resW-15-i.csv", sep=';')
    print("  ✓ Resultados CON Transfer Learning: resW-15-i.csv")
    
    print(f"\n{'='*60}")
    print(f"TRANSFER LEARNING COMPLETADO")
    print(f"Tiempo total: {tiempo:.2f} segundos ({tiempo/60:.2f} minutos)")
    print(f"Resultados guardados en:")
    print(f"  - resWO-15-i.csv (Sin TL)")
    print(f"  - resW-15-i.csv (Con TL)")
    print(f"  - metricas_TL/cluster_*/edificio_*.csv")
    print(f"{'='*60}\n")
