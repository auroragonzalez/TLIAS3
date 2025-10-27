# Ejemplos de Uso - TLIAS3

## Guía Rápida de Comandos

### 1. Transfer Learning (lanzador.py)

#### Uso Básico - Modo TL

```bash
# Ejecutar Transfer Learning con configuración por defecto
# - 15 clusters
# - Últimas 2000 muestras
python lanzador.py
```

#### Configuración Personalizada - Modo TL

```bash
# Procesar solo 5 clusters
python lanzador.py --mode TL --clusters 5

# Usar solo las últimas 1000 muestras
python lanzador.py --mode TL --samples -1000

# Combinar ambos
python lanzador.py --mode TL --clusters 10 --samples -1500
```

### 2. Federated Learning (lanzador.py)

#### Uso Básico - Modo FL

```bash
# Ejecutar Federated Learning con configuración por defecto
# - 15 clientes (prototipos)
# - 15 rondas
# - alpha = 2.9
# - lr = 0.001
python lanzador.py --mode FL
```

#### Configuración Personalizada - Modo FL

```bash
# FL con 20 rondas
python lanzador.py --mode FL --rounds 20

# FL con parámetros Fed+ personalizados
python lanzador.py --mode FL --alpha 3.5 --lr 0.0001

# FL con menos clusters y más rondas
python lanzador.py --mode FL --clusters 10 --rounds 25

# Configuración completa personalizada
python lanzador.py --mode FL --clusters 12 --rounds 30 --alpha 2.5 --lr 0.0005
```

### 3. Otros Scripts de Federated Learning

#### lanzador_parse.py

```bash
# FL con 15 prototipos usando argumentos posicionales
python lanzador_parse.py <alpha> <lr>

# Ejemplos:
python lanzador_parse.py 2.9 0.001
python lanzador_parse.py 3.0 0.0001
```

#### fl_mismocluster.py

```bash
# FL solo con edificios del mismo cluster
# Nota: Editar el archivo para cambiar el cluster

# 1. Abrir fl_mismocluster.py
# 2. Cambiar: cluster = 1  # al cluster deseado
# 3. Ejecutar:
python fl_mismocluster.py <alpha> <lr>

# Ejemplo:
python fl_mismocluster.py 2.9 0.001
```

#### lanzador_allClients.py

```bash
# FL con TODOS los edificios de un cluster específico
python lanzador_allClients.py <cluster_id> <alpha> <lr>

# Ejemplos:
python lanzador_allClients.py 0 2.9 0.001    # Cluster 0
python lanzador_allClients.py 5 3.0 0.0001   # Cluster 5
python lanzador_allClients.py 14 2.5 0.001   # Cluster 14
```

### 4. Ver Ayuda

```bash
# Ver todas las opciones disponibles de lanzador.py
python lanzador.py --help

# Salida esperada:
# usage: lanzador.py [-h] [--mode {TL,FL}] [--clusters N] [--rounds N]
#                    [--samples N] [--alpha X] [--lr X]
#
# Lanzador de Transfer Learning y Federated Learning
#
# optional arguments:
#   -h, --help       show this help message and exit
#   --mode {TL,FL}   Modo de ejecución: TL (Transfer Learning) o FL (Federated Learning)
#   --clusters N     Número de clusters a procesar (por defecto: 15)
#   --rounds N       Número de rondas para FL (por defecto: 15)
#   --samples N      Número de muestras a usar (negativo = últimas N muestras, por defecto: -2000)
#   --alpha X        Parámetro alpha para Fed+ (por defecto: 2.9)
#   --lr X           Learning rate (por defecto: 0.001)
```

## Comparación de Scripts FL

| Script                     | Propósito              | Clientes                          | Configuración                               |
| -------------------------- | ---------------------- | --------------------------------- | ------------------------------------------- |
| **lanzador.py --mode FL**  | FL con prototipos      | 15 edificios representativos      | `--clusters`, `--rounds`, `--alpha`, `--lr` |
| **lanzador_parse.py**      | FL con prototipos      | 15 edificios representativos      | Argumentos posicionales `alpha lr`          |
| **fl_mismocluster.py**     | FL dentro de 1 cluster | Todos los edificios de UN cluster | Editar `cluster` en el código               |
| **lanzador_allClients.py** | FL por cluster         | Todos los edificios de UN cluster | `cluster_id alpha lr`                       |

## Flujo de Trabajo Recomendado

### Experimento Completo (TL + FL)

```bash
# 1. Crear modelos prototipo (una sola vez)
python 00.createModelsScaled-i.py

# 2. Evaluar Transfer Learning
python lanzador.py --mode TL

# 3. Evaluar Federated Learning
python lanzador.py --mode FL

# 4. Comparar resultados
python barplots.py
python mostrar_metricas.py
```

### Grid Search de Hiperparámetros FL

```bash
# Probar diferentes combinaciones de alpha y lr
# Usando lanzador.py

for alpha in 1.0 2.0 2.9 3.5; do
  for lr in 0.0001 0.001 0.01; do
    echo "Testing alpha=$alpha, lr=$lr"
    python lanzador.py --mode FL --alpha $alpha --lr $lr --rounds 10
    # Renombrar archivos de salida para no sobreescribir
    mv metricas/Metricas_cliente_1.csv metricas/Metricas_a${alpha}_lr${lr}.csv
  done
done
```

### Evaluar FL en Cada Cluster

```bash
# Procesar cada cluster individualmente
for cluster in {0..14}; do
  echo "Processing cluster $cluster"
  python lanzador_allClients.py $cluster 2.9 0.001
  # Mover resultados a carpeta específica
  mkdir -p resultados_cluster_$cluster
  mv metricas/Metricas_cliente_*.csv resultados_cluster_$cluster/
done
```

## Verificación de Resultados

### Transfer Learning

```bash
# Ver archivos de resultados
ls -lh resWO-15-i.csv resW-15-i.csv

# Ver métricas por cluster
ls -lh metricas_TL/cluster_*/

# Comparar CVRMSE promedio
python -c "
import pandas as pd
resWO = pd.read_csv('resWO-15-i.csv', sep=';')
resW = pd.read_csv('resW-15-i.csv', sep=';')
print(f'CVRMSE Sin TL: {resWO.iloc[:,4].mean():.2f}%')
print(f'CVRMSE Con TL: {resW.iloc[:,4].mean():.2f}%')
print(f'Mejora: {(resWO.iloc[:,4].mean() - resW.iloc[:,4].mean()):.2f}%')
"
```

### Federated Learning

```bash
# Ver métricas de cada cliente
ls -lh metricas/Metricas_cliente_*.csv

# Ver última métrica de cada cliente
for i in {1..15}; do
  echo "Cliente $i:"
  tail -1 metricas/Metricas_cliente_$i.csv
done

# Calcular CVRMSE promedio de última ronda
python -c "
import pandas as pd
import glob

cvrmse_values = []
for file in glob.glob('metricas/Metricas_cliente_*.csv'):
    df = pd.read_csv(file)
    if len(df) > 0:
        cvrmse_values.append(df.iloc[-1]['CVRMSE'])

print(f'CVRMSE Promedio (última ronda): {sum(cvrmse_values)/len(cvrmse_values):.2f}%')
print(f'CVRMSE Min: {min(cvrmse_values):.2f}%')
print(f'CVRMSE Max: {max(cvrmse_values):.2f}%')
"
```

## Solución de Problemas Comunes

### Error: "Connection refused" en FL

```bash
# Aumentar tiempo de espera en lanzador.py
# Editar y cambiar sleep de 30 a 60 segundos
# O ejecutar servidor y clientes manualmente
```

### Ejecutar servidor y clientes manualmente (debugging)

```bash
# Terminal 1: Servidor
python -c "
from FuncionesAux import *
import multiprocessing

k = 15
lista_clientes, lastCommits = create_clients(k)
start_server(k, lista_clientes, lastCommits, rounds=5, alp=2.9, lea=0.001)
"

# Terminal 2, 3, ..., 16: Clientes (abrir 15 terminales)
python -c "
from FuncionesAux import *
k = 15
lista_clientes, lastCommits = create_clients(k)
start_client(1, lista_clientes, lastCommits)  # Cambiar 1 por 2, 3, ..., 15
"
```

### Reducir uso de memoria

```bash
# Usar menos muestras
python lanzador.py --mode TL --samples -500

# Procesar menos clusters
python lanzador.py --mode TL --clusters 5

# FL con menos rondas
python lanzador.py --mode FL --rounds 3
```

## Notas Importantes

1. **Tiempo de ejecución:**

   - TL completo (15 clusters, ~500 edificios): 4-8 horas
   - FL (15 clientes, 15 rondas): 1-2 horas
   - Depende de hardware (GPU recomendada)

2. **Requisitos de espacio:**

   - Modelos prototipo: ~50 MB
   - Resultados TL: ~100 MB
   - Resultados FL: ~10 MB

3. **Conexión a Internet:**

   - Necesaria para descargar datos de edificios
   - Repositorio: https://bitbucket.org/aurorax/datangi

4. **Reproducibilidad:**
   - Los resultados pueden variar ligeramente por:
     - Inicialización aleatoria de pesos
     - Orden de procesamiento en multiprocessing
     - Versión de TensorFlow/Keras

---

**Actualizado:** Octubre 2025
