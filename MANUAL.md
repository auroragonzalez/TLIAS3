# Manual del Proyecto TLIAS3

## Transfer Learning and Federated Learning for Energy Prediction in Smart Buildings

---

## Índice

1. [Descripción General](#1-descripción-general)
2. [Arquitectura del Proyecto](#2-arquitectura-del-proyecto)
3. [Requisitos y Configuración](#3-requisitos-y-configuración)
4. [Estructura de Directorios](#4-estructura-de-directorios)
5. [Flujo de Trabajo Completo](#5-flujo-de-trabajo-completo)
6. [Módulos Principales](#6-módulos-principales)
7. [Scripts de Ejecución](#7-scripts-de-ejecución)
8. [Guía Paso a Paso](#8-guía-paso-a-paso)
9. [Métricas y Resultados](#9-métricas-y-resultados)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Descripción General

### 1.1 Objetivo del Proyecto

Este proyecto implementa un **framework de Transfer Learning (TL) y Federated Learning (FL)** para mejorar la eficiencia energética en edificios inteligentes. El sistema:

- **Agrupa edificios similares** usando clustering K-Prototypes basado en características estructurales y contextuales
- **Entrena modelos base** (prototipos) para cada cluster usando redes ConvLSTM2D
- **Aplica Transfer Learning** para adaptar estos modelos a edificios individuales
- **Utiliza Federated Learning** (con algoritmo Fed+) para entrenamiento colaborativo preservando la privacidad
- **Predice consumo energético** a 24 horas vista usando datos de consumo histórico y temperatura

### 1.2 Tecnologías Utilizadas

- **Python 3.10**
- **TensorFlow/Keras**: Modelos de Deep Learning (ConvLSTM2D)
- **Flower Framework**: Implementación de Federated Learning
- **K-Prototypes**: Clustering con variables mixtas (numéricas y categóricas)
- **scikit-learn**: Preprocesamiento y métricas
- **pandas/numpy**: Manipulación de datos
- **multiprocessing**: Ejecución paralela de clientes FL

---

## 2. Arquitectura del Proyecto

### 2.1 Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA TLIAS3                       │
└─────────────────────────────────────────────────────────────┘

1. CLUSTERING (K-Prototypes)
   ├── Datos: Características de edificios (11 features)
   ├── Algoritmo: K-Prototypes con 15 clusters
   └── Output: Labels y centroides

2. CREACIÓN DE MODELOS PROTOTIPO
   ├── Para cada cluster: entrenar modelo ConvLSTM2D
   ├── Input: Consumo + Temperatura (7 días)
   ├── Output: Predicción 24h
   └── Guardar: modelsProtoScaled5/mod{i}.json + .weights.h5

3. FEDERATED LEARNING
   ├── Servidor: Coordina entrenamiento (FedAvg + Fed+)
   ├── Clientes: Entrenan localmente
   ├── Agregación: Promedio ponderado de pesos
   └── Iteraciones: Múltiples rondas (rounds)

4. TRANSFER LEARNING
   ├── Cargar modelo del cluster correspondiente
   ├── Fine-tuning con datos del edificio específico
   └── Comparar: Con TL vs Sin TL
```

### 2.2 Modelo de Red Neuronal

**Arquitectura ConvLSTM2D:**

```
Input: (n_seq=7, 1, n_substeps=24, n_features=2)
  ↓
ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu')
  ↓
Flatten()
  ↓
Dense(24)  # Predicción 24 horas
  ↓
Output: 24 valores de consumo predichos
```

**Características:**

- **n_seq**: 7 secuencias (7 días de entrada)
- **n_substeps**: 24 pasos temporales por día
- **n_features**: 2 (consumo energético + temperatura)
- **Output**: 24 valores (predicción del día siguiente)

---

## 3. Requisitos y Configuración

### 3.1 Requisitos del Sistema

- **Python**: 3.10
- **RAM**: Mínimo 8GB (recomendado 16GB para FL)
- **GPU**: Opcional pero recomendada (CUDA compatible)
- **CPU**: Múltiples núcleos para paralelización

### 3.2 Instalación

#### Paso 1: Instalar Python 3.10

**En Linux/Ubuntu:**

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

**En Windows:**

- Descargar desde python.org
- Asegurarse de marcar "Add to PATH"

#### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno
python3.10 -m venv excellCity-FLTL

# Activar entorno
# Linux/Mac:
source excellCity-FLTL/bin/activate

# Windows:
excellCity-FLTL\Scripts\activate
```

#### Paso 3: Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Dependencias Principales

```
tensorflow==2.17.0
keras==3.11.3
flwr==0.19.0              # Federated Learning
kmodes==0.12.2            # K-Prototypes clustering
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
matplotlib==3.9.2
GitPython==3.1.43
```

---

## 4. Estructura de Directorios

```
TLIAS3/
│
├── FuncionesAux.py              # Biblioteca principal de funciones
├── FuncionesAuxi.py             # Funciones auxiliares (versión alternativa)
│
├── 00.data/                     # Datos originales
│   └── I-BLEND/
│       └── meta_open.csv
│
├── 01.k-prot-clustering/        # Clustering K-Prototypes
│   ├── clustering-iii15.py      # Script de clustering
│   ├── Kprot.py                 # Algoritmo K-Prototypes
│   ├── kprototypes-labels-dist-iii15.csv  # Labels y distancias
│   ├── kprototypes-centroids-iii15.csv    # Centroides
│   └── kprototypes-X_hat-iii15.csv        # Datos imputados
│
├── transfer-learning/           # Modelos de Transfer Learning
│   └── modelsProtoScaled5/
│       ├── mod0.json            # Arquitectura modelo cluster 0
│       ├── mod0.weights.h5      # Pesos modelo cluster 0
│       ├── mod0_metrics.csv     # Métricas modelo cluster 0
│       └── ... (mod1 a mod14)
│
├── metricas/                    # Resultados de experimentos FL
│   ├── Metricas_cliente_1.csv
│   ├── Metricas_cliente_2.csv
│   └── ...
│
├── metricas_TL/                 # Resultados Transfer Learning
│   ├── cluster_0/
│   │   ├── edificio_1.csv
│   │   └── ...
│   └── cluster_1/...
│
├── 02.tl/                       # Resultados comparativos TL
│   ├── resW-15-iv.csv           # Resultados CON Transfer Learning
│   └── resWO-15-iv.csv          # Resultados SIN Transfer Learning
│
├── imagenes/                    # Visualizaciones
│   ├── pdf/
│   └── ...
│
├── Scripts de Ejecución:
├── 00.createModelsScaled-i.py   # Crear modelos prototipo
├── lanzador.py                  # Launcher principal
├── lanzador_parse.py            # Launcher con argumentos
├── lanzador_allClients.py       # FL para todos los clientes de un cluster
├── fl_mismocluster.py           # FL dentro del mismo cluster
├── barplots.py                  # Visualización de resultados
├── mostrar_metricas.py          # Mostrar métricas
│
└── requirements.txt             # Dependencias del proyecto
```

---

## 5. Flujo de Trabajo Completo

### 5.1 Diagrama de Flujo

```
┌────────────────────────────────────────────────────────────┐
│                  PASO 1: CLUSTERING                         │
│  Agrupar edificios por características estructurales        │
│  Script: 01.k-prot-clustering/clustering-iii15.py          │
│  Output: Labels, centroides, datos imputados               │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│         PASO 2: CREAR MODELOS PROTOTIPO (1 por cluster)    │
│  Entrenar modelo ConvLSTM2D para edificio representativo   │
│  Script: 00.createModelsScaled-i.py                        │
│  Output: transfer-learning/modelsProtoScaled5/mod{i}.*     │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│              PASO 3A: FEDERATED LEARNING                    │
│  Entrenamiento colaborativo distribuido (Fed+)             │
│  Scripts: lanzador_parse.py, fl_mismocluster.py,          │
│           lanzador_allClients.py                           │
│  Output: metricas/Metricas_cliente_{i}.csv                 │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│              PASO 3B: TRANSFER LEARNING                     │
│  Adaptar modelo prototipo a edificios individuales         │
│  Script: lanzador.py (función transfer_learning)           │
│  Output: metricas_TL/cluster_{i}/edificio_{j}.csv          │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│           PASO 4: ANÁLISIS Y VISUALIZACIÓN                  │
│  Comparar resultados, generar gráficos                      │
│  Scripts: barplots.py, mostrar_metricas.py                 │
│  Output: imagenes/pdf/                                      │
└────────────────────────────────────────────────────────────┘
```

---

## 6. Módulos Principales

### 6.1 FuncionesAux.py

Biblioteca central con todas las funciones del proyecto.

#### 6.1.1 Funciones de Preprocesamiento

```python
myround(x, base=5)
```

Redondea un número al múltiplo más cercano de `base`.

```python
split_dataset(data, factor=7, n_out=7)
```

Divide datos en train/test (77%/23%) en ventanas temporales.

```python
to_supervised(train, n_input, n_out=7)
```

Convierte series temporales a formato supervisado (X, y).

#### 6.1.2 Funciones de Modelado

```python
create_model(N_OUT, n_seq, n_substeps, n_features)
```

Crea modelo ConvLSTM2D con parámetros configurables.

- **N_OUT**: Horizonte de predicción (24h)
- **n_seq**: Secuencias de entrada (7 días)
- **n_substeps**: Pasos por secuencia (24h)
- **n_features**: Variables de entrada (2: consumo + temp)

```python
FitForecast_e(model, X_train, y_train, X_test, N_OUT,
              n_seq, n_substeps, n_features, epochs, number, trained_model)
```

Entrena modelo y genera predicciones.

- **trained_model**: Si no es None, usa Transfer Learning

#### 6.1.3 Clustering K-Prototypes

```python
kprot_missing(X, n_clusters, cat_index, centroids, max_iter=10)
```

K-Prototypes con manejo de valores faltantes.

- **cat_index**: Índices de variables categóricas [1,2,3,5,7,8,10]
- Imputa valores faltantes con media (numéricos) y moda (categóricos)

```python
make_Kpods()
```

Pipeline completo de clustering:

1. Cargar datos de edificios
2. Convertir variables categóricas a numéricas
3. Ejecutar K-Prototypes (15 clusters)
4. Guardar resultados

#### 6.1.4 Transfer Learning

```python
transfer_learning(cluster, lastCommits, resWO, resW, count, n_muestras)
```

Evalúa Transfer Learning para un cluster:

1. Carga modelo prototipo del cluster
2. Para cada edificio del cluster:
   - Entrena SIN modelo cargado (baseline)
   - Entrena CON modelo cargado (TL)
   - Compara métricas
3. Guarda resultados en `metricas_TL/cluster_{i}/`

#### 6.1.5 Federated Learning

**Clase Principal:**

```python
class tfmlpClient(fl.client.NumPyClient)
```

Cliente de Flower Framework con:

- `fit()`: Entrenamiento local con Fed+
- `evaluate()`: Evaluación del modelo global
- Guarda modelos finales en última ronda

**Algoritmo Fed+:**

```python
fedplus(weights, mean, theta)
```

```
new_weights = mean + theta * (local_weights - mean)
theta = 1 / (1 + alpha * lr)
```

**Funciones de Servidor:**

```python
start_server(parties, lista_parties, lastscommits, rounds, alp, lea)
start_server_tl(parties, rounds, alp, lea)
```

Inicia servidor FL con estrategia FedAvg.

**Funciones de Cliente:**

```python
start_client(client_n, lista_clientes, lastscommits)
start_client_tl(client_n, names, cluster)
```

Inicia cliente FL que se conecta al servidor.

#### 6.1.6 Métricas

```python
compute_metrics_fn(y_valid_resc, y_hat_resc)
```

Calcula 5 métricas:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **CVRMSE**: Coefficient of Variation of RMSE (%)
- **MAPE**: Mean Absolute Percentage Error (%)

---

## 7. Scripts de Ejecución

### 7.1 Clustering: clustering-iii15.py

**Propósito:** Agrupar edificios en 15 clusters usando K-Prototypes.

**Características usadas:**

1. `energystarscore` (num)
2. `heatingtype` (cat)
3. `industry` (cat)
4. `numberoffloors` (cat)
5. `occupants` (num)
6. `rating` (cat)
7. `sqm` (num)
8. `subindustry` (cat)
9. `timezone` (cat)
10. `yearbuilt` (num)
11. `primaryspaceuse_abbrev` (cat)

**Ejecución:**

```bash
cd 01.k-prot-clustering
python clustering-iii15.py
```

**Outputs:**

- `kprototypes-labels-dist-iii15.csv`: Labels y distancias
- `kprototypes-centroids-iii15.csv`: Centroides de cada cluster
- `kprototypes-df-iii15.csv`: Datos originales
- `kprototypes-X_hat-iii15.csv`: Datos con valores imputados

---

### 7.2 Creación de Modelos: 00.createModelsScaled-i.py

**Propósito:** Entrenar 15 modelos prototipo (uno por cluster).

**Proceso:**

1. Para cada cluster (0-14):
   - Identifica edificio más cercano al centroide
   - Descarga datos de consumo y temperatura
   - Preprocesa: normalización, ventanas temporales
   - Entrena modelo ConvLSTM2D (5 épocas)
   - Guarda modelo (.json + .weights.h5) y métricas

**Ejecución:**

```bash
python 00.createModelsScaled-i.py
```

**Outputs:**

```
transfer-learning/modelsProtoScaled5/
├── mod0.json, mod0.weights.h5, mod0_metrics.csv
├── mod1.json, mod1.weights.h5, mod1_metrics.csv
└── ... (hasta mod14)
```

---

### 7.3 Transfer Learning y Federated Learning: lanzador.py

**Propósito:** Evaluar Transfer Learning en todos los edificios O ejecutar Federated Learning con edificios representativos.

**Modos de Ejecución:**

El script ahora soporta dos modos de ejecución mediante argumentos de línea de comandos:

1. **Modo TL (Transfer Learning)**: Evalúa TL en todos los edificios
2. **Modo FL (Federated Learning)**: Ejecuta FL con los edificios representativos

**Argumentos:**

```bash
python lanzador.py [--mode {TL,FL}] [--clusters N] [--rounds N] [--samples N] [--alpha X] [--lr X]
```

| Argumento    | Descripción                               | Por defecto |
| ------------ | ----------------------------------------- | ----------- |
| `--mode`     | Modo de ejecución: `TL` o `FL`            | `TL`        |
| `--clusters` | Número de clusters a procesar             | `15`        |
| `--rounds`   | Número de rondas para FL                  | `15`        |
| `--samples`  | Número de muestras (negativo = últimas N) | `-2000`     |
| `--alpha`    | Parámetro alpha para Fed+                 | `2.9`       |
| `--lr`       | Learning rate                             | `0.001`     |

**Ejemplos de Uso:**

**1. Ejecutar Transfer Learning (modo por defecto):**

```bash
# Transfer Learning con configuración por defecto (15 clusters, -2000 muestras)
python lanzador.py

# Transfer Learning con 10 clusters
python lanzador.py --mode TL --clusters 10

# Transfer Learning con solo las últimas 1000 muestras
python lanzador.py --mode TL --samples -1000
```

**2. Ejecutar Federated Learning:**

```bash
# Federated Learning con configuración por defecto
python lanzador.py --mode FL

# FL con parámetros personalizados
python lanzador.py --mode FL --rounds 20 --alpha 2.5 --lr 0.0001

# FL con menos clusters (prototipos)
python lanzador.py --mode FL --clusters 10 --rounds 10
```

**Proceso en Modo TL:**

1. Crea clientes (edificios representativos de cada cluster)
2. Lee labels del clustering
3. Para cada cluster (0 a k-1):
   - Para cada edificio del cluster:
     - Entrena modelo baseline (sin TL)
     - Entrena modelo con TL
     - Compara métricas
4. Guarda resultados comparativos

**Proceso en Modo FL:**

1. Crea clientes (edificios representativos de cada cluster)
2. Inicia servidor FL en localhost:8080
3. Espera 30s para que servidor esté listo
4. Lanza k procesos cliente en paralelo
5. Cada cliente:
   - Carga sus datos
   - Entrena localmente con Fed+
   - Envía pesos al servidor
6. Servidor agrega pesos (FedAvg)
7. Repite por `rounds` iteraciones

**Outputs:**

**Modo TL:**

- `resWO-15-i.csv`: Resultados SIN Transfer Learning
- `resW-15-i.csv`: Resultados CON Transfer Learning
- `metricas_TL/cluster_{i}/edificio_{j}.csv`: Métricas detalladas por edificio

**Modo FL:**

- `metricas/Metricas_cliente_{i}.csv`: Métricas por cliente (i=1 a k)
- `transfer-learning/modelsProtoScaled5/mod{i}.json`: Modelos entrenados (última ronda)
- `transfer-learning/modelsProtoScaled5/mod{i}.weights.h5`: Pesos de modelos

**Ejecución:**

```bash
# Modo Transfer Learning (por defecto)
python lanzador.py

# Modo Federated Learning
python lanzador.py --mode FL
```

**Outputs:**

**Modo TL:**

- `metricas_TL/cluster_{i}/edificio_{j}.csv`
- `resWO-15-i.csv` (resultados sin TL)
- `resW-15-i.csv` (resultados con TL)

**Modo FL:**

- `metricas/Metricas_cliente_{i}.csv` (i=1 a k)

**Verificación:**

```bash
# Verificar resultados TL
ls resWO-15-i.csv resW-15-i.csv
ls metricas_TL/cluster_*/edificio_*.csv

# Verificar resultados FL
ls metricas/Metricas_cliente_*.csv
```

---

### 7.4 Federated Learning con Parámetros: lanzador_parse.py

**Propósito:** Ejecutar FL con 15 clientes (uno por cluster) con parámetros ajustables.

**Argumentos:**

```bash
python lanzador_parse.py <alpha> <lr>
```

- **alpha**: Parámetro del algoritmo Fed+ (ej: 2.9)
- **lr**: Learning rate (ej: 0.001)

**Configuración:**

```python
rondas = 4  # Número de rondas FL
n_clientes = 15  # 15 clientes (prototipos)
```

**Proceso:**

1. Crea 15 clientes (edificios representativos)
2. Inicia servidor FL en localhost:8080
3. Espera 30s para que servidor esté listo
4. Lanza 15 procesos cliente en paralelo
5. Cada cliente:
   - Carga sus datos
   - Entrena localmente con Fed+
   - Envía pesos al servidor
6. Servidor agrega pesos (FedAvg)
7. Repite por `rondas` iteraciones

**Ejecución:**

```bash
python lanzador_parse.py 2.9 0.001
```

**Outputs:**

- `metricas/Metricas_cliente_{i}.csv` (i=1 a 15)

---

### 7.5 FL por Cluster: fl_mismocluster.py

**Propósito:** FL solo con edificios del mismo cluster.

**Argumentos:**

```bash
python fl_mismocluster.py <alpha> <lr>
```

**Configuración:**

```python
cluster = 1  # Cluster a procesar
rondas = 3
```

**Proceso:**

1. Identifica edificios del `cluster` especificado
2. Cuenta número de edificios → número de clientes
3. Inicia servidor FL con TL (usa modelo prototipo)
4. Lanza un proceso por cada edificio del cluster
5. Entrenamiento FL dentro del cluster

**Ejecución:**

```bash
python fl_mismocluster.py 2.9 0.001
```

---

### 7.6 FL Todos los Edificios de un Cluster: lanzador_allClients.py

**Propósito:** FL con TODOS los edificios de un cluster específico (no solo el representativo).

**Argumentos:**

```bash
python lanzador_allClients.py <cluster> <alpha> <lr>
```

- **cluster**: ID del cluster (0-14)

**Proceso:**

1. Lee labels del clustering
2. Obtiene TODOS los edificios del cluster
3. Lanza FL con todos ellos

**Ejecución:**

```bash
python lanzador_allClients.py 0 2.9 0.001
```

---

### 7.7 Visualización: barplots.py

**Propósito:** Generar gráficos de barras comparativos.

**Ejemplo de uso:**

```python
# Comparar métricas de los 15 prototipos
cvrmse = [6.48, 34.86, 8.18, ...]  # Para cada cluster
plt.bar(x_range, cvrmse)
plt.savefig('imagenes/metrics.pdf')
```

---

## 8. Guía Paso a Paso

### 8.1 Proyecto Completo desde Cero

#### PASO 1: Preparación del Entorno

```bash
# 1. Clonar/descargar proyecto
cd TLIAS3

# 2. Crear entorno virtual
python3.10 -m venv excellCity-FLTL
source excellCity-FLTL/bin/activate  # Linux/Mac
# excellCity-FLTL\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

#### PASO 2: Clustering de Edificios

```bash
cd 01.k-prot-clustering
python clustering-iii15.py
cd ..
```

**Verificación:**

```bash
# Deben existir:
ls 01.k-prot-clustering/kprototypes-labels-dist-iii15.csv
ls 01.k-prot-clustering/kprototypes-centroids-iii15.csv
```

#### PASO 3: Crear Modelos Prototipo

```bash
python 00.createModelsScaled-i.py
```

**Duración:** ~30-60 minutos (depende de hardware)

**Verificación:**

```bash
# Deben existir 15 modelos:
ls transfer-learning/modelsProtoScaled5/mod*.json
ls transfer-learning/modelsProtoScaled5/mod*.weights.h5
```

#### PASO 4A: Evaluar Transfer Learning

```bash
# Ejecutar en modo Transfer Learning (por defecto)
python lanzador.py

# O especificar explícitamente el modo TL
python lanzador.py --mode TL

# Con configuración personalizada
python lanzador.py --mode TL --clusters 10 --samples -1000
```

**Duración:** Varias horas (procesa ~500 edificios)

**Verificación:**

```bash
ls metricas_TL/cluster_*/edificio_*.csv
ls resWO-15-i.csv  # Sin TL
ls resW-15-i.csv   # Con TL
```

#### PASO 4B: Ejecutar Federated Learning

**Opción 1: FL con Prototipos (15 clientes representativos)**

```bash
# Ejecutar en modo Federated Learning
python lanzador.py --mode FL

# Con parámetros personalizados
python lanzador.py --mode FL --rounds 20 --alpha 2.5 --lr 0.0001
```

**Opción 2: FL con Parámetros Específicos**

```bash
python lanzador_parse.py 2.9 0.001
```

**Opción 3: FL dentro de un Cluster**

```bash
# Editar fl_mismocluster.py: cambiar cluster = 1 al deseado
python fl_mismocluster.py 2.9 0.001
```

**Opción 4: FL con Todos los Edificios de un Cluster**

```bash
python lanzador_allClients.py 0 2.9 0.001  # Cluster 0
```

**Verificación:**

```bash
ls metricas/Metricas_cliente_*.csv
```

#### PASO 5: Analizar Resultados

```bash
# Ver métricas
python mostrar_metricas.py

# Generar gráficos
python barplots.py
ls imagenes/pdf/metrics.pdf
```

---

### 8.2 Experimentos Típicos

#### Experimento 1: Comparar TL vs No TL

```bash
# 1. Entrenar modelos base
python 00.createModelsScaled-i.py

# 2. Evaluar TL
python lanzador.py

# 3. Analizar resWO-15-i.csv (sin TL) vs resW-15-i.csv (con TL)
python -c "
import pandas as pd
resWO = pd.read_csv('resWO-15-i.csv', sep=';')
resW = pd.read_csv('resW-15-i.csv', sep=';')
print('CVRMSE Promedio Sin TL:', resWO.iloc[:,4].mean())
print('CVRMSE Promedio Con TL:', resW.iloc[:,4].mean())
"
```

#### Experimento 2: Optimizar Hiperparámetros Fed+

```bash
# Probar diferentes combinaciones de alpha y lr
for alpha in 1.0 2.0 2.9 3.5; do
  for lr in 0.0001 0.001 0.01; do
    echo "Testing alpha=$alpha, lr=$lr"
    python lanzador_parse.py $alpha $lr
    # Guardar resultados en tunning_fl_alpha_${alpha}_lr_${lr}.csv
  done
done
```

#### Experimento 3: Evaluar FL por Cluster

```bash
# Para cada cluster, ejecutar FL solo con sus edificios
for cluster in {0..14}; do
  echo "Processing cluster $cluster"
  python lanzador_allClients.py $cluster 2.9 0.001
done
```

---

## 9. Métricas y Resultados

### 9.1 Métricas Utilizadas

| Métrica    | Fórmula                                                      | Interpretación            |
| ---------- | ------------------------------------------------------------ | ------------------------- |
| **MAE**    | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$                        | Error absoluto medio      |
| **MSE**    | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$                        | Error cuadrático medio    |
| **RMSE**   | $\sqrt{MSE}$                                                 | Raíz del error cuadrático |
| **CVRMSE** | $\frac{RMSE}{\bar{y}} \times 100$                            | RMSE normalizado (%)      |
| **MAPE**   | $\frac{1}{n}\sum \frac{\|y_i - \hat{y}_i\|}{y_i} \times 100$ | Error porcentual medio    |

**Métrica Principal:** **CVRMSE** (usado en papers de energía)

### 9.2 Estructura de Archivos de Resultados

#### Métricas FL (metricas/)

```csv
MAE,MSE,RSME,CVRMSE,MAPE
2.98,14.81,3.85,6.48,4.98
1.99,11.13,3.34,34.86,400.01
...
```

Cada fila = 1 ronda de FL

#### Métricas TL (metricas_TL/cluster_X/edificio_Y.csv)

```csv
Cargado,MAE,MSE,RSME,CVRMSE,MAPE,Muestras
Si,2.5,10.2,3.2,8.5,5.1,-2000
No,3.1,12.5,3.5,9.2,6.3,-2000
```

- **Cargado=Si:** Con Transfer Learning
- **Cargado=No:** Sin Transfer Learning

#### Resultados Comparativos TL (02.tl/)

```csv
cluster,edificio_id,MAE,MSE,RMSE,CVRMSE,MAPE
0,5,2.98,14.81,3.85,6.48,4.98
0,12,3.2,15.3,3.91,6.52,5.1
...
```

### 9.3 Resultados Esperados

**CVRMSE típico por tipo de edificio:**

- **Oficinas:** 6-10%
- **Escuelas Primarias:** 8-12%
- **Universidades (aulas):** 20-35%
- **Dormitorios:** 30-40%
- **Laboratorios:** 40-60%

**Mejora con TL:** 5-15% reducción en CVRMSE

**Mejora con FL:** 10-25% reducción vs entrenamiento aislado

---

## 10. Troubleshooting

### 10.1 Problemas Comunes

#### Error: "ModuleNotFoundError: No module named 'flwr'"

**Solución:**

```bash
pip install flwr==0.19.0
```

#### Error: "GPU memory allocation failed"

**Solución:**

```python
# Ya implementado en scripts:
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

#### Error: "Connection refused" en FL

**Causa:** Servidor no está listo cuando clientes intentan conectar.

**Solución:**

```python
# Aumentar tiempo de espera en scripts:
time.sleep(30)  # Cambiar a 60 si es necesario
```

#### Error: "FileNotFoundError: kprototypes-labels-dist-iii15.csv"

**Solución:**

```bash
# Ejecutar clustering primero:
cd 01.k-prot-clustering
python clustering-iii15.py
```

#### Advertencia: "UserWarning: X has missing values"

**Causa:** Datos de edificios incompletos (esperado).

**Solución:** Ignorar (el algoritmo kprot_missing imputa automáticamente).

### 10.2 Optimización de Rendimiento

#### Para GPU limitada:

```python
# En scripts, reducir batch size:
config = {
    "batch_size": 32,  # Cambiar a 16 o 8
    ...
}
```

#### Para acelerar clustering:

```python
# En clustering-iii15.py:
cls = KPrototypes(n_clusters=15, init='Huang',
                  n_init=1,  # Reducir de 10 a 1
                  max_iter=100)  # Reducir de 1000 a 100
```

#### Para reducir tiempo de entrenamiento:

```python
# En 00.createModelsScaled-i.py:
n_epochs = 5  # Ya está en mínimo
# Reducir datos:
dataset = dataset[-1000:]  # En lugar de todo el dataset
```

### 10.3 Debugging

#### Verificar datos de un edificio:

```python
from FuncionesAux import *
refs = lsremote('https://bitbucket.org/aurorax/datangi')
lastCommit = refs['HEAD']
building = 'Office_Gustavo'
cons = pd.read_csv(f'https://bitbucket.org/aurorax/datangi/raw/{lastCommit}/processed/study1-energy/{building}-tsCons.csv', sep=';', index_col=0)
print(cons.head())
print(cons.shape)
```

#### Ver labels de clustering:

```python
import pandas as pd
labels = pd.read_csv('01.k-prot-clustering/kprototypes-labels-dist-iii15.csv')
print(labels.groupby(labels.columns[1]).size())  # Edificios por cluster
```

#### Verificar modelo guardado:

```python
from keras.models import model_from_json
import os

json_file = open('transfer-learning/modelsProtoScaled5/mod0.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights('transfer-learning/modelsProtoScaled5/mod0.weights.h5')
print(model.summary())
```

---

## Apéndices

### A. Configuración de Hiperparámetros

**Parámetros del Modelo:**

```python
N_OUT = 24              # Horizonte predicción (horas)
n_seq = 7               # Días de entrada
n_substeps = 24         # Horas por día
n_features = 2          # Consumo + Temperatura
filters = 64            # Filtros ConvLSTM2D
kernel_size = (1, 2)    # Kernel ConvLSTM2D
activation = 'relu'     # Función activación
optimizer = 'adam'      # Optimizador
loss = 'mse'            # Función de pérdida
```

**Parámetros de Entrenamiento:**

```python
epochs = 300            # Épocas entrenamiento local
train_split = 0.77      # 77% train, 23% test
```

**Parámetros Federated Learning:**

```python
rondas = 4              # Rondas FL
alpha = 2.9             # Parámetro Fed+
lr = 0.001              # Learning rate
fraction_fit = 0.3      # Fracción clientes por ronda
min_fit_clients = 15    # Mínimo clientes
```

### B. Datasets Utilizados

**Fuente de Datos:**

- **Repositorio:** https://bitbucket.org/aurorax/datangi
- **Studies:**
  - Study 1: Edificios comerciales/educativos
  - Study 2 (I-BLEND): Edificios adicionales

**Archivos:**

- `{building}-tsCons.csv`: Series temporales de consumo
- `t{building}-ts.csv`: Series temporales de temperatura
- `meta_open.csv`: Metadatos de edificios

### C. Referencias

**Algoritmos:**

- K-Prototypes: Huang, Z. (1998)
- Federated Learning: McMahan et al. (2017) - FedAvg
- Fed+: Extensión de FedAvg con regularización

**Frameworks:**

- Flower: https://flower.dev/
- TensorFlow: https://tensorflow.org/
- kmodes: https://github.com/nicodv/kmodes

---

## Contacto y Contribuciones

**Autor:** auroragonzalez  
**Repositorio:** TLIAS3  
**Licencia:** Ver archivo LICENSE

---

**Última actualización:** Octubre 2025
