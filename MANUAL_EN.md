# TLIAS3 Project Manual

## Transfer Learning and Federated Learning for Energy Prediction in Smart Buildings

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Architecture](#2-project-architecture)
3. [Requirements and Setup](#3-requirements-and-setup)
4. [Directory Structure](#4-directory-structure)
5. [Complete Workflow](#5-complete-workflow)
6. [Main Modules](#6-main-modules)
7. [Execution Scripts](#7-execution-scripts)
8. [Step-by-Step Guide](#8-step-by-step-guide)
9. [Metrics and Results](#9-metrics-and-results)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

### 1.1 Project Objective

This project implements a **Transfer Learning (TL) and Federated Learning (FL) framework** to improve energy efficiency in smart buildings. The system:

- **Groups similar buildings** using K-Prototypes clustering based on structural and contextual features
- **Trains base models** (prototypes) for each cluster using ConvLSTM2D networks
- **Applies Transfer Learning** to adapt these models to individual buildings
- **Uses Federated Learning** (with Fed+ algorithm) for collaborative training while preserving privacy
- **Predicts energy consumption** 24 hours ahead using historical consumption and temperature data

### 1.2 Technologies Used

- **Python 3.10**
- **TensorFlow/Keras**: Deep Learning models (ConvLSTM2D)
- **Flower Framework**: Federated Learning implementation
- **K-Prototypes**: Clustering with mixed variables (numerical and categorical)
- **scikit-learn**: Preprocessing and metrics
- **pandas/numpy**: Data manipulation
- **multiprocessing**: Parallel execution of FL clients

---

## 2. Project Architecture

### 2.1 Main Components

```
┌─────────────────────────────────────────────────────────────┐
│                    TLIAS3 ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────┘

1. CLUSTERING (K-Prototypes)
   ├── Data: Building features (11 features)
   ├── Algorithm: K-Prototypes with 15 clusters
   └── Output: Labels and centroids

2. PROTOTYPE MODEL CREATION
   ├── For each cluster: train ConvLSTM2D model
   ├── Input: Consumption + Temperature (7 days)
   ├── Output: 24h prediction
   └── Save: modelsProtoScaled5/mod{i}.json + .weights.h5

3. TRANSFER LEARNING
   ├── Load model from corresponding cluster
   ├── Fine-tuning with specific building data
   └── Compare: With TL vs Without TL

4. FEDERATED LEARNING
   ├── Server: Coordinates training (FedAvg + Fed+)
   ├── Clients: Train locally
   ├── Aggregation: Weighted average of weights
   └── Iterations: Multiple rounds
```

### 2.2 Neural Network Model

**ConvLSTM2D Architecture:**

```
Input: (n_seq=7, 1, n_substeps=24, n_features=2)
  ↓
ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu')
  ↓
Flatten()
  ↓
Dense(24)  # 24-hour prediction
  ↓
Output: 24 predicted consumption values
```

**Features:**

- **n_seq**: 7 sequences (7 days of input)
- **n_substeps**: 24 time steps per day
- **n_features**: 2 (energy consumption + temperature)
- **Output**: 24 values (next day prediction)

---

## 3. Requirements and Setup

### 3.1 System Requirements

- **Python**: 3.10
- **RAM**: Minimum 8GB (recommended 16GB for FL)
- **GPU**: Optional but recommended (CUDA compatible)
- **CPU**: Multiple cores for parallelization

### 3.2 Installation

#### Step 1: Install Python 3.10

**On Linux/Ubuntu:**

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

**On Windows:**

- Download from python.org
- Make sure to check "Add to PATH"

#### Step 2: Create Virtual Environment

```bash
# Create environment
python3.10 -m venv excellCity-FLTL

# Activate environment
# Linux/Mac:
source excellCity-FLTL/bin/activate

# Windows:
excellCity-FLTL\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Main Dependencies

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

## 4. Directory Structure

```
TLIAS3/
│
├── FuncionesAux.py              # Main function library
├── FuncionesAuxi.py             # Auxiliary functions (alternative version)
│
├── 00.data/                     # Original data
│   └── I-BLEND/
│       └── meta_open.csv
│
├── 01.k-prot-clustering/        # K-Prototypes clustering
│   ├── clustering-iii15.py      # Clustering script
│   ├── Kprot.py                 # K-Prototypes algorithm
│   ├── kprototypes-labels-dist-iii15.csv  # Labels and distances
│   ├── kprototypes-centroids-iii15.csv    # Centroids
│   └── kprototypes-X_hat-iii15.csv        # Imputed data
│
├── transfer-learning/           # Transfer Learning models
│   └── modelsProtoScaled5/
│       ├── mod0.json            # Cluster 0 model architecture
│       ├── mod0.weights.h5      # Cluster 0 model weights
│       ├── mod0_metrics.csv     # Cluster 0 model metrics
│       └── ... (mod1 to mod14)
│
├── metricas/                    # FL experiment results
│   ├── Metricas_cliente_1.csv
│   ├── Metricas_cliente_2.csv
│   └── ...
│
├── metricas_TL/                 # Transfer Learning results
│   ├── cluster_0/
│   │   ├── edificio_1.csv
│   │   └── ...
│   └── cluster_1/...
│
├── 02.tl/                       # TL comparative results
│   ├── resW-15-iv.csv           # Results WITH Transfer Learning
│   └── resWO-15-iv.csv          # Results WITHOUT Transfer Learning
│
├── imagenes/                    # Visualizations
│   ├── pdf/
│   └── ...
│
├── Execution Scripts:
├── 00.createModelsScaled-i.py   # Create prototype models
├── lanzador.py                  # Main launcher
├── lanzador_parse.py            # Launcher with arguments
├── lanzador_allClients.py       # FL for all clients in a cluster
├── fl_mismocluster.py           # FL within the same cluster
├── barplots.py                  # Results visualization
├── mostrar_metricas.py          # Display metrics
│
└── requirements.txt             # Project dependencies
```

---

## 5. Complete Workflow

### 5.1 Flow Diagram

```
┌────────────────────────────────────────────────────────────┐
│                  STEP 1: CLUSTERING                         │
│  Group buildings by structural characteristics              │
│  Script: 01.k-prot-clustering/clustering-iii15.py          │
│  Output: Labels, centroids, imputed data                   │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│         STEP 2: CREATE PROTOTYPE MODELS (1 per cluster)    │
│  Train ConvLSTM2D model for representative building        │
│  Script: 00.createModelsScaled-i.py                        │
│  Output: transfer-learning/modelsProtoScaled5/mod{i}.*     │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│              STEP 3A: TRANSFER LEARNING                     │
│  Adapt prototype model to individual buildings             │
│  Script: lanzador.py (transfer_learning function)          │
│  Output: metricas_TL/cluster_{i}/edificio_{j}.csv          │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│              STEP 3B: FEDERATED LEARNING                    │
│  Distributed collaborative training (Fed+)                  │
│  Scripts: lanzador_parse.py, fl_mismocluster.py,          │
│           lanzador_allClients.py                           │
│  Output: metricas/Metricas_cliente_{i}.csv                 │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────┐
│           STEP 4: ANALYSIS AND VISUALIZATION                │
│  Compare results, generate charts                          │
│  Scripts: barplots.py, mostrar_metricas.py                 │
│  Output: imagenes/pdf/                                      │
└────────────────────────────────────────────────────────────┘
```

---

## 6. Main Modules

### 6.1 FuncionesAux.py

Central library with all project functions.

#### 6.1.1 Preprocessing Functions

```python
myround(x, base=5)
```

Rounds a number to the nearest multiple of `base`.

```python
split_dataset(data, factor=7, n_out=7)
```

Splits data into train/test (77%/23%) in time windows.

```python
to_supervised(train, n_input, n_out=7)
```

Converts time series to supervised format (X, y).

#### 6.1.2 Modeling Functions

```python
create_model(N_OUT, n_seq, n_substeps, n_features)
```

Creates ConvLSTM2D model with configurable parameters.

- **N_OUT**: Prediction horizon (24h)
- **n_seq**: Input sequences (7 days)
- **n_substeps**: Steps per sequence (24h)
- **n_features**: Input variables (2: consumption + temp)

```python
FitForecast_e(model, X_train, y_train, X_test, N_OUT,
              n_seq, n_substeps, n_features, epochs, number, trained_model)
```

Trains model and generates predictions.

- **trained_model**: If not None, uses Transfer Learning

#### 6.1.3 K-Prototypes Clustering

```python
kprot_missing(X, n_clusters, cat_index, centroids, max_iter=10)
```

K-Prototypes with missing value handling.

- **cat_index**: Categorical variable indices [1,2,3,5,7,8,10]
- Imputes missing values with mean (numerical) and mode (categorical)

```python
make_Kpods()
```

Complete clustering pipeline:

1. Load building data
2. Convert categorical variables to numerical
3. Run K-Prototypes (15 clusters)
4. Save results

#### 6.1.4 Transfer Learning

```python
transfer_learning(cluster, lastCommits, resWO, resW, count, n_muestras)
```

Evaluates Transfer Learning for a cluster:

1. Loads cluster prototype model
2. For each building in the cluster:
   - Trains WITHOUT loaded model (baseline)
   - Trains WITH loaded model (TL)
   - Compares metrics
3. Saves results in `metricas_TL/cluster_{i}/`

#### 6.1.5 Federated Learning

**Main Class:**

```python
class tfmlpClient(fl.client.NumPyClient)
```

Flower Framework client with:

- `fit()`: Local training with Fed+
- `evaluate()`: Global model evaluation
- Saves final models in last round

**Fed+ Algorithm:**

```python
fedplus(weights, mean, theta)
```

```
new_weights = mean + theta * (local_weights - mean)
theta = 1 / (1 + alpha * lr)
```

**Server Functions:**

```python
start_server(parties, lista_parties, lastscommits, rounds, alp, lea)
start_server_tl(parties, rounds, alp, lea)
```

Starts FL server with FedAvg strategy.

**Client Functions:**

```python
start_client(client_n, lista_clientes, lastscommits)
start_client_tl(client_n, names, cluster)
```

Starts FL client that connects to the server.

#### 6.1.6 Metrics

```python
compute_metrics_fn(y_valid_resc, y_hat_resc)
```

Calculates 5 metrics:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **CVRMSE**: Coefficient of Variation of RMSE (%)
- **MAPE**: Mean Absolute Percentage Error (%)

---

## 7. Execution Scripts

### 7.1 Clustering: clustering-iii15.py

**Purpose:** Group buildings into 15 clusters using K-Prototypes.

**Features used:**

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

**Execution:**

```bash
cd 01.k-prot-clustering
python clustering-iii15.py
```

**Outputs:**

- `kprototypes-labels-dist-iii15.csv`: Labels and distances
- `kprototypes-centroids-iii15.csv`: Centroids for each cluster
- `kprototypes-df-iii15.csv`: Original data
- `kprototypes-X_hat-iii15.csv`: Data with imputed values

---

### 7.2 Model Creation: 00.createModelsScaled-i.py

**Purpose:** Train 15 prototype models (one per cluster).

**Process:**

1. For each cluster (0-14):
   - Identifies building closest to centroid
   - Downloads consumption and temperature data
   - Preprocesses: normalization, time windows
   - Trains ConvLSTM2D model (5 epochs)
   - Saves model (.json + .weights.h5) and metrics

**Execution:**

```bash
python 00.createModelsScaled-i.py
```

**Outputs:**

```
transfer-learning/modelsProtoScaled5/
├── mod0.json, mod0.weights.h5, mod0_metrics.csv
├── mod1.json, mod1.weights.h5, mod1_metrics.csv
└── ... (up to mod14)
```

---

### 7.3 Transfer Learning and Federated Learning: lanzador.py

**Purpose:** Evaluate Transfer Learning on all buildings OR run Federated Learning with representative buildings.

**Execution Modes:**

The script now supports two execution modes via command-line arguments:

1. **TL Mode (Transfer Learning)**: Evaluates TL on all buildings
2. **FL Mode (Federated Learning)**: Runs FL with representative buildings

**Arguments:**

```bash
python lanzador.py [--mode {TL,FL}] [--clusters N] [--rounds N] [--samples N] [--alpha X] [--lr X]
```

| Argument     | Description                           | Default |
| ------------ | ------------------------------------- | ------- |
| `--mode`     | Execution mode: `TL` or `FL`          | `TL`    |
| `--clusters` | Number of clusters to process         | `15`    |
| `--rounds`   | Number of rounds for FL               | `15`    |
| `--samples`  | Number of samples (negative = last N) | `-2000` |
| `--alpha`    | Alpha parameter for Fed+              | `2.9`   |
| `--lr`       | Learning rate                         | `0.001` |

**Usage Examples:**

**1. Run Transfer Learning (default mode):**

```bash
# Transfer Learning with default configuration (15 clusters, -2000 samples)
python lanzador.py

# Transfer Learning with 10 clusters
python lanzador.py --mode TL --clusters 10

# Transfer Learning with only the last 1000 samples
python lanzador.py --mode TL --samples -1000
```

**2. Run Federated Learning:**

```bash
# Federated Learning with default configuration
python lanzador.py --mode FL

# FL with custom parameters
python lanzador.py --mode FL --rounds 20 --alpha 2.5 --lr 0.0001

# FL with fewer clusters (prototypes)
python lanzador.py --mode FL --clusters 10 --rounds 10
```

**TL Mode Process:**

1. Creates clients (representative buildings from each cluster)
2. Reads clustering labels
3. For each cluster (0 to k-1):
   - For each building in the cluster:
     - Trains baseline model (without TL)
     - Trains model with TL
     - Compares metrics
4. Saves comparative results

**FL Mode Process:**

1. Creates clients (representative buildings from each cluster)
2. Starts FL server on localhost:8080
3. Waits 30s for server to be ready
4. Launches k client processes in parallel
5. Each client:
   - Loads its data
   - Trains locally with Fed+
   - Sends weights to server
6. Server aggregates weights (FedAvg)
7. Repeats for `rounds` iterations

**Outputs:**

**TL Mode:**

- `resWO-15-i.csv`: Results WITHOUT Transfer Learning
- `resW-15-i.csv`: Results WITH Transfer Learning
- `metricas_TL/cluster_{i}/edificio_{j}.csv`: Detailed metrics per building

**FL Mode:**

- `metricas/Metricas_cliente_{i}.csv`: Metrics per client (i=1 to k)
- `transfer-learning/modelsProtoScaled5/mod{i}.json`: Trained models (last round)
- `transfer-learning/modelsProtoScaled5/mod{i}.weights.h5`: Model weights

**Execution:**

```bash
# Transfer Learning mode (default)
python lanzador.py

# Federated Learning mode
python lanzador.py --mode FL
```

**Verification:**

```bash
# Verify TL results
ls resWO-15-i.csv resW-15-i.csv
ls metricas_TL/cluster_*/edificio_*.csv

# Verify FL results
ls metricas/Metricas_cliente_*.csv
```

---

### 7.4 Federated Learning with Parameters: lanzador_parse.py

**Purpose:** Run FL with 15 clients (one per cluster) with adjustable parameters.

**Arguments:**

```bash
python lanzador_parse.py <alpha> <lr>
```

- **alpha**: Fed+ algorithm parameter (e.g., 2.9)
- **lr**: Learning rate (e.g., 0.001)

**Configuration:**

```python
rondas = 4  # Number of FL rounds
n_clientes = 15  # 15 clients (prototypes)
```

**Process:**

1. Creates 15 clients (representative buildings)
2. Starts FL server on localhost:8080
3. Waits 30s for server to be ready
4. Launches 15 client processes in parallel
5. Each client:
   - Loads its data
   - Trains locally with Fed+
   - Sends weights to server
6. Server aggregates weights (FedAvg)
7. Repeats for `rondas` iterations

**Execution:**

```bash
python lanzador_parse.py 2.9 0.001
```

**Outputs:**

- `metricas/Metricas_cliente_{i}.csv` (i=1 to 15)

---

### 7.5 FL per Cluster: fl_mismocluster.py

**Purpose:** FL only with buildings from the same cluster.

**Arguments:**

```bash
python fl_mismocluster.py <alpha> <lr>
```

**Configuration:**

```python
cluster = 1  # Cluster to process
rondas = 3
```

**Process:**

1. Identifies buildings in specified `cluster`
2. Counts number of buildings → number of clients
3. Starts FL server with TL (uses prototype model)
4. Launches one process per building in the cluster
5. FL training within the cluster

**Execution:**

```bash
python fl_mismocluster.py 2.9 0.001
```

---

### 7.6 FL All Buildings in a Cluster: lanzador_allClients.py

**Purpose:** FL with ALL buildings in a specific cluster (not just the representative one).

**Arguments:**

```bash
python lanzador_allClients.py <cluster> <alpha> <lr>
```

- **cluster**: Cluster ID (0-14)

**Process:**

1. Reads clustering labels
2. Gets ALL buildings in the cluster
3. Launches FL with all of them

**Execution:**

```bash
python lanzador_allClients.py 0 2.9 0.001
```

---

### 7.7 Visualization: barplots.py

**Purpose:** Generate comparative bar charts.

**Usage example:**

```python
# Compare metrics from 15 prototypes
cvrmse = [6.48, 34.86, 8.18, ...]  # For each cluster
plt.bar(x_range, cvrmse)
plt.savefig('imagenes/metrics.pdf')
```

---

## 8. Step-by-Step Guide

### 8.1 Complete Project from Scratch

#### STEP 1: Environment Setup

```bash
# 1. Clone/download project
cd TLIAS3

# 2. Create virtual environment
python3.10 -m venv excellCity-FLTL
source excellCity-FLTL/bin/activate  # Linux/Mac
# excellCity-FLTL\Scripts\activate  # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### STEP 2: Building Clustering

```bash
cd 01.k-prot-clustering
python clustering-iii15.py
cd ..
```

**Verification:**

```bash
# Should exist:
ls 01.k-prot-clustering/kprototypes-labels-dist-iii15.csv
ls 01.k-prot-clustering/kprototypes-centroids-iii15.csv
```

#### STEP 3: Create Prototype Models

```bash
python 00.createModelsScaled-i.py
```

**Duration:** ~30-60 minutes (depends on hardware)

**Verification:**

```bash
# Should have 15 models:
ls transfer-learning/modelsProtoScaled5/mod*.json
ls transfer-learning/modelsProtoScaled5/mod*.weights.h5
```

#### STEP 4A: Evaluate Transfer Learning

```bash
# Run in Transfer Learning mode (default)
python lanzador.py

# Or explicitly specify TL mode
python lanzador.py --mode TL

# With custom configuration
python lanzador.py --mode TL --clusters 10 --samples -1000
```

**Duration:** Several hours (processes ~500 buildings)

**Verification:**

```bash
ls metricas_TL/cluster_*/edificio_*.csv
ls resWO-15-i.csv  # Without TL
ls resW-15-i.csv   # With TL
```

#### STEP 4B: Run Federated Learning

**Option 1: FL with Prototypes (15 representative clients)**

```bash
# Run in Federated Learning mode
python lanzador.py --mode FL

# With custom parameters
python lanzador.py --mode FL --rounds 20 --alpha 2.5 --lr 0.0001
```

**Option 2: FL with Specific Parameters**

```bash
python lanzador_parse.py 2.9 0.001
```

**Option 3: FL within a Cluster**

```bash
# Edit fl_mismocluster.py: change cluster = 1 to desired one
python fl_mismocluster.py 2.9 0.001
```

**Option 4: FL with All Buildings in a Cluster**

```bash
python lanzador_allClients.py 0 2.9 0.001  # Cluster 0
```

**Verification:**

```bash
ls metricas/Metricas_cliente_*.csv
```

#### STEP 5: Analyze Results

```bash
# View metrics
python mostrar_metricas.py

# Generate charts
python barplots.py
ls imagenes/pdf/metrics.pdf
```

---

### 8.2 Typical Experiments

#### Experiment 1: Compare TL vs No TL

```bash
# 1. Train base models
python 00.createModelsScaled-i.py

# 2. Evaluate TL
python lanzador.py

# 3. Analyze resWO-15-i.csv (without TL) vs resW-15-i.csv (with TL)
python -c "
import pandas as pd
resWO = pd.read_csv('resWO-15-i.csv', sep=';')
resW = pd.read_csv('resW-15-i.csv', sep=';')
print('Average CVRMSE Without TL:', resWO.iloc[:,4].mean())
print('Average CVRMSE With TL:', resW.iloc[:,4].mean())
"
```

#### Experiment 2: Optimize Fed+ Hyperparameters

```bash
# Test different combinations of alpha and lr
for alpha in 1.0 2.0 2.9 3.5; do
  for lr in 0.0001 0.001 0.01; do
    echo "Testing alpha=$alpha, lr=$lr"
    python lanzador_parse.py $alpha $lr
    # Save results in tunning_fl_alpha_${alpha}_lr_${lr}.csv
  done
done
```

#### Experiment 3: Evaluate FL per Cluster

```bash
# For each cluster, run FL only with its buildings
for cluster in {0..14}; do
  echo "Processing cluster $cluster"
  python lanzador_allClients.py $cluster 2.9 0.001
done
```

---

## 9. Metrics and Results

### 9.1 Metrics Used

| Metric     | Formula                                                      | Interpretation                 |
| ---------- | ------------------------------------------------------------ | ------------------------------ |
| **MAE**    | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$                        | Mean absolute error            |
| **MSE**    | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$                        | Mean squared error             |
| **RMSE**   | $\sqrt{MSE}$                                                 | Root mean squared error        |
| **CVRMSE** | $\frac{RMSE}{\bar{y}} \times 100$                            | Normalized RMSE (%)            |
| **MAPE**   | $\frac{1}{n}\sum \frac{\|y_i - \hat{y}_i\|}{y_i} \times 100$ | Mean absolute percentage error |

**Main Metric:** **CVRMSE** (used in energy papers)

### 9.2 Results File Structure

#### FL Metrics (metricas/)

```csv
MAE,MSE,RSME,CVRMSE,MAPE
2.98,14.81,3.85,6.48,4.98
1.99,11.13,3.34,34.86,400.01
...
```

Each row = 1 FL round

#### TL Metrics (metricas_TL/cluster_X/edificio_Y.csv)

```csv
Cargado,MAE,MSE,RSME,CVRMSE,MAPE,Muestras
Si,2.5,10.2,3.2,8.5,5.1,-2000
No,3.1,12.5,3.5,9.2,6.3,-2000
```

- **Cargado=Si:** With Transfer Learning
- **Cargado=No:** Without Transfer Learning

#### TL Comparative Results (02.tl/)

```csv
cluster,edificio_id,MAE,MSE,RMSE,CVRMSE,MAPE
0,5,2.98,14.81,3.85,6.48,4.98
0,12,3.2,15.3,3.91,6.52,5.1
...
```

### 9.3 Expected Results

**Typical CVRMSE by building type:**

- **Offices:** 6-10%
- **Primary Schools:** 8-12%
- **Universities (classrooms):** 20-35%
- **Dormitories:** 30-40%
- **Laboratories:** 40-60%

**Improvement with TL:** 5-15% reduction in CVRMSE

**Improvement with FL:** 10-25% reduction vs isolated training

---

## 10. Troubleshooting

### 10.1 Common Problems

#### Error: "ModuleNotFoundError: No module named 'flwr'"

**Solution:**

```bash
pip install flwr==0.19.0
```

#### Error: "GPU memory allocation failed"

**Solution:**

```python
# Already implemented in scripts:
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

#### Error: "Connection refused" in FL

**Cause:** Server is not ready when clients try to connect.

**Solution:**

```python
# Increase wait time in scripts:
time.sleep(30)  # Change to 60 if necessary
```

#### Error: "FileNotFoundError: kprototypes-labels-dist-iii15.csv"

**Solution:**

```bash
# Run clustering first:
cd 01.k-prot-clustering
python clustering-iii15.py
```

#### Warning: "UserWarning: X has missing values"

**Cause:** Incomplete building data (expected).

**Solution:** Ignore (the kprot_missing algorithm imputes automatically).

### 10.2 Performance Optimization

#### For limited GPU:

```python
# In scripts, reduce batch size:
config = {
    "batch_size": 32,  # Change to 16 or 8
    ...
}
```

#### To speed up clustering:

```python
# In clustering-iii15.py:
cls = KPrototypes(n_clusters=15, init='Huang',
                  n_init=1,  # Reduce from 10 to 1
                  max_iter=100)  # Reduce from 1000 to 100
```

#### To reduce training time:

```python
# In 00.createModelsScaled-i.py:
n_epochs = 5  # Already at minimum
# Reduce data:
dataset = dataset[-1000:]  # Instead of entire dataset
```

### 10.3 Debugging

#### Verify building data:

```python
from FuncionesAux import *
refs = lsremote('https://bitbucket.org/aurorax/datangi')
lastCommit = refs['HEAD']
building = 'Office_Gustavo'
cons = pd.read_csv(f'https://bitbucket.org/aurorax/datangi/raw/{lastCommit}/processed/study1-energy/{building}-tsCons.csv', sep=';', index_col=0)
print(cons.head())
print(cons.shape)
```

#### View clustering labels:

```python
import pandas as pd
labels = pd.read_csv('01.k-prot-clustering/kprototypes-labels-dist-iii15.csv')
print(labels.groupby(labels.columns[1]).size())  # Buildings per cluster
```

#### Verify saved model:

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

## Appendices

### A. Hyperparameter Configuration

**Model Parameters:**

```python
N_OUT = 24              # Prediction horizon (hours)
n_seq = 7               # Input days
n_substeps = 24         # Hours per day
n_features = 2          # Consumption + Temperature
filters = 64            # ConvLSTM2D filters
kernel_size = (1, 2)    # ConvLSTM2D kernel
activation = 'relu'     # Activation function
optimizer = 'adam'      # Optimizer
loss = 'mse'            # Loss function
```

**Training Parameters:**

```python
epochs = 300            # Local training epochs
train_split = 0.77      # 77% train, 23% test
```

**Federated Learning Parameters:**

```python
rondas = 4              # FL rounds
alpha = 2.9             # Fed+ parameter
lr = 0.001              # Learning rate
fraction_fit = 0.3      # Fraction of clients per round
min_fit_clients = 15    # Minimum clients
```

### B. Datasets Used

**Data Source:**

- **Repository:** https://bitbucket.org/aurorax/datangi
- **Studies:**
  - Study 1: Commercial/educational buildings
  - Study 2 (I-BLEND): Additional buildings

**Files:**

- `{building}-tsCons.csv`: Consumption time series
- `t{building}-ts.csv`: Temperature time series
- `meta_open.csv`: Building metadata

### C. References

**Algorithms:**

- K-Prototypes: Huang, Z. (1998)
- Federated Learning: McMahan et al. (2017) - FedAvg
- Fed+: Extension of FedAvg with regularization

**Frameworks:**

- Flower: https://flower.dev/
- TensorFlow: https://tensorflow.org/
- kmodes: https://github.com/nicodv/kmodes

---

## Contact and Contributions

**Author:** auroragonzalez  
**Repository:** TLIAS3  
**License:** See LICENSE file

---

**Last updated:** October 2025
