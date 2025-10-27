# TLIAS3 - Transfer Learning & Federated Learning para Predicción Energética

Sistema de predicción de consumo energético en edificios inteligentes combinando Transfer Learning y Federated Learning.

## 🚀 Inicio Rápido

### 1. Activar Entorno Virtual

**Windows:**

```powershell
.\excellCity-FLTL\Scripts\activate
```

**Linux/Mac:**

```bash
source excellCity-FLTL/bin/activate
```

### 2. Verificar Instalación

```bash
python lanzador.py --help
```

### 3. Ejecutar Transfer Learning

```bash
# Configuración por defecto (15 clusters, -2000 muestras)
python lanzador.py

# Modo explícito
python lanzador.py --mode TL
```

### 4. Ejecutar Federated Learning

```bash
# Configuración por defecto
python lanzador.py --mode FL

# Personalizado
python lanzador.py --mode FL --rounds 10 --alpha 2.5 --lr 0.0001
```

## 📚 Documentación

- **[MANUAL.md](MANUAL.md)** - Manual completo del proyecto (paso a paso)
- **[EJEMPLOS_USO.md](EJEMPLOS_USO.md)** - Ejemplos prácticos de uso

## 🎯 Modos de Ejecución

### Modo TL (Transfer Learning)

Evalúa Transfer Learning en todos los edificios del dataset.

```bash
python lanzador.py --mode TL [--clusters N] [--samples N]
```

**Outputs:**

- `resWO-15-i.csv` - Resultados SIN Transfer Learning
- `resW-15-i.csv` - Resultados CON Transfer Learning
- `metricas_TL/cluster_*/edificio_*.csv` - Métricas detalladas

### Modo FL (Federated Learning)

Ejecuta Federated Learning con edificios representativos.

```bash
python lanzador.py --mode FL [--rounds N] [--alpha X] [--lr X]
```

**Outputs:**

- `metricas/Metricas_cliente_*.csv` - Métricas por cliente/ronda

## 🛠️ Argumentos Disponibles

| Argumento        | Descripción        | Por Defecto |
| ---------------- | ------------------ | ----------- |
| `--mode {TL,FL}` | Modo de ejecución  | `TL`        |
| `--clusters N`   | Número de clusters | `15`        |
| `--rounds N`     | Rondas de FL       | `15`        |
| `--samples N`    | Muestras a usar    | `-2000`     |
| `--alpha X`      | Parámetro Fed+     | `2.9`       |
| `--lr X`         | Learning rate      | `0.001`     |

## 📊 Flujo de Trabajo Típico

```bash
# 1. Clustering (una vez)
cd 01.k-prot-clustering
python clustering-iii15.py
cd ..

# 2. Crear modelos prototipo (una vez)
python 00.createModelsScaled-i.py

# 3. Evaluar Transfer Learning
python lanzador.py --mode TL

# 4. Evaluar Federated Learning
python lanzador.py --mode FL

# 5. Visualizar resultados
python barplots.py
```

## 🔧 Otros Scripts FL

### lanzador_parse.py

```bash
python lanzador_parse.py <alpha> <lr>
```

### fl_mismocluster.py

FL dentro de un cluster específico (editar `cluster` en el código):

```bash
python fl_mismocluster.py <alpha> <lr>
```

### lanzador_allClients.py

FL con todos los edificios de un cluster:

```bash
python lanzador_allClients.py <cluster_id> <alpha> <lr>
```

## 📈 Métricas Calculadas

- **MAE** - Mean Absolute Error
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **CVRMSE** - CV of RMSE (%) ← _métrica principal_
- **MAPE** - Mean Absolute Percentage Error (%)

## 🐛 Troubleshooting

### ModuleNotFoundError

```bash
# Activar entorno virtual primero
.\excellCity-FLTL\Scripts\activate  # Windows
source excellCity-FLTL/bin/activate # Linux/Mac
```

### Connection refused (FL)

```bash
# Aumentar tiempo de espera o ejecutar con menos clientes
python lanzador.py --mode FL --clusters 5
```

### GPU out of memory

```bash
# Reducir número de muestras
python lanzador.py --mode TL --samples -500
```

## 📦 Estructura del Proyecto

```
TLIAS3/
├── lanzador.py                    # Script principal (TL/FL)
├── FuncionesAux.py               # Biblioteca de funciones
├── 00.createModelsScaled-i.py   # Crear prototipos
├── 01.k-prot-clustering/        # Clustering K-Prototypes
├── transfer-learning/           # Modelos prototipo
├── metricas/                    # Resultados FL
├── metricas_TL/                 # Resultados TL
├── MANUAL.md                    # Manual completo
└── EJEMPLOS_USO.md             # Ejemplos de uso
```

## 👤 Autor

**Aurora González**

- Repository: [auroragonzalez/TLIAS3](https://github.com/auroragonzalez/TLIAS3)

## 📄 Licencia

Ver archivo [LICENSE](LICENSE)

---

**Última actualización:** Octubre 2025
