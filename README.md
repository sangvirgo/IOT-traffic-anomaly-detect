# CICIDS2018 Data Preprocessing Pipeline

Complete pipeline for cleaning, splitting, and preparing CICIDS2018 dataset for GNN, CNN, and LSTM models.

## 📁 Project Structure
venv\Scripts\activate
```
project/
├── raw_data/                    # Raw CSV files từ CICIDS2018
│   ├── 02-14-2018.csv
│   ├── 02-15-2018.csv
│   └── ...
├── cleaned_data/               # Cleaned data
│   ├── for_gnn/               # Version có IP addresses (cho GNN)
│   │   ├── cleaned_02-14-2018.csv
│   │   └── ...
│   └── for_cnn_lstm/          # Version không có IP (cho CNN/LSTM)
│       ├── cleaned_02-14-2018.csv
│       └── ...
├── split_data/                # Train/Val/Test splits
│   ├── processed/
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_val.npy
│   │   ├── y_val.npy
│   │   ├── X_test.npy
│   │   └── y_test.npy
│   ├── scalers/
│   │   ├── feature_scaler.pkl
│   │   └── feature_names.pkl
│   └── metadata.pkl
├── clean_cicids.py            # Cleaning script
├── split_data.py              # Splitting script
└── utils.py                   # Inspection utilities
```

## 🚀 Quick Start

### Requirement
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

# For deep learning:
pip install tensorflow  # For CNN/LSTM

# For GNN:
pip install torch torch-geometric
```

### Step 1: Clean Raw Data

```bash
# For both GNN and CNN/LSTM (recommended)
python clean_cicids.py \
    --input_dir ./raw_data \
    --output_dir ./cleaned_data \
    --mode both

# Only for GNN (keeps IP addresses)
python clean_cicids.py \
    --input_dir ./raw_data \
    --output_dir ./cleaned_data \
    --mode gnn

# Only for CNN/LSTM (removes IP addresses)
python ./pretrain/clean_cicids.py --input_dir ./raw_data --output_dir ./cleaned_data --mode cnn_lstm
```

**What it does:**
- ✅ Removes duplicates
- ✅ Handles missing values (NaN, Inf)
- ✅ Removes invalid flows (negative duration, etc.)
- ✅ Parses timestamps
- ✅ Maps attack labels to numeric (0-5)
- ✅ Removes low-variance features
- ✅ Handles outliers (winsorization)
- ✅ Log transforms skewed features

### Step 2: Split Data

```bash
# Temporal split (recommended for time-series data)
python ./pretrain/split_data.py --input_dir ./cleaned_data/ --output_dir ./split_data --temporal
```

**What it does:**
- ✅ Splits data 70-15-15 (train-val-test)
- ✅ Balances classes using SMOTE + Undersampling
- ✅ Normalizes features using StandardScaler
- ✅ Saves as numpy arrays for fast loading

### Step 3: Inspect Data

```bash
# Inspect cleaned files
python ./pretrain/utils.py --inspect --data_dir ./cleaned_data/

# Inspect split data
python ./pretrain/utils.py --inspect --data_dir ./split_data

# Visualize label distribution
python ./pretrain/utils.py --visualize --data_dir ./split_data

# Show feature statistics
python ./pretrain/utils.py --statistics --data_dir ./split_data
```

## 📊 Label Mapping

```python
label_mapping = {
    'Benign': 0,
    'Bot': 1,
    'DDoS': 2,              # Includes all DDoS variants
    'DoS-*': 3,             # All DoS attacks grouped
    'Brute Force': 4,       # FTP, SSH, Web, XSS, SQL Injection
    'Infiltration': 5,
}
```

**Đặc điểm:**
- Class 0 (Benign): Traffic bình thường
- Class 1 (Bot): Botnet traffic
- Class 2 (DDoS): Distributed Denial of Service
- Class 3 (DoS): Denial of Service (single source)
- Class 4 (Brute Force): Các attack brute force
- Class 5 (Infiltration): Infiltration attacks

## 💻 Load Data for Training

### For CNN/LSTM

```python
import numpy as np
from utils import DataLoader

# Load data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    DataLoader.load_data('./split_data')

# Load metadata
metadata = DataLoader.load_metadata('./split_data')
print(f"Number of features: {metadata['n_features']}")
print(f"Number of classes: {metadata['n_classes']}")

# Reshape for CNN (example: 1D CNN)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Reshape for LSTM (example: sequence of 10 timesteps)
timesteps = 10
X_train_lstm = X_train.reshape(X_train.shape[0], timesteps, -1)
X_val_lstm = X_val.reshape(X_val.shape[0], timesteps, -1)
X_test_lstm = X_test.reshape(X_test.shape[0], timesteps, -1)
```

### For GNN

```python
import pandas as pd
import torch
from torch_geometric.data import Data

# Load cleaned data with IP addresses
df = pd.read_csv('./cleaned_data/for_gnn/cleaned_02-14-2018.csv')

# Create nodes (unique IPs)
all_ips = pd.concat([df['Src_IP'], df['Dst_IP']]).unique()
ip_to_idx = {ip: idx for idx, ip in enumerate(all_ips)}

# Create edges
edge_index = torch.tensor([
    [ip_to_idx[src] for src in df['Src_IP']],
    [ip_to_idx[dst] for dst in df['Dst_IP']]
], dtype=torch.long)

# Node features (aggregate per IP)
node_features = []  # Aggregate flow stats per IP
node_labels = []    # Label per IP

# Create PyG Data object
data = Data(
    x=torch.tensor(node_features, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(node_labels, dtype=torch.long)
)
```

## 📈 Expected Data Sizes

### CICIDS2018 Full Dataset
- **Raw**: ~15 GB (10 CSV files)
- **Cleaned**: ~10 GB
- **Split numpy**: ~2-3 GB

### After Class Balancing
- Training set: ~1-2M samples (balanced)
- Validation set: ~200-300K samples
- Test set: ~200-300K samples

## ⚙️ Configuration Options

### Cleaning Parameters

```python
# In clean_cicids.py

# Label mapping (customize for your needs)
label_mapping = {
    'Benign': 0,
    'Bot': 1,
    'DDoS': 2,
    # Add more...
}

# Features to remove
features_to_remove = [
    'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg',
    # Add more...
]

# Outlier handling (percentiles)
lower_percentile = 0.01  # 1st percentile
upper_percentile = 0.99  # 99th percentile
```

### Splitting Parameters

```python
# In split_data.py

# Split ratios
train_ratio = 0.7   # 70% train
val_ratio = 0.15    # 15% validation
test_ratio = 0.15   # 15% test

# Class balancing strategy
over_sampling = SMOTE(sampling_strategy='auto')
under_sampling = RandomUnderSampler(sampling_strategy='auto')
```

## 🔍 Troubleshooting

### Problem: Out of Memory

**Solution 1**: Process files one by one
```python
# Modify clean_cicids.py
chunksize = 50000  # Reduce chunk size
```

**Solution 2**: Sample data
```python
# After loading
df = df.sample(frac=0.5, random_state=42)  # Use 50% of data
```

### Problem: Class Imbalance Still High

**Solution**: Adjust sampling strategy
```python
# In split_data.py
# Manual sampling strategy
sampling_strategy = {
    0: 100000,  # Benign: reduce to 100k
    1: 50000,   # Bot: oversample to 50k
    2: 50000,   # DDoS: oversample to 50k
    # etc.
}
over = SMOTE(sampling_strategy=sampling_strategy)
```

### Problem: Features Not Scaling Properly

**Solution**: Check for constant features
```python
# Remove constant or near-constant features
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_train_filtered = selector.fit_transform(X_train)
```



CICIDS2018 Dataset: https://www.unb.ca/cic/datasets/ids-2018.html


## ✅ Checklist

- [ ] Downloaded CICIDS2018 raw data
- [ ] Cleaned data (both GNN and CNN/LSTM versions)
- [ ] Split data (train/val/test)
- [ ] Inspected data quality
- [ ] Implemented CNN model
- [ ] Implemented LSTM model
- [ ] Implemented GNN model
- [ ] Compared model performances
- [ ] Wrote report

## 📞 Support

Nếu gặp lỗi, check:
1. File paths đúng chưa
2. Memory đủ không (recommend 16GB RAM)
3. Dependencies đã cài đủ chưa
4. Data format đúng chưa (CSV with headers)
