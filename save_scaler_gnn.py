import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("=" * 60)
print("Creating GNN Scaler from Cleaned Data")
print("=" * 60)

# Load data
print("\nLoading cleaned data...")
df = pd.read_csv(
    "./cleaned_data/cleaned_02-14-2018.csv",
    low_memory=False,
    nrows=100000
)
print(f"Loaded {len(df)} rows")

# Drop columns
drop_cols = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Flow ID']
drop_cols = [col for col in drop_cols if col in df.columns]

print(f"\nDropping columns: {drop_cols}")
X = df.drop(columns=drop_cols)

# Convert to numeric
print("\nConverting to numeric and handling NaN/Inf...")
X = X.apply(pd.to_numeric, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.values

print(f"Features shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Fit scaler
print("\nFitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X)

# Save scaler
os.makedirs("./saved_gnn", exist_ok=True)
scaler_path = "./saved_gnn/gnn_scaler.pkl"

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nScaler saved to: {scaler_path}")
print(f"Mean shape: {scaler.mean_.shape}")
print(f"Sample mean (first 5): {scaler.mean_[:5]}")

# Verify
print("\nVerifying scaler...")
with open(scaler_path, 'rb') as f:
    loaded_scaler = pickle.load(f)

test_sample = X[0].reshape(1, -1)
transformed = loaded_scaler.transform(test_sample)

print(f"Scaler type: {type(loaded_scaler).__name__}")
print(f"Test sample shape: {test_sample.shape}")
print(f"Transformed shape: {transformed.shape}")

print("\n" + "=" * 60)
print(f"GNN Scaler ready! Use with {X.shape[1]} features")
print("=" * 60)
