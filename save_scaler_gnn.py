import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

print("=" * 70)
print("Creating GNN Scaler from Cleaned Data")
print("=" * 70)

# === CHỈ CẦN 1 FILE CLEANED ===
CLEANED_DIR = Path("./cleaned_data")
FILES = sorted(CLEANED_DIR.glob("cleaned_*.csv"))

if not FILES:
    raise FileNotFoundError("No cleaned files found! Run clean_tmp_data.py first")

# Lấy file đầu tiên
csv_file = FILES[0]
print(f"\nLoading: {csv_file.name}")

# Load data (đủ 100K-300K rows để đại diện)
df = pd.read_csv(csv_file, low_memory=False, nrows=300_000)
print(f"Loaded {len(df):,} rows")

# === PREPROCESSING GIỐNG NOTEBOOK ===
# 1. Standardize columns
df.columns = df.columns.str.strip().str.replace(' ', '_')

# 2. Separate Label
if 'Label' not in df.columns:
    raise ValueError("No Label column!")

# 3. Define columns to exclude (EXACTLY AS NOTEBOOK)
exclude_cols = [
    'Label', 'Label_Numeric', 'Timestamp', 'Flow_ID', 'Flow_ID_',
    'Src_IP', 'Source_IP', 'SrcIP', 'Src_Port',
    'Dst_IP', 'Destination_IP', 'DstIP', 'Dest_IP', 'Dst_Port'
]

# 4. Extract numeric features
print("\nExtracting numeric features...")
feature_cols = []
for col in df.columns:
    # Skip excluded columns
    if col in exclude_cols or any(excl in col for excl in exclude_cols):
        continue
    
    # Convert to numeric
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            feature_cols.append(col)
    except:
        continue

print(f"Found {len(feature_cols)} features")

# 5. Get features
X = df[feature_cols].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nFeatures shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# === CRITICAL: Check feature count ===
EXPECTED = 77  # From notebook
if X.shape[1] != EXPECTED:
    print(f"\n⚠️  WARNING: Expected {EXPECTED} features, got {X.shape[1]}")
    print("\nFeature list:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Không raise error, vì có thể notebook dùng file khác
    print(f"\n⚠️  Model expects {EXPECTED} features!")
    print("   Solutions:")
    print("   1. Use the SAME raw file as notebook training")
    print("   2. OR retrain model with current features")
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(1)

# 6. Fit scaler
print("\nFitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X)

# 7. Save
os.makedirs("./saved_gnn", exist_ok=True)
scaler_path = "./saved_gnn/gnn_scaler.pkl"

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# 8. Save feature names
feature_names_path = "./saved_gnn/feature_names.txt"
with open(feature_names_path, 'w') as f:
    f.write(f"# Total: {len(feature_cols)} features\n")
    for i, col in enumerate(feature_cols, 1):
        f.write(f"{i:2d}. {col}\n")

print(f"\n✅ Scaler saved to: {scaler_path}")
print(f"✅ Feature names saved to: {feature_names_path}")

# 9. Verify
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

with open(scaler_path, 'rb') as f:
    loaded = pickle.load(f)

test_sample = X[0].reshape(1, -1)
transformed = loaded.transform(test_sample)

print(f"✓ Scaler type: {type(loaded).__name__}")
print(f"✓ Input shape: {test_sample.shape}")
print(f"✓ Output shape: {transformed.shape}")
print(f"✓ Mean shape: {scaler.mean_.shape}")
print(f"✓ Std shape: {scaler.scale_.shape}")

print("\n" + "=" * 70)
print(f"✅ Scaler ready with {X.shape[1]} features")
print("=" * 70)
