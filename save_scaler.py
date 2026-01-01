"""
Lưu scaler từ cleaned data - CHẠY FILE NÀY TRƯỚC
"""
import pickle
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dữ liệu (giống như trong notebook)
print("[+] Loading cleaned data...")
csv_files = glob.glob('./cleaned_data/*.csv')
dfs = []

for csv_file in csv_files[:3]:  # Load 3 file đầu cho nhanh
    print(f"    Reading {csv_file}...")
    # ⚠️ FIX: Đọc với low_memory=False và chỉ định header
    df = pd.read_csv(csv_file, low_memory=False, header=0)
    
    # Kiểm tra columns
    if 'Label' not in df.columns:
        print(f"    ⚠️  Warning: No 'Label' column in {csv_file}, skipping...")
        continue
    
    # Drop các cột không cần (nếu có Timestamp, Dst Port, etc.)
    drop_cols = [col for col in df.columns if col in ['Timestamp', 'Dst Port', 'Src Port', 'Src IP', 'Dst IP']]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"    Dropped columns: {drop_cols}")
    
    dfs.append(df)

if not dfs:
    raise ValueError("Không load được dữ liệu nào! Kiểm tra lại thư mục './cleaned_data'")

df_combined = pd.concat(dfs, ignore_index=True)
print(f"[+] Total flows loaded: {len(df_combined)}")
print(f"[+] Columns: {list(df_combined.columns)}")

# Tách features và label
if 'Label' not in df_combined.columns:
    raise ValueError("Dataset phải có cột 'Label'!")

X_df = df_combined.drop('Label', axis=1)
y = df_combined['Label'].values

print(f"\n[+] Features shape before cleaning: {X_df.shape}")
print(f"[+] Feature columns: {list(X_df.columns[:10])}... (showing first 10)")

# ⚠️ FIX: Convert tất cả về numeric, thay NaN/Inf bằng 0
print("\n[+] Converting to numeric and handling NaN/Inf...")
X_df = X_df.apply(pd.to_numeric, errors='coerce')  # Convert string -> NaN
X_df = X_df.replace([np.inf, -np.inf], np.nan)     # Inf -> NaN
X_df = X_df.fillna(0)                               # NaN -> 0

X = X_df.values  # Convert to numpy array

print(f"[+] X shape after cleaning: {X.shape}")
print(f"[+] y shape: {y.shape}")
print(f"[+] Data type: {X.dtype}")

# Verify không còn string
if X.dtype == 'object':
    raise ValueError("❌ X vẫn chứa object/string! Kiểm tra lại dữ liệu.")

# Fit scaler
print("\n[+] Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X)

print("[+] Scaler fitted successfully!")
print(f"    Mean shape: {scaler.mean_.shape}")
print(f"    Var shape: {scaler.var_.shape}")
print(f"    Sample mean (first 5): {scaler.mean_[:5]}")

# Save scaler
import os
os.makedirs('./CNN-LSTM/Time-Based Split/cnn_lstm', exist_ok=True)

scaler_path = './CNN-LSTM/Time-Based Split/cnn_lstm/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n[+] Scaler saved to {scaler_path}")

# Verify
print("\n[+] Verifying scaler...")
with open(scaler_path, 'rb') as f:
    loaded_scaler = pickle.load(f)

print(f"    Type: {type(loaded_scaler)}")
print(f"    Has transform: {hasattr(loaded_scaler, 'transform')}")

# Test transform
sample = X[0:1]  # 1 flow
print(f"    Sample shape: {sample.shape}")
print(f"    Sample values (first 5): {sample[0][:5]}")

transformed = loaded_scaler.transform(sample)
print(f"    Transformed shape: {transformed.shape}")
print(f"    Transformed values (first 5): {transformed[0][:5]}")

print("\n✅ Scaler created and verified successfully!")
print(f"✅ Ready to use with {X.shape[1]} features")
