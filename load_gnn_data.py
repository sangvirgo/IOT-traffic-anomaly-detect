import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class CICIDS_Loader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
    
    def load_selected_files(self, file_names, sample_per_file=None, chunk_size=100000):
        dfs = []
        
        for filename in file_names:
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"Warning: {filename} not found, skipping...")
                continue
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 500:
                df = self._load_file_chunked(filepath, sample_per_file, chunk_size)
            else:
                df = pd.read_csv(filepath)
                if sample_per_file and len(df) > sample_per_file:
                    df = df.sample(sample_per_file, random_state=42)
            
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No files loaded!")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df):,} rows from {len(dfs)} files")
        
        return combined_df
    
    def _load_file_chunked(self, filepath, sample_size=None, chunk_size=100000):
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if sample_size and total_rows >= sample_size * 1.2:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        
        return df
    
    def preprocess(self, df):
        if 'Label_Numeric' in df.columns:
            y = df['Label_Numeric'].apply(lambda x: 0 if x == 0 else 1).values
        else:
            raise ValueError("No Label_Numeric column!")
        
        exclude_cols = ['Label', 'Label_Numeric', 'Timestamp', 'Flow_ID']
        
        has_ip = 'Src_IP' in df.columns and 'Dst_IP' in df.columns
        
        if has_ip:
            src_ips = df['Src_IP'].values
            dst_ips = df['Dst_IP'].values
            exclude_cols.extend(['Src_IP', 'Dst_IP', 'Src_Port', 'Dst_Port'])
        else:
            src_ips = None
            dst_ips = None
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Features: {X_scaled.shape}, Attack: {np.sum(y==1):,} ({np.sum(y==1)/len(y)*100:.1f}%), Benign: {np.sum(y==0):,} ({np.sum(y==0)/len(y)*100:.1f}%)")
        
        return X_scaled, y, feature_cols, src_ips, dst_ips