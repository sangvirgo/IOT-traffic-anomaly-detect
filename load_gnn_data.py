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
            
            # ✅ FIX: Force all columns to be read properly
            if file_size_mb > 500:
                df = self._load_file_chunked(filepath, sample_per_file, chunk_size)
            else:
                # Read with proper dtype handling
                df = pd.read_csv(filepath, low_memory=False, encoding='utf-8', 
                                encoding_errors='ignore', on_bad_lines='skip')
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
        
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False,
                                 encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip'):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            if sample_size and total_rows >= sample_size * 1.2:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)
        
        return df
    
    def preprocess(self, df):
        print(f"Preprocessing {len(df):,} rows...")
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Create Label_Numeric if not exists
        if 'Label_Numeric' not in df.columns:
            print("Creating binary labels (Benign=0, Attack=1)...")
            
            label_col = None
            for col in df.columns:
                if col.lower() in ['label', 'label_']:
                    label_col = col
                    break
            
            if label_col is None:
                raise ValueError("No Label column found!")
            
            df = df.copy()
            df['Label'] = df[label_col].astype(str).str.strip()
            df['Label_Numeric'] = df['Label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)
            
            benign = (df['Label_Numeric'] == 0).sum()
            attack = (df['Label_Numeric'] == 1).sum()
            print(f"Labels: Benign={benign:,} ({benign/len(df)*100:.1f}%), Attack={attack:,} ({attack/len(df)*100:.1f}%)")
        
        y = df['Label_Numeric'].values
        
        # Exclude columns
        exclude_cols = ['Label', 'Label_Numeric', 'Timestamp', 'Flow_ID', 'Flow_ID_']
        
        # Check IP columns
        src_ip_col = None
        dst_ip_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['src_ip', 'source_ip', 'srcip']:
                src_ip_col = col
            elif col_lower in ['dst_ip', 'destination_ip', 'dstip', 'dest_ip']:
                dst_ip_col = col
        
        has_ip = (src_ip_col is not None) and (dst_ip_col is not None)
        
        if has_ip:
            src_ips = df[src_ip_col].values
            dst_ips = df[dst_ip_col].values
            exclude_cols.extend([src_ip_col, dst_ip_col, 'Src_Port', 'Dst_Port'])
        else:
            src_ips = None
            dst_ips = None
        
        # Get numeric features
        print("Converting features to numeric...", end=" ")
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols or any(excl in col for excl in exclude_cols):
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    feature_cols.append(col)
            except:
                continue
        
        if not feature_cols:
            raise ValueError(f"No numeric features found!")
        
        print(f"{len(feature_cols)} features")
        
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("Scaling features...", end=" ")
        X_scaled = self.scaler.fit_transform(X)
        print(f"Done. Shape: {X_scaled.shape}")
        
        return X_scaled, y, feature_cols, src_ips, dst_ips