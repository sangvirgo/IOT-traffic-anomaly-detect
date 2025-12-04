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
                df = pd.read_csv(filepath, low_memory=False)
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
        # ✅ AUTO-CLEAN: Create Label_Numeric if not exists
        if 'Label_Numeric' not in df.columns:
            print("⚠️  Label_Numeric not found. Auto-cleaning data...")
            
            # Label mapping
            label_mapping = {
                'Benign': 0,
                'Bot': 1,
                'DDoS': 2,
                'DDOS attack-HOIC': 2,
                'DDOS attack-LOIC-UDP': 2,
                'DoS-GoldenEye': 3,
                'DoS-Slowloris': 3,
                'DoS-SlowHTTPTest': 3,
                'DoS-Hulk': 3,
                'DoS attacks-GoldenEye': 3,
                'DoS attacks-Slowloris': 3,
                'FTP-BruteForce': 4,
                'SSH-Bruteforce': 4,
                'Brute Force -Web': 4,
                'Brute Force -XSS': 4,
                'SQL Injection': 4,
                'Infiltration': 5,
            }
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            
            # Find Label column
            label_col = None
            for col in df.columns:
                if col.strip().lower() in ['label', ' label', 'label ']:
                    label_col = col
                    break
            
            if label_col is None:
                raise ValueError("No Label column found in dataset!")
            
            # Clean and map labels
            df['Label'] = df[label_col].astype(str).str.strip()
            df['Label_Numeric'] = df['Label'].map(label_mapping)
            
            # Remove rows with unmapped labels
            before_drop = len(df)
            df = df.dropna(subset=['Label_Numeric'])
            after_drop = len(df)
            
            if after_drop < before_drop:
                print(f"  ⚠️  Dropped {before_drop - after_drop:,} rows with unmapped labels")
            
            df['Label_Numeric'] = df['Label_Numeric'].astype(int)
            print(f"  ✓ Created Label_Numeric column")
            
            # Print label distribution
            print(f"  Label distribution:")
            for label, count in df['Label_Numeric'].value_counts().sort_index().items():
                print(f"    Class {label}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Binary classification: Benign (0) vs Attack (1)
        y = df['Label_Numeric'].apply(lambda x: 0 if x == 0 else 1).values
        
        # Exclude columns
        exclude_cols = ['Label', 'Label_Numeric', 'Timestamp', 'Flow_ID']
        
        # Check for IP columns
        has_ip = False
        src_ip_col = None
        dst_ip_col = None
        
        # Find IP columns (handle different naming)
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ['src ip', 'src_ip', 'source ip', 'source_ip']:
                src_ip_col = col
            elif col_lower in ['dst ip', 'dst_ip', 'destination ip', 'destination_ip', 'dest ip']:
                dst_ip_col = col
        
        has_ip = (src_ip_col is not None) and (dst_ip_col is not None)
        
        if has_ip:
            src_ips = df[src_ip_col].values
            dst_ips = df[dst_ip_col].values
            exclude_cols.extend([src_ip_col, dst_ip_col, 'Src_Port', 'Dst_Port', 'Src Port', 'Dst Port'])
        else:
            src_ips = None
            dst_ips = None
            print("  ℹ️  No IP columns found - using Flow-based approach only")
        
        # Get feature columns
        feature_cols = []
        for col in df.columns:
            # Skip excluded columns
            if any(excl.lower() in col.lower() for excl in exclude_cols):
                continue
            # Only numeric columns
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                feature_cols.append(col)
        
        if not feature_cols:
            raise ValueError("No numeric feature columns found!")
        
        X = df[feature_cols].values
        
        # Handle NaN and Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Features: {X_scaled.shape}, Attack: {np.sum(y==1):,} ({np.sum(y==1)/len(y)*100:.1f}%), Benign: {np.sum(y==0):,} ({np.sum(y==0)/len(y)*100:.1f}%)")
        
        return X_scaled, y, feature_cols, src_ips, dst_ips