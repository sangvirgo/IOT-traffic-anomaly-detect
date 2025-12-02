"""
CICIDS2018 Dataset Cleaning Pipeline - Memory Optimized Version
Supports both GNN (with IPs) and CNN/LSTM (without IPs)

Usage:
    python clean_cicids.py --input_dir ./raw_data --output_dir ./cleaned_data --mode cnn_lstm
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CICIDS2018Cleaner:
    def __init__(self, input_dir, output_dir, mode='both', chunk_size=500000):
        """
        Args:
            input_dir: Path to raw CSV files
            output_dir: Path to save cleaned files
            mode: 'gnn' (keep IPs), 'cnn_lstm' (remove IPs), 'both' (save both versions)
            chunk_size: Number of rows to process at once
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.chunk_size = chunk_size
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if mode == 'both':
            (self.output_dir / 'for_gnn').mkdir(exist_ok=True)
            (self.output_dir / 'for_cnn_lstm').mkdir(exist_ok=True)
        
        # Label mappings
        self.label_mapping = {
            'Benign': 0,
            'Bot': 1,
            'DDoS': 2,
            'DoS-GoldenEye': 3,
            'DoS-Slowloris': 3,
            'DoS-SlowHTTPTest': 3,
            'DoS-Hulk': 3,
            'FTP-BruteForce': 4,
            'SSH-Bruteforce': 4,
            'Brute Force -Web': 4,
            'Brute Force -XSS': 4,
            'SQL Injection': 4,
            'Infiltration': 5,
            'DDOS attack-HOIC': 2,
            'DDOS attack-LOIC-UDP': 2,
            'DoS attacks-GoldenEye': 3,
            'DoS attacks-Slowloris': 3,
        }
        
        # Features to remove (low variance or redundant)
        self.features_to_remove = [
            'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
            'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
            'Fwd Header Len.1',
        ]
        
    def standardize_column_names(self, df):
        """Standardize column names"""
        df.columns = df.columns.str.strip()
        
        rename_map = {
            'Flow ID': 'Flow_ID',
            'Src IP': 'Src_IP',
            'Dst IP': 'Dst_IP',
            'Src Port': 'Src_Port',
            'Dst Port': 'Dst_Port',
            ' Label': 'Label',
            'Label ': 'Label',
        }
        df.rename(columns=rename_map, inplace=True)
        df.columns = df.columns.str.replace(' ', '_')
        
        return df
    
    def clean_chunk(self, chunk):
        """Clean a single chunk of data"""
        # Standardize columns
        chunk = self.standardize_column_names(chunk)
        
        # Check for Label column
        if 'Label' not in chunk.columns:
            return None
        
        # Remove duplicates
        chunk = chunk.drop_duplicates()
        
        # Handle missing values MORE EFFICIENTLY
        # Instead of dropna with thresh (which causes memory issues),
        # just drop rows with more than 30% missing
        missing_threshold = int(len(chunk.columns) * 0.3)
        missing_counts = chunk.isnull().sum(axis=1)
        chunk = chunk[missing_counts <= missing_threshold]
        
        # Fill remaining NaN values
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if chunk[col].isnull().any():
                chunk[col].fillna(chunk[col].median(), inplace=True)
        
        categorical_cols = chunk.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Label' and chunk[col].isnull().any():
                mode_val = chunk[col].mode()
                chunk[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)
        
        # Replace infinity values
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in numeric_cols:
            if chunk[col].isnull().any():
                chunk[col].fillna(chunk[col].median(), inplace=True)
        
        # Remove invalid flows
        if 'Flow_Duration' in chunk.columns:
            chunk = chunk[chunk['Flow_Duration'] >= 0]
        
        # Handle timestamps
        if 'Timestamp' in chunk.columns:
            try:
                chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce')
                chunk = chunk.dropna(subset=['Timestamp'])
            except:
                pass
        
        # Clean and map labels
        chunk['Label'] = chunk['Label'].str.strip()
        chunk['Label_Numeric'] = chunk['Label'].map(self.label_mapping)
        chunk = chunk.dropna(subset=['Label_Numeric'])
        chunk['Label_Numeric'] = chunk['Label_Numeric'].astype(int)
        
        # Remove low-variance features
        features_found = [f for f in self.features_to_remove if f in chunk.columns]
        if features_found:
            chunk = chunk.drop(columns=features_found)
        
        return chunk
    
    def process_file_in_chunks(self, filepath):
        """Process file in chunks to avoid memory issues"""
        print(f"\n{'='*60}")
        print(f"Processing: {filepath.name}")
        print(f"{'='*60}")
        
        # First pass: get column names and row count
        print(f"Reading file structure...")
        first_chunk = pd.read_csv(filepath, nrows=1000)
        first_chunk = self.standardize_column_names(first_chunk)
        
        print(f"  Original columns: {list(first_chunk.columns[:10])}...")
        
        # Check for IP columns
        has_ips = 'Src_IP' in first_chunk.columns and 'Dst_IP' in first_chunk.columns
        if has_ips:
            print(f"  ✓ IP columns preserved: Src_IP, Dst_IP")
        else:
            print(f"  ⚠️ WARNING: IP columns not found after standardization!")
            print(f"  Available columns: {list(first_chunk.columns)}")
        
        # Process in chunks
        print(f"\n=== Processing in chunks of {self.chunk_size:,} rows ===")
        
        cleaned_chunks = []
        total_rows = 0
        removed_rows = 0
        chunk_num = 0
        
        # Statistics tracking
        label_counts = {}
        
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, low_memory=False):
            chunk_num += 1
            initial_size = len(chunk)
            total_rows += initial_size
            
            print(f"\rProcessing chunk {chunk_num} ({initial_size:,} rows)...", end='', flush=True)
            
            # Clean chunk
            cleaned_chunk = self.clean_chunk(chunk)
            
            if cleaned_chunk is not None and len(cleaned_chunk) > 0:
                # Track labels
                for label, count in cleaned_chunk['Label'].value_counts().items():
                    label_counts[label] = label_counts.get(label, 0) + count
                
                cleaned_chunks.append(cleaned_chunk)
                removed_rows += (initial_size - len(cleaned_chunk))
        
        print()  # New line after progress
        
        if not cleaned_chunks:
            print("  Error: No data after cleaning")
            return None
        
        # Combine all chunks
        print(f"\n=== Combining {len(cleaned_chunks)} chunks ===")
        df = pd.concat(cleaned_chunks, ignore_index=True)
        
        print(f"\n=== Cleaning Complete ===")
        print(f"Total rows processed: {total_rows:,}")
        print(f"Final shape: {df.shape}")
        print(f"Removed {removed_rows:,} rows ({removed_rows/total_rows*100:.1f}%)")
        
        # Print label distribution
        print(f"\n=== Label Distribution ===")
        print(f"\nOriginal Labels:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print(f"\nNumeric Labels (Grouped):")
        numeric_counts = df['Label_Numeric'].value_counts().sort_index()
        for num, count in numeric_counts.items():
            examples = [k for k, v in self.label_mapping.items() if v == num]
            group_name = examples[0] if examples else f"Class_{num}"
            print(f"  {num} ({group_name}): {count:,} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def handle_outliers(self, df):
        """Handle outliers using winsorization - optimized"""
        print("\n=== Handling Outliers ===")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ['Label_Numeric', 'Src_Port', 'Dst_Port', 'Protocol']
        cols_to_process = [c for c in numeric_cols if c not in exclude]
        
        # Process in batches to save memory
        batch_size = 10
        for i in range(0, len(cols_to_process), batch_size):
            batch_cols = cols_to_process[i:i+batch_size]
            for col in batch_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        print(f"✓ Winsorized {len(cols_to_process)} columns")
        return df
    
    def log_transform_skewed(self, df):
        """Apply log transformation to highly skewed features"""
        print("\n=== Log Transforming Skewed Features ===")
        
        skewed_features = [
            'Flow_Duration', 'Flow_IAT_Mean', 'Flow_IAT_Max', 'Flow_IAT_Min',
            'Fwd_IAT_Mean', 'Fwd_IAT_Max', 'Bwd_IAT_Mean', 'Bwd_IAT_Max',
            'Active_Mean', 'Active_Max', 'Idle_Mean', 'Idle_Max'
        ]
        
        transformed = 0
        for col in skewed_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])
                transformed += 1
        
        print(f"✓ Log transformed {transformed} skewed features")
        return df
    
    def save_cleaned_data(self, df, filename):
        """Save cleaned data based on mode"""
        print(f"\n=== Saving Cleaned Data ===")
        
        if self.mode == 'gnn' or self.mode == 'both':
            if all(col in df.columns for col in ['Src_IP', 'Dst_IP']):
                gnn_df = df.copy()
                if self.mode == 'both':
                    output_path = self.output_dir / 'for_gnn' / filename
                else:
                    output_path = self.output_dir / filename
                
                gnn_df.to_csv(output_path, index=False)
                print(f"✓ Saved GNN version: {output_path}")
                print(f"  Shape: {gnn_df.shape}")
            else:
                print("  Warning: IP columns not found, cannot save GNN version")
        
        if self.mode == 'cnn_lstm' or self.mode == 'both':
            cnn_lstm_df = df.copy()
            
            cols_to_remove = ['Src_IP', 'Dst_IP', 'Flow_ID']
            cols_to_remove = [c for c in cols_to_remove if c in cnn_lstm_df.columns]
            cnn_lstm_df = cnn_lstm_df.drop(columns=cols_to_remove)
            
            if self.mode == 'both':
                output_path = self.output_dir / 'for_cnn_lstm' / filename
            else:
                output_path = self.output_dir / filename
            
            cnn_lstm_df.to_csv(output_path, index=False)
            print(f"✓ Saved CNN/LSTM version: {output_path}")
            print(f"  Shape: {cnn_lstm_df.shape}")
            print(f"  Removed: {', '.join(cols_to_remove) if cols_to_remove else 'None'}")
    
    def process_file(self, filepath):
        """Process a single CSV file"""
        # Process in chunks
        df = self.process_file_in_chunks(filepath)
        if df is None or len(df) == 0:
            return
        
        # Handle outliers
        df = self.handle_outliers(df)
        
        # Log transform skewed features
        df = self.log_transform_skewed(df)
        
        # Save cleaned data
        output_filename = f"cleaned_{filepath.stem}.csv"
        self.save_cleaned_data(df, output_filename)
        
        print(f"\n✓ Successfully processed {filepath.name}")
    
    def process_all_files(self):
        """Process all CSV files in input directory"""
        csv_files = sorted(list(self.input_dir.glob('*.csv')))
        
        if not csv_files:
            print(f"No CSV files found in {self.input_dir}")
            return
        
        print(f"\nFound {len(csv_files)} CSV files to process")
        print(f"Mode: {self.mode}")
        print(f"Output directory: {self.output_dir}")
        print(f"Chunk size: {self.chunk_size:,} rows")
        
        for i, filepath in enumerate(csv_files, 1):
            print(f"\n{'#'*60}")
            print(f"File {i}/{len(csv_files)}")
            print(f"{'#'*60}")
            self.process_file(filepath)
        
        print(f"\n{'='*60}")
        print(f"ALL FILES PROCESSED SUCCESSFULLY!")
        print(f"{'='*60}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean CICIDS2018 Dataset (Memory Optimized)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save cleaned files')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['gnn', 'cnn_lstm', 'both'],
                        help='Processing mode')
    parser.add_argument('--chunk_size', type=int, default=500000,
                        help='Chunk size for processing (default: 500000)')
    
    args = parser.parse_args()
    
    cleaner = CICIDS2018Cleaner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        chunk_size=args.chunk_size
    )
    
    cleaner.process_all_files()


if __name__ == '__main__':
    main()