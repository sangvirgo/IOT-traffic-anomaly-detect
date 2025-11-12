"""
CICIDS2018 Dataset Cleaning Pipeline
Supports both GNN (with IPs) and CNN/LSTM (without IPs)

Usage:
    python clean_cicids.py --input_dir ./raw_data --output_dir ./cleaned_data --mode gnn
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CICIDS2018Cleaner:
    def __init__(self, input_dir, output_dir, mode='both'):
        """
        Args:
            input_dir: Path to raw CSV files
            output_dir: Path to save cleaned files
            mode: 'gnn' (keep IPs), 'cnn_lstm' (remove IPs), 'both' (save both versions)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode
        
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
        }
        
        # Features to remove (low variance or redundant)
        self.features_to_remove = [
            'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
            'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
            'Fwd Header Len.1',  # Duplicate if exists
        ]
        
        # Columns to keep for GNN (including IP addresses)
        self.gnn_keep_columns = [
            'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol'
        ]
        
    def load_csv_chunked(self, filepath, chunksize=100000):
        """Load large CSV in chunks"""
        print(f"Loading {filepath.name}...")
        chunks = []
        try:
            for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            print(f"  Loaded {len(df):,} rows")
            return df
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            return None
    
    def standardize_column_names(self, df):
        """Standardize column names"""
        # Remove leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Common column name variations
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
        
        # Remove any remaining spaces in column names
        df.columns = df.columns.str.replace(' ', '_')
        
        return df
    
    def clean_data(self, df):
        """Main cleaning function"""
        print("\n=== Starting Data Cleaning ===")
        initial_rows = len(df)
        
        # 1. Standardize column names
        df = self.standardize_column_names(df)
        print(f"✓ Column names standardized")
        
        # 2. Check if Label column exists
        if 'Label' not in df.columns:
            print("  Warning: 'Label' column not found!")
            return None
        
        # 3. Remove duplicates
        df = df.drop_duplicates()
        print(f"✓ Removed {initial_rows - len(df):,} duplicates")
        
        # 4. Handle missing values
        missing_before = df.isnull().sum().sum()
        
        # Remove rows with too many missing values (>30% columns)
        threshold = len(df.columns) * 0.3
        df = df.dropna(thresh=len(df.columns) - threshold)
        
        # Fill remaining NaN with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Label':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        print(f"✓ Handled {missing_before:,} missing values")
        
        # 5. Replace infinity values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(f"✓ Replaced infinity values")
        
        # 6. Remove invalid flows
        if 'Flow_Duration' in df.columns:
            invalid = df['Flow_Duration'] < 0
            df = df[~invalid]
            print(f"✓ Removed {invalid.sum():,} invalid flows (negative duration)")
        
        # 7. Handle timestamps
        if 'Timestamp' in df.columns:
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df = df.dropna(subset=['Timestamp'])
                print(f"✓ Parsed timestamps")
            except:
                print("  Warning: Could not parse timestamps")
        
        # 8. Clean and map labels
        df['Label'] = df['Label'].str.strip()
        
        # Map labels to numeric
        df['Label_Numeric'] = df['Label'].map(self.label_mapping)
        
        # Handle unmapped labels
        unmapped = df['Label_Numeric'].isnull()
        if unmapped.any():
            print(f"  Warning: {unmapped.sum():,} unmapped labels found")
            unique_unmapped = df[unmapped]['Label'].unique()
            print(f"  Unmapped labels: {unique_unmapped}")
            # Map unmapped to a new category or drop
            df = df.dropna(subset=['Label_Numeric'])
        
        df['Label_Numeric'] = df['Label_Numeric'].astype(int)
        print(f"✓ Mapped labels to numeric")
        
        # 9. Remove low-variance features
        features_found = [f for f in self.features_to_remove if f in df.columns]
        if features_found:
            df = df.drop(columns=features_found)
            print(f"✓ Removed {len(features_found)} low-variance features")
        
        print(f"\n=== Cleaning Complete ===")
        print(f"Final shape: {df.shape}")
        print(f"Removed {initial_rows - len(df):,} rows ({(initial_rows - len(df))/initial_rows*100:.1f}%)")
        
        return df
    
    def handle_outliers(self, df):
        """Handle outliers using winsorization"""
        print("\n=== Handling Outliers ===")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude label and ID columns
        exclude = ['Label_Numeric', 'Src_Port', 'Dst_Port', 'Protocol']
        cols_to_process = [c for c in numeric_cols if c not in exclude]
        
        for col in cols_to_process:
            # Winsorize at 1st and 99th percentile
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
                # Add 1 to handle zeros, then log
                df[col] = np.log1p(df[col])
                transformed += 1
        
        print(f"✓ Log transformed {transformed} skewed features")
        return df
    
    def save_cleaned_data(self, df, filename):
        """Save cleaned data based on mode"""
        print(f"\n=== Saving Cleaned Data ===")
        
        if self.mode == 'gnn' or self.mode == 'both':
            # Save with IP addresses for GNN
            if all(col in df.columns for col in ['Src_IP', 'Dst_IP']):
                gnn_df = df.copy()
                if self.mode == 'both':
                    output_path = self.output_dir / 'for_gnn' / filename
                else:
                    output_path = self.output_dir / filename
                
                gnn_df.to_csv(output_path, index=False)
                print(f"✓ Saved GNN version: {output_path}")
                print(f"  Shape: {gnn_df.shape}")
                print(f"  Includes IP addresses: Src_IP, Dst_IP")
            else:
                print("  Warning: IP columns not found, cannot save GNN version")
        
        if self.mode == 'cnn_lstm' or self.mode == 'both':
            # Save without IP addresses for CNN/LSTM
            cnn_lstm_df = df.copy()
            
            # Remove IP and identifier columns
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
    
    def print_label_distribution(self, df):
        """Print label distribution"""
        print("\n=== Label Distribution ===")
        
        # Original labels
        print("\nOriginal Labels:")
        label_counts = df['Label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Numeric labels
        print("\nNumeric Labels:")
        numeric_counts = df['Label_Numeric'].value_counts().sort_index()
        label_names = {v: k for k, v in self.label_mapping.items()}
        for num, count in numeric_counts.items():
            examples = [k for k, v in self.label_mapping.items() if v == num]
            print(f"  {num} ({examples[0]}): {count:,} ({count/len(df)*100:.1f}%)")
    
    def process_file(self, filepath):
        """Process a single CSV file"""
        print(f"\n{'='*60}")
        print(f"Processing: {filepath.name}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_csv_chunked(filepath)
        if df is None:
            return
        
        # Clean data
        df = self.clean_data(df)
        if df is None or len(df) == 0:
            print("  Error: No data after cleaning")
            return
        
        # Print label distribution
        self.print_label_distribution(df)
        
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
        csv_files = list(self.input_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {self.input_dir}")
            return
        
        print(f"\nFound {len(csv_files)} CSV files to process")
        print(f"Mode: {self.mode}")
        print(f"Output directory: {self.output_dir}")
        
        for i, filepath in enumerate(csv_files, 1):
            print(f"\n{'#'*60}")
            print(f"File {i}/{len(csv_files)}")
            print(f"{'#'*60}")
            self.process_file(filepath)
        
        print(f"\n{'='*60}")
        print(f"ALL FILES PROCESSED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"\nCleaned files saved to: {self.output_dir}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of cleaned files"""
        print("\n=== Summary ===")
        
        if self.mode == 'gnn' or self.mode == 'both':
            gnn_dir = self.output_dir / 'for_gnn' if self.mode == 'both' else self.output_dir
            gnn_files = list(gnn_dir.glob('cleaned_*.csv'))
            print(f"\nGNN files: {len(gnn_files)}")
            for f in gnn_files:
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  {f.name}: {size_mb:.1f} MB")
        
        if self.mode == 'cnn_lstm' or self.mode == 'both':
            cnn_dir = self.output_dir / 'for_cnn_lstm' if self.mode == 'both' else self.output_dir
            cnn_files = list(cnn_dir.glob('cleaned_*.csv'))
            print(f"\nCNN/LSTM files: {len(cnn_files)}")
            for f in cnn_files:
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  {f.name}: {size_mb:.1f} MB")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean CICIDS2018 Dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing raw CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save cleaned files')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['gnn', 'cnn_lstm', 'both'],
                        help='Processing mode: gnn (keep IPs), cnn_lstm (remove IPs), both (save both)')
    parser.add_argument('--chunksize', type=int, default=100000,
                        help='Chunk size for reading large files')
    
    args = parser.parse_args()
    
    # Create cleaner instance
    cleaner = CICIDS2018Cleaner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode
    )
    
    # Process all files
    cleaner.process_all_files()


if __name__ == '__main__':
    main()