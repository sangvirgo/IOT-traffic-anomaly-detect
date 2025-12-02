"""
CICIDS2018 Data Splitting Script - Ultra Memory Optimized
Performs temporal split for train/val/test sets with class balancing
Optimized for 16GB RAM systems

Usage:
    python split_data_optimized.py --input_dir ./cleaned_data/for_cnn_lstm --output_dir ./split_data --temporal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')


class MemoryEfficientDataSplitter:
    def __init__(self, input_dir, output_dir, temporal_split=True):
        """
        Args:
            input_dir: Directory with cleaned CSV files
            output_dir: Directory to save split datasets
            temporal_split: If True, split by file order; if False, random split
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temporal_split = temporal_split
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'scalers').mkdir(exist_ok=True)
        
        # For tracking feature stats
        self.feature_names = None
        self.scaler = None
    
    def scan_files(self):
        """Scan files to get structure and sizes"""
        print("="*70)
        print("SCANNING FILES")
        print("="*70)
        
        csv_files = sorted(self.input_dir.glob('cleaned_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No cleaned CSV files found in {self.input_dir}")
        
        file_info = []
        total_rows = 0
        
        for filepath in csv_files:
            # Count rows efficiently
            with open(filepath, 'r') as f:
                row_count = sum(1 for _ in f) - 1  # -1 for header
            
            # Get column info from first row
            sample = pd.read_csv(filepath, nrows=1)
            
            file_info.append({
                'path': filepath,
                'name': filepath.name,
                'rows': row_count,
                'columns': sample.columns.tolist()
            })
            
            total_rows += row_count
            print(f"  ✓ {filepath.name}: {row_count:,} rows")
        
        print(f"\n✓ Total: {len(csv_files)} files, {total_rows:,} rows")
        return file_info, total_rows
    
    def determine_splits(self, file_info, train_ratio=0.7, val_ratio=0.15):
        """Determine which files go to which split"""
        print("\n" + "="*70)
        print("DETERMINING SPLITS (FILE-BASED)")
        print("="*70)
        
        total_rows = sum(f['rows'] for f in file_info)
        train_size = int(total_rows * train_ratio)
        val_size = int(total_rows * val_ratio)
        
        cumulative = 0
        train_files = []
        val_files = []
        test_files = []
        
        for f in file_info:
            if cumulative < train_size:
                train_files.append(f)
            elif cumulative < train_size + val_size:
                val_files.append(f)
            else:
                test_files.append(f)
            cumulative += f['rows']
        
        print(f"\nTrain files ({len(train_files)}):")
        for f in train_files:
            print(f"  - {f['name']}: {f['rows']:,} rows")
        
        print(f"\nValidation files ({len(val_files)}):")
        for f in val_files:
            print(f"  - {f['name']}: {f['rows']:,} rows")
        
        print(f"\nTest files ({len(test_files)}):")
        for f in test_files:
            print(f"  - {f['name']}: {f['rows']:,} rows")
        
        return train_files, val_files, test_files
    
    def extract_features_labels(self, df):
        """Extract features and labels from dataframe"""
        # Columns to exclude (including ports that might be inconsistent)
        exclude_cols = ['Label', 'Label_Numeric', 'Timestamp', 'Src_IP', 'Dst_IP', 
                       'Flow_ID', 'Src_Port', 'Dst_Port']
        exclude_cols = [c for c in exclude_cols if c in df.columns]
        
        # Get feature columns (excluding ports for consistency)
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # If this is the first chunk, save feature names
        if self.feature_names is None:
            self.feature_names = sorted(feature_cols)  # Sort for consistency
            print(f"\n  ✓ Standard feature set: {len(self.feature_names)} features")
            print(f"  ✓ Excluded: Label, Label_Numeric, Timestamp, Ports")
        else:
            # Ensure this chunk has the same features
            current_features = sorted(feature_cols)
            if current_features != self.feature_names:
                print(f"\n  ⚠️  Feature mismatch detected - auto-fixing...")
                print(f"     Expected: {len(self.feature_names)} features")
                print(f"     Got: {len(current_features)} features")
                
                # Find differences
                missing = set(self.feature_names) - set(current_features)
                extra = set(current_features) - set(self.feature_names)
                
                if missing:
                    print(f"     Adding missing: {missing}")
                    # Add missing columns with zeros
                    for col in missing:
                        df[col] = 0.0
                
                if extra:
                    print(f"     Removing extra: {extra}")
                    # Remove extra columns
                    df = df.drop(columns=list(extra))
        
        # Extract in the correct order (sorted)
        X = df[self.feature_names].values.astype(np.float32)
        y = df['Label_Numeric'].values.astype(np.int32)
        
        return X, y, self.feature_names
    
    def process_files_to_arrays(self, file_list, split_name, chunk_size=100000):
        """Process files directly to numpy arrays without loading all into memory"""
        print(f"\n{'='*70}")
        print(f"PROCESSING {split_name.upper()} SET")
        print(f"{'='*70}")
        
        # Initialize collectors
        X_chunks = []
        y_chunks = []
        total_processed = 0
        
        for file_info in file_list:
            print(f"\n  Processing {file_info['name']}...")
            
            # Process file in chunks
            chunk_num = 0
            for chunk in pd.read_csv(file_info['path'], chunksize=chunk_size, low_memory=False):
                chunk_num += 1
                
                # Extract features and labels (handles feature alignment)
                X_chunk, y_chunk, _ = self.extract_features_labels(chunk)
                
                X_chunks.append(X_chunk)
                y_chunks.append(y_chunk)
                total_processed += len(chunk)
                
                print(f"    Chunk {chunk_num}: {len(chunk):,} rows (Total: {total_processed:,})", end='\r')
                
                # Combine chunks if accumulated too many
                if len(X_chunks) >= 5:
                    print(f"\n    Combining {len(X_chunks)} chunks...")
                    X_chunks = [np.vstack(X_chunks)]
                    y_chunks = [np.concatenate(y_chunks)]
                    gc.collect()
                    print(f"    Combined shape: {X_chunks[0].shape}")
                    print(f"    Continuing...", end='')
            
            print()  # New line after file processing
        
        # Final combination
        print(f"\n  Final combination for {split_name}...")
        X = np.vstack(X_chunks)
        y = np.concatenate(y_chunks)
        
        # Clear chunks from memory
        del X_chunks, y_chunks
        gc.collect()
        
        print(f"✓ {split_name} set: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        # Print label distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n  Label distribution:")
        for label, count in zip(unique, counts):
            print(f"    Class {label}: {count:,} ({count/len(y)*100:.1f}%)")
        
        return X, y
    
    def balance_classes(self, X, y):
        """Balance classes using SMOTE + Undersampling"""
        print("\n" + "="*70)
        print("BALANCING CLASSES (Training Set Only)")
        print("="*70)
        
        print(f"\nBefore balancing:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count:,} ({count/len(y)*100:.1f}%)")
        
        # Check if we have enough samples per class for SMOTE
        min_samples = counts.min()
        if min_samples < 6:
            print(f"\n⚠️  Warning: Class with only {min_samples} samples detected!")
            print(f"  SMOTE requires at least 6 samples per class.")
            print(f"  Skipping class balancing...")
            return X, y
        
        # Use smaller k_neighbors if needed
        k = min(5, min_samples - 1)
        
        print(f"\nApplying SMOTE (k_neighbors={k}) + Random Undersampling...")
        
        over = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k)
        under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        
        pipeline = Pipeline([
            ('over', over),
            ('under', under)
        ])
        
        try:
            X_resampled, y_resampled = pipeline.fit_resample(X, y)
            
            print(f"\nAfter balancing:")
            unique, counts = np.unique(y_resampled, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  Class {label}: {count:,} ({count/len(y_resampled)*100:.1f}%)")
            
            print(f"\n✓ Balanced from {len(y):,} to {len(y_resampled):,} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"\n⚠️  Warning: Could not balance classes: {e}")
            print(f"  Returning original data")
            return X, y
    
    def fit_scaler(self, X_train, batch_size=100000):
        """Fit scaler on training data in batches to save memory"""
        print("\n" + "="*70)
        print("FITTING SCALER ON TRAINING DATA")
        print("="*70)
        
        self.scaler = StandardScaler()
        
        # Fit in batches
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"  Processing {n_batches} batches...")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch = X_train[start_idx:end_idx]
            
            if i == 0:
                self.scaler.fit(batch)
            else:
                # Update scaler statistics
                self.scaler.partial_fit(batch)
            
            print(f"    Batch {i+1}/{n_batches} processed", end='\r')
        
        print(f"\n✓ Scaler fitted on {n_samples:,} samples")
        
        return self.scaler
    
    def transform_data(self, X, name, batch_size=100000):
        """Transform data in batches"""
        print(f"\n  Transforming {name} data...")
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        X_transformed = np.zeros_like(X)
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            X_transformed[start_idx:end_idx] = self.scaler.transform(X[start_idx:end_idx])
            
            print(f"    Batch {i+1}/{n_batches} transformed", end='\r')
        
        print(f"\n  ✓ Transformed {n_samples:,} samples")
        
        return X_transformed
    
    def normalize_all_splits(self, X_train, X_val, X_test):
        """Normalize all splits"""
        print("\n" + "="*70)
        print("NORMALIZING FEATURES")
        print("="*70)
        
        # Fit on training data
        self.fit_scaler(X_train)
        
        # Transform all splits
        X_train_scaled = self.transform_data(X_train, "train")
        X_val_scaled = self.transform_data(X_val, "validation")
        X_test_scaled = self.transform_data(X_test, "test")
        
        # Save scaler and feature names
        print("\n  Saving scaler and feature names...")
        with open(self.output_dir / 'scalers' / 'feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(self.output_dir / 'scalers' / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"  ✓ Saved scaler and feature names")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_splits(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Save train/val/test splits"""
        print("\n" + "="*70)
        print("SAVING SPLITS")
        print("="*70)
        
        processed_dir = self.output_dir / 'processed'
        
        print("  Saving training data...")
        np.save(processed_dir / 'X_train.npy', X_train)
        np.save(processed_dir / 'y_train.npy', y_train)
        
        print("  Saving validation data...")
        np.save(processed_dir / 'X_val.npy', X_val)
        np.save(processed_dir / 'y_val.npy', y_val)
        
        print("  Saving test data...")
        np.save(processed_dir / 'X_test.npy', X_test)
        np.save(processed_dir / 'y_test.npy', y_test)
        
        # Save metadata
        metadata = {
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'temporal_split': self.temporal_split,
            'feature_names': self.feature_names
        }
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\n" + "="*70)
        print("✓ ALL DATA SAVED SUCCESSFULLY")
        print("="*70)
        print(f"\nTrain:      X={X_train.shape}, y={y_train.shape}")
        print(f"Validation: X={X_val.shape}, y={y_val.shape}")
        print(f"Test:       X={X_test.shape}, y={y_test.shape}")
        print(f"\nOutput directory: {self.output_dir}")
    
    def process(self, balance=True, normalize=True):
        """Main processing pipeline"""
        print("="*70)
        print("CICIDS2018 DATA SPLITTING - ULTRA MEMORY OPTIMIZED")
        print("Mode: CNN/LSTM (No IP addresses needed)")
        print("="*70)
        
        # 1. Scan files
        file_info, total_rows = self.scan_files()
        
        # 2. Determine splits
        if self.temporal_split:
            train_files, val_files, test_files = self.determine_splits(file_info)
        else:
            # For random split, we'd need to load all data
            # Not recommended for 16GB RAM
            raise NotImplementedError("Random split requires more memory. Use --temporal flag.")
        
        # 3. Process each split separately
        X_train, y_train = self.process_files_to_arrays(train_files, "train")
        gc.collect()
        
        X_val, y_val = self.process_files_to_arrays(val_files, "validation")
        gc.collect()
        
        X_test, y_test = self.process_files_to_arrays(test_files, "test")
        gc.collect()
        
        # 4. Balance classes (only training set)
        if balance:
            X_train, y_train = self.balance_classes(X_train, y_train)
            gc.collect()
        
        # 5. Normalize features
        if normalize:
            X_train, X_val, X_test = self.normalize_all_splits(X_train, X_val, X_test)
            gc.collect()
        
        # 6. Save everything
        self.save_splits(X_train, y_train, X_val, y_val, X_test, y_test)
        
        print("\n" + "="*70)
        print("✓ DATA PREPROCESSING COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Load data using utils.py DataLoader")
        print("  2. Train CNN/LSTM models")
        print("  3. Evaluate and compare results")
        print("\nMemory tips for training:")
        print("  - Use batch_size=32 or 64")
        print("  - Use float32 instead of float64")
        print("  - Clear session between runs")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split CICIDS2018 data - Optimized for 16GB RAM'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with cleaned CSV files (for_cnn_lstm)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save split data')
    parser.add_argument('--temporal', action='store_true',
                        help='Use temporal split (REQUIRED for memory efficiency)')
    parser.add_argument('--no_balance', action='store_true',
                        help='Skip class balancing')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip feature normalization')
    
    args = parser.parse_args()
    
    if not args.temporal:
        print("⚠️  WARNING: --temporal flag is REQUIRED for 16GB RAM systems!")
        print("Random split requires loading all data into memory.")
        response = input("Continue with temporal split? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
        args.temporal = True
    
    splitter = MemoryEfficientDataSplitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        temporal_split=args.temporal
    )
    
    splitter.process(
        balance=not args.no_balance,
        normalize=not args.no_normalize
    )


if __name__ == '__main__':
    main()