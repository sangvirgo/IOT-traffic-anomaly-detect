"""
CICIDS2018 Data Splitting Script
Performs temporal split for train/val/test sets with class balancing

Usage:
    python split_data.py --input_dir ./cleaned_data/for_cnn_lstm --output_dir ./split_data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle

class DataSplitter:
    def __init__(self, input_dir, output_dir, temporal_split=True):
        """
        Args:
            input_dir: Directory with cleaned CSV files
            output_dir: Directory to save split datasets
            temporal_split: If True, split by time; if False, random stratified split
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temporal_split = temporal_split
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'scalers').mkdir(exist_ok=True)
    
    def load_all_files(self):
        """Load and concatenate all cleaned CSV files"""
        print("Loading all cleaned files...")
        
        csv_files = sorted(self.input_dir.glob('cleaned_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No cleaned CSV files found in {self.input_dir}")
        
        dfs = []
        for filepath in csv_files:
            print(f"  Loading {filepath.name}...")
            df = pd.read_csv(filepath)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Loaded {len(csv_files)} files")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  Total columns: {len(combined_df.columns)}")
        
        return combined_df
    
    def temporal_split_data(self, df, train_ratio=0.7, val_ratio=0.15):
        """Split data by timestamp (temporal split)"""
        print("\n=== Temporal Split ===")
        
        if 'Timestamp' not in df.columns:
            print("  Warning: No Timestamp column, falling back to random split")
            return self.random_split_data(df, train_ratio, val_ratio)
        
        # Sort by timestamp
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        print(f"  Train: {len(train_df):,} rows ({len(train_df)/n*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} rows ({len(val_df)/n*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} rows ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def random_split_data(self, df, train_ratio=0.7, val_ratio=0.15):
        """Random stratified split"""
        print("\n=== Random Stratified Split ===")
        
        test_ratio = 1 - train_ratio - val_ratio
        
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_ratio + test_ratio),
            stratify=df['Label_Numeric'],
            random_state=42
        )
        
        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio/(val_ratio + test_ratio),
            stratify=temp_df['Label_Numeric'],
            random_state=42
        )
        
        print(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def print_label_distribution(self, train_df, val_df, test_df):
        """Print label distribution for all splits"""
        print("\n=== Label Distribution ===")
        
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"\n{name} Set:")
            counts = df['Label_Numeric'].value_counts().sort_index()
            for label, count in counts.items():
                print(f"  Class {label}: {count:,} ({count/len(df)*100:.1f}%)")
    
    def balance_classes(self, X, y, strategy='auto'):
        """Balance classes using SMOTE + Undersampling"""
        print("\n=== Balancing Classes ===")
        print(f"Before balancing:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count:,}")
        
        # Define resampling strategy
        # Undersample majority class to 50% of original
        # Oversample minority classes to match majority
        
        over = SMOTE(sampling_strategy='auto', random_state=42)
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
                print(f"  Class {label}: {count:,}")
            
            return X_resampled, y_resampled
        except Exception as e:
            print(f"  Warning: Could not balance classes: {e}")
            print(f"  Returning original data")
            return X, y
    
    def prepare_features(self, df):
        """Prepare features and labels"""
        # Separate features and labels
        exclude_cols = ['Label', 'Label_Numeric', 'Timestamp']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['Label_Numeric'].values
        
        # Keep timestamp if exists (for later reference)
        timestamp = df['Timestamp'].values if 'Timestamp' in df.columns else None
        
        return X, y, feature_cols, timestamp
    
    def normalize_features(self, X_train, X_val, X_test, feature_names):
        """Normalize features using StandardScaler"""
        from sklearn.preprocessing import StandardScaler
        
        print("\n=== Normalizing Features ===")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = self.output_dir / 'scalers' / 'feature_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved scaler to {scaler_path}")
        
        # Save feature names
        feature_path = self.output_dir / 'scalers' / 'feature_names.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"✓ Saved feature names to {feature_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def save_splits(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Save train/val/test splits"""
        print("\n=== Saving Splits ===")
        
        # Save as numpy arrays
        np.save(self.output_dir / 'processed' / 'X_train.npy', X_train)
        np.save(self.output_dir / 'processed' / 'y_train.npy', y_train)
        np.save(self.output_dir / 'processed' / 'X_val.npy', X_val)
        np.save(self.output_dir / 'processed' / 'y_val.npy', y_val)
        np.save(self.output_dir / 'processed' / 'X_test.npy', X_test)
        np.save(self.output_dir / 'processed' / 'y_test.npy', y_test)
        
        print(f"✓ Saved training data:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"✓ Saved validation data:")
        print(f"  X_val: {X_val.shape}")
        print(f"  y_val: {y_val.shape}")
        print(f"✓ Saved test data:")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Save metadata
        metadata = {
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'temporal_split': self.temporal_split
        }
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"\n✓ Saved metadata to {self.output_dir / 'metadata.pkl'}")
    
    def process(self, balance=True, normalize=True):
        """Main processing pipeline"""
        print("="*60)
        print("CICIDS2018 Data Splitting Pipeline")
        print("="*60)
        
        # 1. Load all files
        df = self.load_all_files()
        
        # 2. Split data
        if self.temporal_split:
            train_df, val_df, test_df = self.temporal_split_data(df)
        else:
            train_df, val_df, test_df = self.random_split_data(df)
        
        # 3. Print label distribution
        self.print_label_distribution(train_df, val_df, test_df)
        
        # 4. Prepare features
        X_train, y_train, feature_names, _ = self.prepare_features(train_df)
        X_val, y_val, _, _ = self.prepare_features(val_df)
        X_test, y_test, _, _ = self.prepare_features(test_df)
        
        # 5. Balance classes (only on training set)
        if balance:
            X_train, y_train = self.balance_classes(X_train, y_train)
        
        # 6. Normalize features
        if normalize:
            X_train, X_val, X_test, scaler = self.normalize_features(
                X_train, X_val, X_test, feature_names
            )
        
        # 7. Save splits
        self.save_splits(X_train, y_train, X_val, y_val, X_test, y_test)
        
        print("\n" + "="*60)
        print("DATA SPLITTING COMPLETE!")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nNext steps:")
        print("  1. Load data: np.load('X_train.npy'), np.load('y_train.npy')")
        print("  2. Train your models (CNN/LSTM/GNN)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Split CICIDS2018 cleaned data')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing cleaned CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save split data')
    parser.add_argument('--temporal', action='store_true',
                        help='Use temporal split (default: random stratified)')
    parser.add_argument('--no_balance', action='store_true',
                        help='Skip class balancing')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip feature normalization')
    
    args = parser.parse_args()
    
    splitter = DataSplitter(
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