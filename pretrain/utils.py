"""
CICIDS2018 Utilities and Data Inspection Tools

Usage:
    python utils.py --inspect --data_dir ./cleaned_data/for_cnn_lstm
    python utils.py --visualize --data_dir ./split_data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

class DataInspector:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def inspect_cleaned_files(self):
        """Inspect cleaned CSV files"""
        print("="*60)
        print("CLEANED FILES INSPECTION")
        print("="*60)
        
        csv_files = sorted(self.data_dir.glob('cleaned_*.csv'))
        
        if not csv_files:
            print(f"No cleaned files found in {self.data_dir}")
            return
        
        total_rows = 0
        total_attacks = 0
        
        for filepath in csv_files:
            print(f"\n{filepath.name}")
            print("-"*40)
            
            df = pd.read_csv(filepath, nrows=1000)  # Quick peek
            
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {len(df.columns)}")
            
            if 'Label_Numeric' in df.columns:
                label_counts = df['Label_Numeric'].value_counts()
                print(f"  Labels:")
                for label, count in label_counts.items():
                    print(f"    {label}: {count}")
            
            # Get actual row count
            df_full = pd.read_csv(filepath)
            total_rows += len(df_full)
            
            if 'Label_Numeric' in df_full.columns:
                attacks = (df_full['Label_Numeric'] != 0).sum()
                total_attacks += attacks
        
        print("\n" + "="*60)
        print(f"SUMMARY:")
        print(f"  Total files: {len(csv_files)}")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Total attacks: {total_attacks:,} ({total_attacks/total_rows*100:.1f}%)")
        print(f"  Total benign: {total_rows - total_attacks:,} ({(total_rows - total_attacks)/total_rows*100:.1f}%)")
    
    def inspect_split_data(self):
        """Inspect split numpy arrays"""
        print("="*60)
        print("SPLIT DATA INSPECTION")
        print("="*60)
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        # Load arrays
        processed_dir = self.data_dir / 'processed'
        
        arrays = {
            'X_train': processed_dir / 'X_train.npy',
            'y_train': processed_dir / 'y_train.npy',
            'X_val': processed_dir / 'X_val.npy',
            'y_val': processed_dir / 'y_val.npy',
            'X_test': processed_dir / 'X_test.npy',
            'y_test': processed_dir / 'y_test.npy',
        }
        
        print("\n" + "-"*60)
        for name, path in arrays.items():
            if path.exists():
                arr = np.load(path)
                print(f"{name}:")
                print(f"  Shape: {arr.shape}")
                print(f"  Dtype: {arr.dtype}")
                print(f"  Memory: {arr.nbytes / (1024**2):.2f} MB")
                
                if 'y_' in name:
                    unique, counts = np.unique(arr, return_counts=True)
                    print(f"  Classes: {len(unique)}")
                    for u, c in zip(unique, counts):
                        print(f"    Class {u}: {c:,} ({c/len(arr)*100:.1f}%)")
                print()
    
    def visualize_label_distribution(self):
        """Visualize label distribution"""
        print("\nGenerating label distribution plot...")
        
        processed_dir = self.data_dir / 'processed'
        
        # Load labels
        y_train = np.load(processed_dir / 'y_train.npy')
        y_val = np.load(processed_dir / 'y_val.npy')
        y_test = np.load(processed_dir / 'y_test.npy')
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for ax, y, title in zip(axes, [y_train, y_val, y_test], ['Train', 'Val', 'Test']):
            unique, counts = np.unique(y, return_counts=True)
            ax.bar(unique, counts)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title(f'{title} Set (n={len(y):,})')
            ax.set_xticks(unique)
        
        plt.tight_layout()
        output_path = self.data_dir / 'label_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {output_path}")
        plt.close()
    
    def check_feature_statistics(self):
        """Check basic feature statistics"""
        print("\n" + "="*60)
        print("FEATURE STATISTICS")
        print("="*60)
        
        # Load feature names
        scaler_dir = self.data_dir / 'scalers'
        feature_path = scaler_dir / 'feature_names.pkl'
        
        if not feature_path.exists():
            print("Feature names not found")
            return
        
        with open(feature_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        print(f"\nTotal features: {len(feature_names)}")
        print("\nFirst 10 features:")
        for i, name in enumerate(feature_names[:10], 1):
            print(f"  {i}. {name}")
        
        # Load data
        processed_dir = self.data_dir / 'processed'
        X_train = np.load(processed_dir / 'X_train.npy')
        
        print(f"\nFeature statistics (training set):")
        print(f"  Shape: {X_train.shape}")
        print(f"  Mean: {X_train.mean():.4f}")
        print(f"  Std: {X_train.std():.4f}")
        print(f"  Min: {X_train.min():.4f}")
        print(f"  Max: {X_train.max():.4f}")
        
        # Check for NaN or Inf
        has_nan = np.isnan(X_train).any()
        has_inf = np.isinf(X_train).any()
        print(f"\n  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")


class DataLoader:
    """Helper class to load split data for training"""
    
    @staticmethod
    def load_data(data_dir):
        """Load train/val/test data"""
        data_dir = Path(data_dir)
        processed_dir = data_dir / 'processed'
        
        X_train = np.load(processed_dir / 'X_train.npy')
        y_train = np.load(processed_dir / 'y_train.npy')
        X_val = np.load(processed_dir / 'X_val.npy')
        y_val = np.load(processed_dir / 'y_val.npy')
        X_test = np.load(processed_dir / 'X_test.npy')
        y_test = np.load(processed_dir / 'y_test.npy')
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    @staticmethod
    def load_metadata(data_dir):
        """Load metadata"""
        metadata_path = Path(data_dir) / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_scaler(data_dir):
        """Load feature scaler"""
        scaler_path = Path(data_dir) / 'scalers' / 'feature_scaler.pkl'
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_feature_names(data_dir):
        """Load feature names"""
        feature_path = Path(data_dir) / 'scalers' / 'feature_names.pkl'
        with open(feature_path, 'rb') as f:
            return pickle.load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CICIDS2018 Data Utilities')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory to inspect')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect data files')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--statistics', action='store_true',
                        help='Show feature statistics')
    
    args = parser.parse_args()
    
    inspector = DataInspector(args.data_dir)
    
    if args.inspect:
        # Try to detect if it's cleaned or split data
        data_dir = Path(args.data_dir)
        if (data_dir / 'processed').exists():
            inspector.inspect_split_data()
        else:
            inspector.inspect_cleaned_files()
    
    if args.visualize:
        inspector.visualize_label_distribution()
    
    if args.statistics:
        inspector.check_feature_statistics()
    
    if not (args.inspect or args.visualize or args.statistics):
        print("Please specify at least one action: --inspect, --visualize, or --statistics")


if __name__ == '__main__':
    main()