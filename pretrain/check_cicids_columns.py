"""
CICIDS2018 Column Checker
Check if dataset has all required columns for GNN/CNN/LSTM

Usage:
    python check_cicids_columns.py --input_dir ./raw_data
"""

import pandas as pd
from pathlib import Path
import sys

class ColumnChecker:
    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)
        
        # Required columns for different models
        self.gnn_required = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port']
        self.cnn_lstm_required = [
            'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
            'Protocol', 'Label'
        ]
        
    def check_file(self, filepath):
        """Check a single CSV file"""
        print(f"\n{'='*60}")
        print(f"Checking: {filepath.name}")
        print(f"{'='*60}")
        
        try:
            # Read first few rows to check columns
            df = pd.read_csv(filepath, nrows=5)
            
            print(f"✓ File loaded successfully")
            print(f"  Total columns: {len(df.columns)}")
            print(f"  Total rows (sample): {len(df)}")
            
            # Print all columns
            print(f"\n📋 All columns found:")
            for i, col in enumerate(df.columns, 1):
                print(f"  {i:2d}. {col}")
            
            # Check for GNN requirements
            print(f"\n🔍 GNN Requirements Check:")
            gnn_ok = True
            for col in self.gnn_required:
                variations = [col, col.replace(' ', '_'), col.replace(' ', '')]
                found = any(v in df.columns for v in variations)
                status = "✓" if found else "❌"
                print(f"  {status} {col}: {'FOUND' if found else 'MISSING'}")
                if not found:
                    gnn_ok = False
            
            if gnn_ok:
                print(f"\n✅ This dataset CAN be used for GNN (has IP columns)")
            else:
                print(f"\n❌ This dataset CANNOT be used for GNN (missing IP columns)")
                print(f"   → Only CNN/LSTM models can be trained")
            
            # Check for CNN/LSTM requirements
            print(f"\n🔍 CNN/LSTM Requirements Check:")
            cnn_lstm_ok = True
            for col in self.cnn_lstm_required:
                variations = [col, col.replace(' ', '_'), col.replace(' ', '')]
                found = any(v in df.columns for v in variations)
                status = "✓" if found else "❌"
                print(f"  {status} {col}: {'FOUND' if found else 'MISSING'}")
                if not found:
                    cnn_lstm_ok = False
            
            if cnn_lstm_ok:
                print(f"\n✅ This dataset CAN be used for CNN/LSTM")
            else:
                print(f"\n❌ This dataset is INCOMPLETE")
            
            # Check Label column
            print(f"\n🏷️  Label Column Check:")
            if 'Label' in df.columns:
                unique_labels = df['Label'].unique()
                print(f"  ✓ Label column found")
                print(f"  Unique labels (sample): {unique_labels}")
            else:
                print(f"  ❌ Label column NOT found")
            
            # Sample data
            print(f"\n📊 Sample Data (first row):")
            print(df.head(1).T.to_string())
            
            return gnn_ok, cnn_lstm_ok
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False, False
    
    def check_all_files(self):
        """Check all CSV files"""
        csv_files = sorted(list(self.input_dir.glob('*.csv')))
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.input_dir}")
            return
        
        print(f"\n{'#'*60}")
        print(f"CICIDS2018 DATASET COLUMN CHECKER")
        print(f"{'#'*60}")
        print(f"\nFound {len(csv_files)} CSV files")
        
        results = []
        for filepath in csv_files:
            gnn_ok, cnn_lstm_ok = self.check_file(filepath)
            results.append({
                'file': filepath.name,
                'gnn': gnn_ok,
                'cnn_lstm': cnn_lstm_ok
            })
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        
        gnn_count = sum(1 for r in results if r['gnn'])
        cnn_lstm_count = sum(1 for r in results if r['cnn_lstm'])
        
        print(f"\n📊 Results:")
        print(f"  Total files checked: {len(results)}")
        print(f"  Files with GNN support (has IPs): {gnn_count}/{len(results)}")
        print(f"  Files with CNN/LSTM support: {cnn_lstm_count}/{len(results)}")
        
        if gnn_count == 0:
            print(f"\n⚠️  WARNING: NO FILES HAVE IP ADDRESSES!")
            print(f"   This means you CANNOT train GNN models.")
            print(f"   ")
            print(f"   Possible solutions:")
            print(f"   1. Download the CORRECT CICIDS 2018 dataset from:")
            print(f"      https://www.unb.ca/cic/datasets/ids-2018.html")
            print(f"   2. Use only CNN/LSTM models for this project")
            print(f"   3. Generate synthetic IP addresses (not recommended)")
        
        print(f"\n📝 Detailed Results:")
        for r in results:
            gnn_status = "✓ GNN" if r['gnn'] else "✗ GNN"
            cnn_status = "✓ CNN/LSTM" if r['cnn_lstm'] else "✗ CNN/LSTM"
            print(f"  {r['file']:30s} → {gnn_status:10s} {cnn_status}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check CICIDS2018 columns for GNN/CNN/LSTM compatibility')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing CSV files')
    
    args = parser.parse_args()
    
    checker = ColumnChecker(args.input_dir)
    checker.check_all_files()


if __name__ == '__main__':
    main()