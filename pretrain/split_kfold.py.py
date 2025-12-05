"""
Split dữ liệu bằng K-Fold cho Binary Classification
- Stratified để đảm bảo tỷ lệ 0/1 đều
- Lưu từng fold ra file .npy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle
import gc

class KFoldSplitter:
    def __init__(self, cleaned_dir, output_dir, n_splits=5, random_state=42):
        """
        Args:
            cleaned_dir: Thư mục chứa cleaned CSV files
            output_dir: Thư mục lưu kết quả K-Fold
            n_splits: Số fold (default=5)
            random_state: Random seed
        """
        self.cleaned_dir = Path(cleaned_dir)
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Tạo thư mục output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'folds').mkdir(exist_ok=True)
        (self.output_dir / 'scalers').mkdir(exist_ok=True)
    
    def load_all_data(self, max_samples_per_file=None):
        """
        Load tất cả cleaned files
        max_samples_per_file: Giới hạn số dòng mỗi file (để fit RAM Colab)
        """
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        csv_files = sorted(self.cleaned_dir.glob('cleaned_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"Không tìm thấy file cleaned trong {self.cleaned_dir}")
        
        print(f"Tìm thấy {len(csv_files)} file cleaned\n")
        
        all_data = []
        total_rows = 0
        
        for filepath in csv_files:
            print(f"Đang load: {filepath.name}...", end=' ')
            
            # Nếu muốn giới hạn số dòng (cho file 4GB)
            if max_samples_per_file:
                df = pd.read_csv(filepath, nrows=max_samples_per_file)
            else:
                df = pd.read_csv(filepath)
            
            all_data.append(df)
            total_rows += len(df)
            print(f"✓ {len(df):,} dòng")
        
        # Gộp tất cả
        print(f"\nĐang gộp {len(all_data)} file...")
        df_all = pd.concat(all_data, ignore_index=True)
        del all_data
        gc.collect()
        
        print(f"✅ Tổng cộng: {len(df_all):,} dòng, {len(df_all.columns)} cột")
        
        # Thống kê nhãn
        label_counts = df_all['Label'].value_counts()
        print(f"\nPhân phối nhãn:")
        print(f"  Benign (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df_all)*100:.1f}%)")
        print(f"  Attack (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df_all)*100:.1f}%)")
        
        return df_all
    
    def prepare_features(self, df):
        """Tách features và labels"""
        print("\n" + "="*70)
        print("CHUẨN BỊ FEATURES")
        print("="*70)
        
        # Tách X và y
        y = df['Label'].values.astype(np.int32)
        X = df.drop(columns=['Label']).values.astype(np.float32)
        
        feature_names = [c for c in df.columns if c != 'Label']
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]:,}")
        print(f"Memory: {X.nbytes / (1024**2):.1f} MB")
        
        # Lưu feature names
        with open(self.output_dir / 'scalers' / 'feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        
        return X, y, feature_names
    
    def create_kfold_splits(self, X, y):
        """Tạo K-Fold splits với Stratified"""
        print("\n" + "="*70)
        print(f"TẠO {self.n_splits}-FOLD SPLITS (STRATIFIED)")
        print("="*70)
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                              random_state=self.random_state)
        
        fold_info = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold_idx}/{self.n_splits}:")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Normalize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Thống kê
            train_benign = (y_train == 0).sum()
            train_attack = (y_train == 1).sum()
            test_benign = (y_test == 0).sum()
            test_attack = (y_test == 1).sum()
            
            print(f"  Train: {len(X_train):,} samples")
            print(f"    - Benign: {train_benign:,} ({train_benign/len(y_train)*100:.1f}%)")
            print(f"    - Attack: {train_attack:,} ({train_attack/len(y_train)*100:.1f}%)")
            print(f"  Test: {len(X_test):,} samples")
            print(f"    - Benign: {test_benign:,} ({test_benign/len(y_test)*100:.1f}%)")
            print(f"    - Attack: {test_attack:,} ({test_attack/len(y_test)*100:.1f}%)")
            
            # Lưu fold
            fold_dir = self.output_dir / 'folds' / f'fold_{fold_idx}'
            fold_dir.mkdir(exist_ok=True)
            
            np.save(fold_dir / 'X_train.npy', X_train_scaled)
            np.save(fold_dir / 'y_train.npy', y_train)
            np.save(fold_dir / 'X_test.npy', X_test_scaled)
            np.save(fold_dir / 'y_test.npy', y_test)
            
            # Lưu scaler
            with open(fold_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            fold_info.append({
                'fold': fold_idx,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_benign': train_benign,
                'train_attack': train_attack,
                'test_benign': test_benign,
                'test_attack': test_attack,
            })
            
            print(f"  ✓ Đã lưu tại: {fold_dir}")
        
        # Lưu metadata
        metadata = {
            'n_splits': self.n_splits,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'random_state': self.random_state,
            'fold_info': fold_info
        }
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\n" + "="*70)
        print("✅ HOÀN TẤT K-FOLD SPLIT!")
        print("="*70)
        print(f"Output: {self.output_dir}")
        print(f"\nCách load:")
        print(f"  fold_dir = '{self.output_dir}/folds/fold_1'")
        print(f"  X_train = np.load(fold_dir + '/X_train.npy')")
        print(f"  y_train = np.load(fold_dir + '/y_train.npy')")
    
    def run(self, max_samples_per_file=None):
        """Chạy toàn bộ pipeline"""
        # 1. Load data
        df = self.load_all_data(max_samples_per_file)
        
        # 2. Chuẩn bị features
        X, y, feature_names = self.prepare_features(df)
        
        # Giải phóng df
        del df
        gc.collect()
        
        # 3. Tạo K-Fold splits
        self.create_kfold_splits(X, y)

# # Sử dụng trên Colab
# if __name__ == "__main__":
#     from google.colab import drive
#     drive.mount('/content/drive')
    
#     # Đường dẫn
#     cleaned_dir = '/content/drive/MyDrive/CICIDS/cleaned_data'
#     output_dir = '/content/drive/MyDrive/CICIDS/kfold_splits'
    
#     # Chạy
#     splitter = KFoldSplitter(
#         cleaned_dir=cleaned_dir,
#         output_dir=output_dir,
#         n_splits=5,  # 5-fold cross validation
#         random_state=42
#     )
    
#     # Nếu RAM không đủ, giới hạn mỗi file (ví dụ 1 triệu dòng)
#     # splitter.run(max_samples_per_file=1_000_000)
    
#     # Nếu RAM đủ, load hết
#     splitter.run()
