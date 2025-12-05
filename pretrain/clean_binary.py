"""
Clean CICIDS2018 cho Binary Classification + CNN/LSTM
- Benign = 0, Attack = 1
- Không cần IP
- Xử lý chunk cho file lớn
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BinaryCleaner:
    def __init__(self, input_dir, output_dir, chunk_size=500_000):
        """
        Args:
            input_dir: Thư mục chứa 10 file CSV gốc
            output_dir: Thư mục lưu file cleaned
            chunk_size: Số dòng đọc mỗi lần (quan trọng với file 4GB)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        # Tạo thư mục output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cột cần loại bỏ (IP, timestamp, flow ID, ports)
        self.cols_to_remove = [
            'Src IP', 'Src_IP', 'Source IP', 
            'Dst IP', 'Dst_IP', 'Destination IP',
            'Flow ID', 'Flow_ID',
            'Timestamp',
            'Src Port', 'Src_Port',
            'Dst Port', 'Dst_Port'
        ]
        
    def standardize_columns(self, df):
        """Chuẩn hóa tên cột"""
        df.columns = df.columns.str.strip()
        rename_map = {
            'Src IP': 'Src_IP',
            'Dst IP': 'Dst_IP',
            'Src Port': 'Src_Port',
            'Dst Port': 'Dst_Port',
            'Flow ID': 'Flow_ID',
            ' Label': 'Label',
            'Label ': 'Label',
        }
        df.rename(columns=rename_map, inplace=True)
        df.columns = df.columns.str.replace(' ', '_')
        return df
    
    def clean_chunk(self, chunk):
        """Clean 1 chunk dữ liệu"""
        # Chuẩn hóa tên cột
        chunk = self.standardize_columns(chunk)
        
        # Kiểm tra có Label không
        if 'Label' not in chunk.columns:
            return None
        
        # Xóa duplicate
        chunk = chunk.drop_duplicates()
        
        # Xử lý missing values
        missing_counts = chunk.isnull().sum(axis=1)
        chunk = chunk[missing_counts <= len(chunk.columns) * 0.3]
        
        # Fill NaN cho numeric
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if chunk[col].isnull().any():
                chunk[col].fillna(chunk[col].median(), inplace=True)
        
        # Thay inf bằng NaN rồi fill
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in numeric_cols:
            if chunk[col].isnull().any():
                chunk[col].fillna(chunk[col].median(), inplace=True)
        
        # Binary mapping: Benign=0, tất cả khác=1
        chunk['Label'] = chunk['Label'].str.strip()
        chunk['Label_Binary'] = chunk['Label'].apply(
            lambda x: 0 if x == 'Benign' else 1
        )
        
        # Loại bỏ các cột không cần (IP, timestamp, ports...)
        cols_found = [c for c in self.cols_to_remove if c in chunk.columns]
        if cols_found:
            chunk = chunk.drop(columns=cols_found)
        
        # Loại bỏ cột Label gốc, chỉ giữ Label_Binary
        if 'Label' in chunk.columns:
            chunk = chunk.drop(columns=['Label'])
        
        # Đổi tên Label_Binary → Label cho dễ dùng sau này
        chunk.rename(columns={'Label_Binary': 'Label'}, inplace=True)
        
        return chunk
    
    def process_file(self, filepath):
        """Xử lý 1 file CSV (support chunk cho file lớn)"""
        print(f"\n{'='*70}")
        print(f"Đang xử lý: {filepath.name}")
        print(f"{'='*70}")
        
        # Đếm dòng nhanh
        print("Đang đếm số dòng...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            total_rows = sum(1 for _ in f) - 1  # -1 vì header
        print(f"  Tổng dòng: {total_rows:,}")
        
        # Xác định file lớn hay nhỏ
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Kích thước: {file_size_mb:.1f} MB")
        
        # Xử lý theo chunk
        cleaned_chunks = []
        benign_count = 0
        attack_count = 0
        chunk_num = 0
        
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, 
                         low_memory=False, encoding='utf-8', encoding_errors='ignore'):
            chunk_num += 1
            print(f"  Chunk {chunk_num}: {len(chunk):,} dòng...", end='\r')
            
            cleaned = self.clean_chunk(chunk)
            if cleaned is not None and len(cleaned) > 0:
                # Đếm nhãn
                benign_count += (cleaned['Label'] == 0).sum()
                attack_count += (cleaned['Label'] == 1).sum()
                cleaned_chunks.append(cleaned)
        
        print()  # Newline sau progress
        
        if not cleaned_chunks:
            print("  ❌ Không có dữ liệu sau khi clean!")
            return None
        
        # Gộp tất cả chunks
        print("  Đang gộp chunks...")
        df_final = pd.concat(cleaned_chunks, ignore_index=True)
        
        # In thống kê
        print(f"\n  ✅ Hoàn thành:")
        print(f"    - Dòng cuối cùng: {len(df_final):,}")
        print(f"    - Features: {len(df_final.columns) - 1}")  # -1 vì Label
        print(f"    - Benign: {benign_count:,} ({benign_count/len(df_final)*100:.1f}%)")
        print(f"    - Attack: {attack_count:,} ({attack_count/len(df_final)*100:.1f}%)")
        
        # Lưu file cleaned
        output_path = self.output_dir / f"cleaned_{filepath.stem}.csv"
        df_final.to_csv(output_path, index=False)
        print(f"    - Đã lưu: {output_path.name}")
        
        return df_final
    
    def process_all(self):
        """Xử lý tất cả 10 files"""
        print("="*70)
        print("CLEAN CICIDS2018 - BINARY CLASSIFICATION")
        print("="*70)
        
        csv_files = sorted(self.input_dir.glob('*.csv'))
        if not csv_files:
            print(f"❌ Không tìm thấy file CSV trong {self.input_dir}")
            return
        
        print(f"\n📁 Tìm thấy {len(csv_files)} file")
        
        total_benign = 0
        total_attack = 0
        
        for i, filepath in enumerate(csv_files, 1):
            print(f"\n{'#'*70}")
            print(f"File {i}/{len(csv_files)}")
            df = self.process_file(filepath)
            
            if df is not None:
                total_benign += (df['Label'] == 0).sum()
                total_attack += (df['Label'] == 1).sum()
        
        # Tổng kết
        print("\n" + "="*70)
        print("✅ HOÀN TẤT CLEAN TẤT CẢ FILES!")
        print("="*70)
        print(f"Tổng Benign: {total_benign:,}")
        print(f"Tổng Attack: {total_attack:,}")
        print(f"Tỷ lệ: {total_benign/(total_benign+total_attack)*100:.1f}% Benign")
        print(f"\nFile cleaned đã lưu tại: {self.output_dir}")

