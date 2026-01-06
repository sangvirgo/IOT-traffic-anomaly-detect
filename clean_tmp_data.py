import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("./raw_data")
OUT_DIR = Path("./cleaned_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 50_000
MAX_ROWS_PER_FILE = 200_000

EXCLUDE_COLS = [
    'Timestamp', 'Flow_ID', 'Flow_ID_',
    'Src_IP', 'Source_IP', 'SrcIP', 'Src_Port',
    'Dst_IP', 'Destination_IP', 'DstIP', 'Dest_IP', 'Dst_Port'
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names"""
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    rename_map = {
        'Src_IP': 'Src_IP',
        'Dst_IP': 'Dst_IP',
        'Src_Port': 'Src_Port',
        'Dst_Port': 'Dst_Port',
        'Flow_ID': 'Flow_ID',
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame | None:
    """Clean chunk - EXACTLY AS NOTEBOOK"""
    # === FIX: Copy để tránh SettingWithCopyWarning ===
    chunk = chunk.copy()
    
    chunk = standardize_columns(chunk)
    
    if "Label" not in chunk.columns:
        return None

    # Drop duplicates
    chunk = chunk.drop_duplicates()
    
    # === FIX: Binary label (0=Benign, 1=Attack) ===
    chunk["Label"] = chunk["Label"].astype(str).str.strip().str.lower()
    chunk["Label"] = chunk["Label"].apply(lambda x: 0 if 'benign' in x else 1)

    # Convert to numeric + handle NaN/Inf
    for col in chunk.columns:
        if col == 'Label':
            continue  # Giữ Label
        
        # Skip excluded columns
        if col in EXCLUDE_COLS or any(excl in col for excl in EXCLUDE_COLS):
            continue
        
        try:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        except:
            pass
    
    # Fill NaN/Inf for numeric columns only
    num_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in num_cols:
        num_cols.remove('Label')  # Đừng fill Label
    
    if num_cols:
        chunk[num_cols] = chunk[num_cols].replace([np.inf, -np.inf], np.nan)
        chunk[num_cols] = chunk[num_cols].fillna(0)

    drop_cols = [c for c in chunk.columns 
                 if c != 'Label' and (c in EXCLUDE_COLS or any(excl in c for excl in EXCLUDE_COLS))]
    
    if drop_cols:
        chunk = chunk.drop(columns=drop_cols)

    return chunk

def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))[:1]  # CHỈ 1 FILE
    
    if not csv_files:
        print(f"No CSV files in {RAW_DIR}")
        return

    print("=" * 70)
    print("CLEANING DATA - MATCHING NOTEBOOK PREPROCESSING")
    print("=" * 70)
    
    total_benign, total_attack = 0, 0

    for i, fp in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] {fp.name}")
        cleaned_chunks = []
        rows = 0

        try:
            for chunk in pd.read_csv(fp, chunksize=CHUNKSIZE, low_memory=False,
                                     on_bad_lines='skip', encoding='utf-8', 
                                     encoding_errors='ignore'):
                c = clean_chunk(chunk)
                if c is None or len(c) == 0:
                    continue
                
                cleaned_chunks.append(c)
                rows += len(c)
                
                if rows >= MAX_ROWS_PER_FILE:
                    print(f"  Reached {MAX_ROWS_PER_FILE:,} rows limit")
                    break
        
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue

        if not cleaned_chunks:
            print("  No valid data, skip")
            continue

        # Concatenate chunks
        df_clean = pd.concat(cleaned_chunks, ignore_index=True)
        
        # === VERIFY Label column exists ===
        if 'Label' not in df_clean.columns:
            print(f"  ERROR: Label column missing after cleaning!")
            print(f"  Columns: {df_clean.columns.tolist()}")
            continue
        
        benign = (df_clean["Label"] == 0).sum()
        attack = (df_clean["Label"] == 1).sum()
        total_benign += benign
        total_attack += attack

        out_path = OUT_DIR / f"cleaned_{fp.name}"
        df_clean.to_csv(out_path, index=False)
        
        print(f"  ✓ Saved {len(df_clean):,} rows")
        print(f"    Features: {len(df_clean.columns)-1}")  # -1 for Label
        print(f"    Benign: {benign:,}, Attack: {attack:,}")
        
        # Show first few columns for verification
        print(f"    Columns (first 5): {df_clean.columns.tolist()[:5]}")
        print(f"    Has Label: {'Label' in df_clean.columns}")

    print("\n" + "=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)
    
    if total_benign + total_attack > 0:
        print(f"Total Benign: {total_benign:,}")
        print(f"Total Attack: {total_attack:,}")
        print(f"Ratio: {total_benign/(total_benign+total_attack)*100:.1f}% Benign")
    else:
        print("No data produced")

if __name__ == "__main__":
    main()