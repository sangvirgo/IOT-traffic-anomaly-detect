import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("./raw_data")          # nơi bạn đặt file CICFlowMeter gốc
OUT_DIR = Path("./cleaned_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 50_000     # đọc theo chunk để không tràn RAM
MAX_ROWS_PER_FILE = 200_000

DROP_COLS = [
    "Src IP", "Dst IP", "Source IP", "Destination IP",
    "Src Port", "Dst Port",
    "Flow ID", "FlowID",
    "Timestamp"
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    rename_map = {
        "Src IP": "Src IP",
        "Dst IP": "Dst IP",
        "Src Port": "Src Port",
        "Dst Port": "Dst Port",
        "Flow ID": "Flow ID",
        "Label": "Label"
    }
    df.rename(columns=rename_map, inplace=True)
    df.columns = df.columns.str.replace(" ", "")
    return df

def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame | None:
    chunk = standardize_columns(chunk)
    if "Label" not in chunk.columns:
        return None

    # bỏ trùng, xử lý NaN
    chunk = chunk.drop_duplicates()
    # giữ lại các số, các cột string sẽ chuyển NaN
    num_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
    chunk[num_cols] = chunk[num_cols].replace([np.inf, -np.inf], np.nan)
    chunk[num_cols] = chunk[num_cols].fillna(0)

    # chuẩn hóa nhãn: Benign -> 0, Attack -> 1
    chunk["Label"] = (
        chunk["Label"].astype(str).str.strip().str.lower()
        .map(lambda x: 0 if "benign" in x else 1)
    )

    # drop IP/Port/Timestamp/FlowID
    drop_cols = [c for c in chunk.columns
                 if any(key.replace(" ", "") in c for key in
                        ["SrcIP", "DstIP", "SrcPort", "DstPort", "FlowID", "Timestamp"])]
    drop_cols = [c for c in drop_cols if c in chunk.columns]
    if drop_cols:
        chunk = chunk.drop(columns=drop_cols)

    return chunk

def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))[:1]
    if not csv_files:
        print(f"No CSV files in {RAW_DIR}, please copy raw CICFlowMeter files there.")
        return

    total_benign, total_attack = 0, 0

    for i, fp in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Cleaning {fp.name} ...")
        cleaned_chunks = []
        rows = 0

        for chunk in pd.read_csv(fp, chunksize=CHUNKSIZE, low_memory=False):
            c = clean_chunk(chunk)
            if c is None or len(c) == 0:
                continue
            cleaned_chunks.append(c)
            rows += len(c)
            if rows >= MAX_ROWS_PER_FILE:
                print(f"  Reached {MAX_ROWS_PER_FILE} rows, stop reading this file.")
                break

        if not cleaned_chunks:
            print("  No valid data after cleaning, skip.")
            continue

        df_clean = pd.concat(cleaned_chunks, ignore_index=True)
        benign = (df_clean["Label"] == 0).sum()
        attack = (df_clean["Label"] == 1).sum()
        total_benign += benign
        total_attack += attack

        out_path = OUT_DIR / f"cleaned_{fp.name}"
        df_clean.to_csv(out_path, index=False)
        print(f"  Saved {len(df_clean)} rows -> {out_path.name}")
        print(f"  Benign: {benign}, Attack: {attack}")

    total = total_benign + total_attack
    if total > 0:
        ratio_b = total_benign / total * 100
        ratio_a = total_attack / total * 100
        print(f"\nDONE. Total Benign: {total_benign} ({ratio_b:.1f}%), "
              f"Attack: {total_attack} ({ratio_a:.1f}%)")
    else:
        print("\nDONE but no data produced.")

if __name__ == "__main__":
    main()
