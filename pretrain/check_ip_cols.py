"""
Check if CICIDS2018 files contain IP address columns
Scans all CSV files and reports which ones have Src IP and Dst IP
Uses chunked reading to avoid OOM

Usage:
    python check_ip_columns.py --data_dir ./raw_data
    python check_ip_columns.py --data_dir ./raw_data --quick
    python check_ip_columns.py --data_dir ./raw_data --chunksize 50000
"""

import pandas as pd
from pathlib import Path
import argparse


class IPColumnChecker:
    def __init__(self, data_dir, chunksize=100000):
        self.data_dir = Path(data_dir)
        self.chunksize = chunksize
        
    def check_single_file(self, filepath, quick_mode=False):
        """Check if a single CSV file has IP columns"""
        print(f"\n{'='*60}")
        print(f"File: {filepath.name}")
        print(f"{'='*60}")
        
        try:
            # Quick mode: only read first chunk to check columns
            if quick_mode:
                print(f"⚡ Quick mode: Reading first {self.chunksize:,} rows only")
                df = pd.read_csv(filepath, nrows=self.chunksize)
                total_rows = "Unknown (quick mode)"
            else:
                # Read file info efficiently
                print(f"📊 Analyzing file structure...")
                
                # First, read just the header
                df_header = pd.read_csv(filepath, nrows=0)
                
                # Count total rows by reading in chunks
                total_rows = 0
                chunk_count = 0
                for chunk in pd.read_csv(filepath, chunksize=self.chunksize):
                    total_rows += len(chunk)
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        print(f"  Processed {total_rows:,} rows...", end='\r')
                
                print(f"  Total rows: {total_rows:,}                    ")
                
                # Use the header for column analysis
                df = df_header
            
            # Get all columns
            print(f"\n📋 Total columns: {len(df.columns)}")
            
            # Normalize column names (remove spaces)
            normalized_cols = [col.strip().lower() for col in df.columns]
            
            # Check for IP columns (multiple variations)
            src_ip_variations = ['src ip', 'src_ip', 'source ip', 'source_ip', 'srcip']
            dst_ip_variations = ['dst ip', 'dst_ip', 'destination ip', 'destination_ip', 'dstip', 'dest ip', 'dest_ip']
            
            has_src_ip = any(var in normalized_cols for var in src_ip_variations)
            has_dst_ip = any(var in normalized_cols for var in dst_ip_variations)
            
            # Find actual column names
            src_ip_col = None
            dst_ip_col = None
            
            for col in df.columns:
                col_normalized = col.strip().lower()
                if col_normalized in src_ip_variations:
                    src_ip_col = col
                if col_normalized in dst_ip_variations:
                    dst_ip_col = col
            
            # Print results
            print(f"\n🔍 IP Column Check:")
            if has_src_ip:
                print(f"  ✅ Source IP: YES → '{src_ip_col}'")
                # Show sample values (read small chunk)
                if not quick_mode:
                    sample_df = pd.read_csv(filepath, nrows=1000)
                    sample_ips = sample_df[src_ip_col].dropna().unique()[:5]
                    print(f"     Sample values: {list(sample_ips)}")
            else:
                print(f"  ❌ Source IP: NO")
            
            if has_dst_ip:
                print(f"  ✅ Destination IP: YES → '{dst_ip_col}'")
                # Show sample values
                if not quick_mode:
                    sample_df = pd.read_csv(filepath, nrows=1000)
                    sample_ips = sample_df[dst_ip_col].dropna().unique()[:5]
                    print(f"     Sample values: {list(sample_ips)}")
            else:
                print(f"  ❌ Destination IP: NO")
            
            # Overall status
            print(f"\n📌 Status:", end=" ")
            if has_src_ip and has_dst_ip:
                print("✅ BOTH IPs AVAILABLE - SUITABLE FOR IP-BASED GNN")
            elif has_src_ip or has_dst_ip:
                print("⚠️  ONLY ONE IP AVAILABLE - PARTIAL DATA")
            else:
                print("❌ NO IPs AVAILABLE - USE FLOW-BASED GNN")
            
            # Show first few columns for reference
            print(f"\n📝 First 10 columns:")
            for i, col in enumerate(df.columns[:10], 1):
                print(f"  {i}. {col}")
            if len(df.columns) > 10:
                print(f"  ... and {len(df.columns) - 10} more columns")
            
            # File size info
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"\n💾 File size: {file_size_mb:.2f} MB")
            
            return {
                'filename': filepath.name,
                'has_src_ip': has_src_ip,
                'has_dst_ip': has_dst_ip,
                'src_ip_col': src_ip_col,
                'dst_ip_col': dst_ip_col,
                'total_rows': total_rows if not quick_mode else "Unknown",
                'total_cols': len(df.columns),
                'file_size_mb': file_size_mb
            }
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return {
                'filename': filepath.name,
                'error': str(e)
            }
    
    def check_all_files(self, quick_mode=False):
        """Check all CSV files in directory"""
        print("="*80)
        print("CICIDS2018 IP COLUMN CHECKER")
        print("="*80)
        print(f"\n📁 Scanning directory: {self.data_dir}")
        print(f"🔧 Chunk size: {self.chunksize:,} rows")
        if quick_mode:
            print(f"⚡ Quick mode: ON (only check first chunk)")
        
        # Find all CSV files
        csv_files = sorted(self.data_dir.glob('*.csv'))
        
        if not csv_files:
            print(f"\n❌ No CSV files found in {self.data_dir}")
            return
        
        print(f"📂 Found {len(csv_files)} CSV files\n")
        
        # Check each file
        results = []
        total_size_mb = 0
        
        for i, filepath in enumerate(csv_files, 1):
            print(f"\n{'#'*80}")
            print(f"File {i}/{len(csv_files)}")
            result = self.check_single_file(filepath, quick_mode)
            results.append(result)
            
            if 'file_size_mb' in result:
                total_size_mb += result['file_size_mb']
        
        # Print summary
        self.print_summary(results, total_size_mb)
        
        return results
    
    def print_summary(self, results, total_size_mb):
        """Print summary of all files"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        files_with_both_ips = []
        files_with_one_ip = []
        files_with_no_ips = []
        files_with_errors = []
        
        for result in results:
            if 'error' in result:
                files_with_errors.append(result['filename'])
            elif result['has_src_ip'] and result['has_dst_ip']:
                files_with_both_ips.append(result)
            elif result['has_src_ip'] or result['has_dst_ip']:
                files_with_one_ip.append(result)
            else:
                files_with_no_ips.append(result)
        
        print(f"\n📊 Files with BOTH Src & Dst IP: {len(files_with_both_ips)}/{len(results)}")
        for result in files_with_both_ips:
            fname = result['filename']
            size = result.get('file_size_mb', 0)
            rows = result.get('total_rows', 'Unknown')
            print(f"  ✅ {fname} ({size:.1f} MB, {rows} rows)")
        
        if files_with_one_ip:
            print(f"\n⚠️  Files with ONLY ONE IP: {len(files_with_one_ip)}/{len(results)}")
            for result in files_with_one_ip:
                fname = result['filename']
                has_src = "Src" if result['has_src_ip'] else ""
                has_dst = "Dst" if result['has_dst_ip'] else ""
                print(f"  ⚠️  {fname} (has {has_src}{has_dst} only)")
        
        if files_with_no_ips:
            print(f"\n❌ Files with NO IPs: {len(files_with_no_ips)}/{len(results)}")
            for result in files_with_no_ips:
                print(f"  ❌ {result['filename']}")
        
        if files_with_errors:
            print(f"\n⚠️  Files with ERRORS: {len(files_with_errors)}/{len(results)}")
            for fname in files_with_errors:
                print(f"  ⚠️  {fname}")
        
        # Dataset statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"  Total files: {len(results)}")
        print(f"  Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        
        total_rows = sum(r.get('total_rows', 0) for r in results if isinstance(r.get('total_rows'), int))
        if total_rows > 0:
            print(f"  Total rows: {total_rows:,}")
        
        # Final recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION FOR GNN")
        print("="*80)
        
        if len(files_with_both_ips) == len(results) and len(results) > 0:
            print("\n✅ ALL FILES HAVE BOTH IPs!")
            print("\n💡 You can choose:")
            print("   Option 1: IP-based GNN (traditional approach)")
            print("             └─ Build graph: IP nodes, flow edges")
            print("   Option 2: Flow-based KNN GNN (your advanced approach)")
            print("             └─ Build graph: Flow nodes, similarity edges")
            print("   Option 3: Both (recommended for comparison)")
            print("             └─ Show that KNN approach works without IPs")
        elif len(files_with_both_ips) > 0:
            print(f"\n⚠️  PARTIAL: {len(files_with_both_ips)}/{len(results)} files have IPs")
            print("\n💡 Recommendation:")
            print("   - Use IP-based GNN only for files with IPs")
            print("   - Use Flow-based KNN GNN for all files")
        else:
            print("\n❌ NO FILES HAVE IP ADDRESSES")
            print("\n💡 You MUST use:")
            print("   - Flow-based KNN GNN approach")
            print("   - Cannot use IP-based graph construction")
        
        print("\n🚀 Next steps:")
        print("  1. If you have IPs: clean_cicids.py --mode both")
        print("  2. If no IPs: clean_cicids.py --mode cnn_lstm")
        print("  3. Split for AL: split_for_active_learning.py")
        print("  4. Build KNN graph for GNN")


def main():
    parser = argparse.ArgumentParser(
        description='Check if CICIDS2018 CSV files contain IP address columns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full check (counts all rows, may take time)
  python check_ip_columns.py --data_dir ./raw_data
  
  # Quick check (only check column names)
  python check_ip_columns.py --data_dir ./raw_data --quick
  
  # Custom chunk size
  python check_ip_columns.py --data_dir ./raw_data --chunksize 50000
        """
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help='Directory containing CSV files (raw_data or cleaned_data)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: only check column names without counting rows'
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=100000,
        help='Chunk size for reading large files (default: 100000)'
    )
    
    args = parser.parse_args()
    
    # Create checker
    checker = IPColumnChecker(args.data_dir, chunksize=args.chunksize)
    
    # Run check
    results = checker.check_all_files(quick_mode=args.quick)
    
    print("\n✅ Check complete!")
    
    if args.quick:
        print("\n💡 Tip: Run without --quick to get exact row counts")


if __name__ == '__main__':
    main()