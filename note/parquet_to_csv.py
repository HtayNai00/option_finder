#!/usr/bin/env python3
"""
Convert Parquet files to CSV.

Usage:
  python note/parquet_to_csv.py latest
  python note/parquet_to_csv.py enriched
"""

import os, sys
import pandas as pd

LATEST_PARQUET   = "data/latest/options_puts_latest.parquet"
ENRICHED_PARQUET = "data/latest/options_puts_enriched.parquet"

LATEST_CSV   = "data/latest/options_puts_latest.csv"
ENRICHED_CSV = "data/latest/options_puts_enriched.csv"

def convert(parquet_path, csv_path):
    if not os.path.exists(parquet_path):
        print(f"[ERR] File not found: {parquet_path}")
        return
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Wrote {len(df)} rows to {csv_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python note/parquet_to_csv.py [latest|enriched]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "latest":
        convert(LATEST_PARQUET, LATEST_CSV)
    elif mode == "enriched":
        convert(ENRICHED_PARQUET, ENRICHED_CSV)
    else:
        print("Invalid option. Use 'latest' or 'enriched'.")

if __name__ == "__main__":
    main()