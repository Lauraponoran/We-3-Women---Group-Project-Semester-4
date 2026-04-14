"""
build_corpus.py — Merge available per-event CSVs into corpus_all.parquet
=========================================================================
Run this before topic_model.py if you don't yet have corpus_all.parquet.

Usage
-----
    python build_corpus.py
    python build_corpus.py --input-dir path/to/news_output
"""

from __future__ import annotations

import argparse
import os
import glob

import pandas as pd

OUTPUT_DIR = "news_output"


def build_corpus(input_dir: str) -> None:
    # Find all per-event CSVs (not inside incremental folders)
    csv_files = [
        p for p in glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
        if "incremental" not in p and "corpus" not in os.path.basename(p)
    ]

    if not csv_files:
        print("❌ No CSV files found. Check your --input-dir path.")
        return

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  {f}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        print(f"  Loaded {len(df):,} rows from {os.path.basename(f)}")
        dfs.append(df)

    corpus = pd.concat(dfs, ignore_index=True)
    corpus.drop_duplicates(subset=["url"], inplace=True)

    out_path = os.path.join(input_dir, "corpus_all.parquet")
    corpus.to_parquet(out_path, index=False)

    print(f"\n✅ corpus_all.parquet built:")
    print(f"   Total articles    : {len(corpus):,}")
    print(f"   Unique publishers : {corpus['publisher'].nunique()}")
    print(f"   Event windows     : {corpus['event_label'].nunique()} → {corpus['event_label'].unique().tolist()}")
    print(f"   Saved → {out_path}")
    print(f"\n   Next step: python topic_model.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    build_corpus(args.input_dir)


if __name__ == "__main__":
    main()
