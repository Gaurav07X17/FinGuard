"""
mapping_confirm_cli.py

Simple CLI to show and confirm auto-guessed mapping.
"""
from typing import Dict
import argparse
import pandas as pd
from phase3.mapping import auto_guess_mapping


def confirm_mapping_cli(csv_path: str):
    df = pd.read_csv(csv_path)
    mapping = auto_guess_mapping(list(df.columns))
    print("Auto-detected mapping:")
    for k, v in mapping.items():
        print(f"  {k}: {v}")
    resp = input("Accept mapping? (y/n): ")
    if resp.lower().startswith("y"):
        return mapping
    else:
        custom = {}
        for k in mapping.keys():
            val = input(f"Enter column name for '{k}' (blank to leave None): ")
            custom[k] = val if val.strip() else None
        return custom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    args = parser.parse_args()
    mapping = confirm_mapping_cli(args.csv)
    print("Final mapping:", mapping)
