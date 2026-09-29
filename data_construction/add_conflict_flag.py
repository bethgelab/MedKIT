import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description="Flag rows with conflicting answers for the same (condition, context, endpoint, regimen, comparator).")
parser.add_argument(
    "--input", "--in-csv",
    dest="input",
    default="medkit_v4.csv",
    help="Input CSV path (default: medkit_v4.csv in the current directory).",
)
parser.add_argument(
    "--output", "--out-csv",
    dest="output",
    default=None,
    help="Output CSV path (default: <input stem>_conflict_flagged.csv alongside input).",
)
args = parser.parse_args()

input_path = Path(args.input)
output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_conflict_flagged.csv")

print(f"Loading {input_path} ...")
df = pd.read_csv(input_path)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

n_distinct = df.groupby(["condition", "context", "endpoint", "regimen", "comparator"])["answer"].transform("nunique")
df["conflicting_edit"] = n_distinct > 1

print("\nconflicting_edit distribution:")
print(df["conflicting_edit"].value_counts())

print(f"\nSaving to {output_path} ...")
df.to_csv(output_path, index=False)
print("Done.")
