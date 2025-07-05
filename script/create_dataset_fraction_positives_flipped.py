import pandas as pd
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------------
DATA_DIR = Path("/home/mpradhan/Intern_Research_Project/data")
master_file = DATA_DIR / "X_master_dense_prob.csv"
label_file = DATA_DIR / "y_master.csv"

# Output files
X_out = DATA_DIR / "X_new.csv"
y_true_out = DATA_DIR / "y_new_true.csv"
y_pu_out = DATA_DIR / "y_new_pu.csv"

# How many positives to flip to unlabeled
FLIP_FRACTION = 0.2   # 20% of positives will become unlabeled
N_UNLABELED = 500_000 # How many low-prob unlabeled to keep

# ---------------------------------------------------------------------
# 2. Load
# ---------------------------------------------------------------------
X = pd.read_csv(master_file)
y = pd.read_csv(label_file).squeeze("columns")

assert len(X) == len(y)
df = X.copy()
df["true_label"] = y

# ---------------------------------------------------------------------
# 3. Filter pools
# ---------------------------------------------------------------------
positives = df[df["true_label"] == 1].copy()
negatives = df[(df["true_label"] == 0) & (df["xgb_pos_prob"] < 0.04)].copy()

print(f"Total positives: {len(positives):,}")
print(f"Total low-prob negatives: {len(negatives):,}")

# ---------------------------------------------------------------------
# 4. Flip some positives to fake-unlabeled
# ---------------------------------------------------------------------
n_flip = int(len(positives) * FLIP_FRACTION)
flipped_idx = positives.sample(n=n_flip, random_state=42).index

positives["pu_label"] = 1
positives.loc[flipped_idx, "pu_label"] = 0

# ---------------------------------------------------------------------
# 5. Draw subset of negatives
# ---------------------------------------------------------------------
negatives["pu_label"] = 0  # still unlabeled

negatives_sub = negatives.sample(n=N_UNLABELED, random_state=42)

# ---------------------------------------------------------------------
# 6. Combine
# ---------------------------------------------------------------------
new_df = pd.concat([
    positives, 
    negatives_sub
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Features
X_new = new_df.drop(columns=["true_label", "pu_label"])
# True labels
y_true = new_df["true_label"]
# PU labels (some positives flipped)
y_pu = new_df["pu_label"]

# ---------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------
X_new.to_csv(X_out, index=False)
y_true.to_csv(y_true_out, index=False)
y_pu.to_csv(y_pu_out, index=False)

print(f"Saved:")
print(f"  {X_out}")
print(f"  {y_true_out}")
print(f"  {y_pu_out}")

print(f"Shape: {X_new.shape}, True Positives: {y_true.sum()}, PU Positives: {y_pu.sum()}")
