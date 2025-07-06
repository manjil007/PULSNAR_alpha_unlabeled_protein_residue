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
X_out = DATA_DIR / "X_balanced_validation.csv"
y_true_out = DATA_DIR / "y_balanced_true_validation.csv"
y_pu_out = DATA_DIR / "y_balanced_validation.csv"

# How many positives to flip to unlabeled
FLIP_FRACTION = 0.10   # 10% of positives will become unlabeled
N_UNLABELED = 300_000  # How many lowest-prob unlabeled to keep

# ---------------------------------------------------------------------
# 2. Load and drop NaNs immediately
# ---------------------------------------------------------------------
X = pd.read_csv(master_file)
y = pd.read_csv(label_file).squeeze("columns")

assert len(X) == len(y)

# Identify rows with any NaNs in X
nan_mask = X.isna().any(axis=1)
kept_mask = ~nan_mask

# Drop rows with NaNs in X and keep same rows in y
X = X.loc[kept_mask].reset_index(drop=True)
y = y.loc[kept_mask].reset_index(drop=True)

print(f"After dropping NaNs: {X.shape[0]:,} rows remain")

# ---------------------------------------------------------------------
# 3. Combine features + true labels
# ---------------------------------------------------------------------
df = X.copy()
df["true_label"] = y

# ---------------------------------------------------------------------
# 4. Filter pools
# ---------------------------------------------------------------------
positives = df[df["true_label"] == 1].copy()
negatives = df[df["true_label"] == 0].copy()  # take ALL true negatives

print(f"Total positives: {len(positives):,}")
print(f"Total true negatives: {len(negatives):,}")

# ---------------------------------------------------------------------
# 5. Flip 10% of positives to fake-unlabeled
# ---------------------------------------------------------------------
n_flip = int(len(positives) * FLIP_FRACTION)
flipped_idx = positives.sample(n=n_flip, random_state=42).index

positives["pu_label"] = 1
positives.loc[flipped_idx, "pu_label"] = 0

# ---------------------------------------------------------------------
# 6. Take lowest 300,000 true negatives by xgb_pos_prob
# ---------------------------------------------------------------------
negatives_sorted = negatives.sort_values(by="xgb_pos_prob", ascending=True)
negatives_sub = negatives_sorted.head(N_UNLABELED).copy()
negatives_sub["pu_label"] = 0  # always unlabeled

print(f"Selected lowest-prob unlabeled: {len(negatives_sub):,}")

# ---------------------------------------------------------------------
# 7. Combine and shuffle
# ---------------------------------------------------------------------
new_df = pd.concat([
    positives,
    negatives_sub
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Features only
X_new = new_df.drop(columns=["true_label", "pu_label"])
# True labels (1 for true positives, 0 for true negatives)
y_true = new_df["true_label"]
# PU labels (some positives flipped to unlabeled)
y_pu = new_df["pu_label"]

# ---------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------
X_new.to_csv(X_out, index=False)
y_true.to_csv(y_true_out, index=False)
y_pu.to_csv(y_pu_out, index=False)

print(f"Saved:")
print(f"  {X_out}")
print(f"  {y_true_out}")
print(f"  {y_pu_out}")

print(f"Shape: {X_new.shape}")
print(f"True Positives: {y_true.sum()} | PU Positives: {y_pu.sum()} | PU Unlabeled: {(y_pu == 0).sum()}")

# True positive residues that are still labeled positive
true_pos_pu_pos = ((y_true == 1) & (y_pu == 1)).sum()

# True positive residues that were flipped to unlabeled
true_pos_pu_unlab = ((y_true == 1) & (y_pu == 0)).sum()

# True unlabeled (negatives) that stayed unlabeled
true_unlab_pu_unlab = ((y_true == 0) & (y_pu == 0)).sum()

print(f"\n=== Final label breakdown ===")
print(f" True Positives: {y_true.sum():,}")
print(f"  - Still labeled positive: {true_pos_pu_pos:,}")
print(f"  - Flipped to unlabeled:   {true_pos_pu_unlab:,}")
print(f" True Negatives (unlabeled): {(y_true == 0).sum():,}")
print(f"  - Remain unlabeled:       {true_unlab_pu_unlab:,}")
print(f" Final PU labels: Positives={y_pu.sum():,}, Unlabeled={(y_pu == 0).sum():,}")
