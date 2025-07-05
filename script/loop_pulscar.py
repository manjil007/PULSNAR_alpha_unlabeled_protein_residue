#!/usr/bin/env python
"""
Run PULSNAR on all four prepared datasets (balanced, hi-pos-lo-neg, ran-neg-hi-pos,
ran-pos-low-neg) and save the estimated α values to results/*.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PULSNAR import PULSNAR

# -------------------------------------------------------------------------
# 0.  CONSTANTS  – edit here if your file names ever change
# -------------------------------------------------------------------------

# Root of your data tree on chiltepin (adjust if you move things)
DATA_DIR   = Path("/home/mpradhan007/Academic/Research_Projects/pulscar/data")
YAML_DIR   = DATA_DIR / "yaml_files"
RESULT_DIR = Path("/home/mpradhan007/Academic/Research_Projects/pulscar/results")    # relative to script location
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# One dictionary entry per dataset folder
DATASETS = {
    "balanced_dataset": {
        "X"      : "X_balanced_dense.csv",
        "y_pu"   : "y_balanced.csv",
        "y_true" : "y_balanced.csv",
        "yaml"   : "pulscar_residues_alpha_balanced.yaml",
    },
    "hi_positive_low_negatives": {
        "X"      : "X_prob_hi_positives_lo_unlabeled.csv",
        "y_pu"   : "y_prob_hi_positives_lo_unlabeled_flipped.csv",
        "y_true" : "y_prob_hi_positives_lo_unlabeled_true.csv",
        "yaml"   : "pulscar_residues_alpha_hi_positive_low_negatives.yaml",
    },
    "ran_neg_hi_pos": {
        "X"      : "X_ran_neg_hi_pos.csv",
        "y_pu"   : "y_ran_neg_hi_pos_flipped.csv",
        "y_true" : "y_ran_neg_hi_pos_true.csv",
        "yaml"   : "pulscar_residues_alpha_ran_negatives_hi_pos.yaml",
    },
    "ran_pos_low_neg": {
        "X"      : "X_ran_pos_low_unlab.csv",
        "y_pu"   : "y_ran_pos_low_unlab_flipped.csv",
        "y_true" : "y_ran_pos_low_unlab_true.csv",
        "yaml"   : "pulscar_residues_alpha_ran_positive_low_neg.yaml",
    },
}

# -------------------------------------------------------------------------
# 1.  Small helper to read three CSV files and return NumPy arrays
# -------------------------------------------------------------------------
def read_input_files(feature_file: Path,
                     label_file:   Path,
                     true_file:    Path,
                     seed: int = 42):
    """Return X, Y (PU-labels), Y_true from three CSV files, shuffled in unison."""
    # --- read the first 10 rows
    X_df      = pd.read_csv(feature_file)
    y_df      = pd.read_csv(label_file).squeeze("columns")
    y_true_df = pd.read_csv(true_file).squeeze("columns")

    if y_df.ndim != 1:
        raise ValueError(f"{label_file.name} must contain exactly one column")

    # --- convert to numpy
    X      = X_df.to_numpy(dtype=np.float32)
    Y      = y_df.to_numpy(dtype=np.int32)
    Y_true = y_true_df.to_numpy(dtype=np.int32)

    # --- shuffle in unison
    rng  = np.random.default_rng(seed)          # reproducible shuffle
    perm = rng.permutation(X.shape[0])          # same permutation for all
    X, Y, Y_true = X[perm], Y[perm], Y_true[perm]

    return X, Y, Y_true


# -------------------------------------------------------------------------
# 2.  Main loop
# -------------------------------------------------------------------------
def main():
    for ds_name, files in DATASETS.items():
        print(f"\n=== Running dataset: {ds_name} ===")

        folder = DATA_DIR / ds_name
        X_path      = folder / files["X"]
        y_pu_path   = folder / files["y_pu"]
        y_true_path = folder / files["y_true"]
        yaml_path   = YAML_DIR / files["yaml"]

        # ---- load data
        X, Y, Y_true = read_input_files(X_path, y_pu_path, y_true_path)
        rec_ids      = np.arange(len(Y))        # simple 0..n-1 ID list

        # ---- initialise the classifier
        pls = PULSNAR.PULSNARClassifier(
            scar=True, csrdata=False, classifier='xgboost',
            bin_method='rice', bw_method='hist', lowerbw=0.01, upperbw=0.5,
            optim='local', calibration=False, calibration_data='PU',
            calibration_method='isotonic', calibration_n_bins=100,
            smooth_isotonic=False, classification_metrics=False,
            n_iterations=1, kfold=5, kflips=1,
            pulsnar_params_file=str(yaml_path)
        )

        # ---- run PULSNAR
        res        = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
        est_alpha  = np.atleast_1d(res["estimated_alpha"])   # ensure iterable

        # ---- save the alpha(s)
        out_df = pd.DataFrame({
            "dataset"   : ds_name,
            "iteration" : np.arange(1, len(est_alpha) + 1),
            "est_alpha" : est_alpha,
        })
        out_file = RESULT_DIR / f"{ds_name}_alpha.csv"
        out_df.to_csv(out_file, index=False)

        print(f"  -> estimated α = {est_alpha}")
        print(f"  -> saved to {out_file.resolve()}")

    print("\n✓ All datasets finished.")


if __name__ == "__main__":
    main()
