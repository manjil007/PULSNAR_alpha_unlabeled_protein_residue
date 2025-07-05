from PULSNAR import PULSNAR
import numpy as np
import sys
import os
import pandas as pd
from sklearn.utils import shuffle
from collections import Counter


# def read_input_file(ifile):
#     """
#     Read the input file and return features and labels
#     """
#     return pd.read_csv(ifile)

def load_and_drop_nan_true_label(feature_csv: str, label_csv: str, true_label: str):
    """
    Load dense feature + label CSVs, drop rows that contain any NaN
    in the feature matrix, and return aligned NumPy arrays (X, y, y_true).
    """
    # 1 ▸ Read
    # X_df = pd.read_csv(feature_csv).iloc[:200000, :]
    # y_df = pd.read_csv(label_csv).iloc[:200000, :]
    # y_true_df = pd.read_csv(true_label).iloc[:200000, :]

    X_df = pd.read_csv(feature_csv).iloc
    y_df = pd.read_csv(label_csv).iloc
    y_true_df = pd.read_csv(true_label).iloc

    # 2 ▸ Sanity-check label file
    if y_df.shape[1] != 1 and y_df.shape[1] != 1:
        raise ValueError("Label file must have exactly one column")
    y_series = y_df.iloc[:, 0]

    # 3 ▸ Find rows with at least one NaN in features
    nan_mask = X_df.isna().any(axis=1)

    # 4 ▸ Drop same rows in X and y
    kept_mask = ~nan_mask
    X_df_clean = X_df.loc[kept_mask]
    y_clean    = y_series.loc[kept_mask]
    y_true_clean = y_true_df.loc[kept_mask]

    print(f'Number of true positives and unlabeled examples: {y_true_clean.value_counts()}')
    print(f'Number of labeled positives and unlabeled examples: {y_clean.value_counts()}')

    difference_in_zeros = (y_true_clean == 0).sum() - (y_clean == 0).sum()
    print(f"Difference in the number of zeros: {difference_in_zeros}")

    # 5 ▸ Convert to NumPy
    X = X_df_clean.to_numpy(dtype=np.float32)
    Y = y_clean.to_numpy(dtype=np.int32)
    Y_true = y_true_clean.to_numpy(dtype=np.int32)

    # 6 ▸ Report
    dropped = nan_mask.sum()
    print(f"Dropped {dropped:,} rows containing NaN "
          f"({dropped / len(nan_mask):.2%} of the dataset)")

    return X, Y, Y_true

def load_and_drop_nan(feature_csv: str, label_csv: str):
    """
    Load dense feature + label CSVs, drop rows that contain any NaN
    in the feature matrix, and return aligned NumPy arrays (X, y, y_true).
    """
    # 1 ▸ Read
    X_df = pd.read_csv(feature_csv)
    y_df = pd.read_csv(label_csv)

    # 2 ▸ Sanity-check label file
    if y_df.shape[1] != 1:
        raise ValueError("Label file must have exactly one column")
    y_series = y_df.iloc[:, 0]

    # 3 ▸ Find rows with at least one NaN in features
    nan_mask = X_df.isna().any(axis=1)

    # 4 ▸ Drop same rows in X and y
    kept_mask = ~nan_mask
    X_df_clean = X_df.loc[kept_mask]
    y_clean    = y_series.loc[kept_mask]

    # 5 ▸ Convert to NumPy
    X = X_df_clean.to_numpy(dtype=np.float32)
    Y = y_clean.to_numpy(dtype=np.int32)
    Y_true = Y.copy()

    # 6 ▸ Report
    dropped = nan_mask.sum()
    print(f"Dropped {dropped:,} rows containing NaN "
          f"({dropped / len(nan_mask):.2%} of the dataset)")

    return X, Y, Y_true

def read_input_file(feature_file: str, label_file: str):
    """
    Read feature and label files, convert to NumPy arrays,
    and return X, Y, Y_true.

    Parameters
    ----------
    feature_file : str
        Path to the CSV containing the feature matrix.
    label_file   : str
        Path to the CSV containing a single column of labels (0/1).

    Returns
    -------
    X : np.ndarray        # shape (n_samples, n_features)
    Y : np.ndarray        # shape (n_samples,)
    Y_true : np.ndarray   # identical copy of Y (needed by PULSNAR API)
    """
    # Load CSVs into DataFrames
    X_df = pd.read_csv(feature_file)
    y_df = pd.read_csv(label_file)

    # Sanity-check: label file should have exactly one column
    if y_df.shape[1] != 1:
        raise ValueError("Label file must contain exactly one column")

    # Convert to NumPy arrays
    X = X_df.to_numpy(dtype=np.float32)
    Y = y_df.iloc[:, 0].to_numpy(dtype=np.int32)

    # PULSNAR expects both Y and Y_true even when they’re identical
    Y_true = Y.copy()

    return X, Y, Y_true



def create_ml_dataset(data, pf, itr=0):
    """
    Create positive and unlabeled set
    """
    np.random.seed(itr)
    labels = data['label'].to_numpy()
    # print(Counter(labels))

    # find position of positive and unlabeled examples
    idx0, idx1 = np.where(labels == 1)[0], np.where(labels == 0)[0]

    # positive and unlabeled sets
    Y, Y_true = np.asarray([1] * len(labels)), np.asarray([1] * len(labels))
    Y[idx0], Y_true[idx0] = 0, 0
    # print("original pos and unlab count: ", np.sum(Y), len(Y) - np.sum(Y))

    # generate data
    dat = data.drop(columns=['label'])
    dat.replace(' ', 0, inplace=True)
    dat.replace(np.nan, 0, inplace=True)
    dat = dat.values  # dataframe to numpy

    # how many positives need to be flipped
    label_change_count = int(len(idx0) * pf / (1 - pf))
    np.random.shuffle(idx1)
    i = idx1[: label_change_count]
    # print("total label change: ", len(i), label_change_count)
    Y[i] = 0

    return dat, Y, Y_true


# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
# inpfile = "/home/mpradhan/Intern_Research_Project/data/X_balanced_dense.csv"
# inpt_label_file = "/home/mpradhan/Intern_Research_Project/data/y_balanced.csv"

inpfile = "/home/mpradhan/Intern_Research_Project/data/X_manipulated_new.csv"
inpt_label_file = "/home/mpradhan/Intern_Research_Project/data/y_manipulated_new_pu.csv"
inpt_true_label_file = "/home/mpradhan/Intern_Research_Project/data/y_manipulated_new_true.csv"


# get parameters from user for PULSNAR algorithm.
# update pulscar_diabetes_alpha.yaml if you want to override the default parameters
# if len(sys.argv) < 2:
#     user_param_file = 'testparams/pulscar_diabetes_alpha.yaml'
# else:
#     user_param_file = sys.argv[1]

if len(sys.argv) < 2:
    user_param_file = '/home/mpradhan/Intern_Research_Project/data/pulsnar_residues_alpha_validation.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("../results"):
    os.makedirs("../results")

# X, Y, Y_true = read_input_file(inpfile, inpt_label_file)

# X, Y, Y_true = load_and_drop_nan(inpfile, inpt_label_file)

X, Y, Y_true = load_and_drop_nan_true_label(inpfile, inpt_label_file, inpt_true_label_file)

## PRINT KNOWN NUMBER AND PROPORTION OF POSITIVES IN UNLABELED DATASET BY 


print("Data shape, #positive, #unlab: ", X.shape, np.sum(Y), len(Y) - np.sum(Y))
rec_ids = np.array([i for i in range(len(Y_true))])

pls = PULSNAR.PULSNARClassifier(scar=False, csrdata=False, classifier='xgboost',
                                        bin_method='rice', bw_method='hist', lowerbw=0.01, upperbw=0.5, optim='local',
                                        calibration=False, calibration_data='PU', calibration_method='isotonic',
                                        calibration_n_bins=100, smooth_isotonic=False,
                                        classification_metrics=False, n_iterations=1, kfold=5, kflips=1,
                                        pulsnar_params_file=user_param_file)

        # get results
res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)

iter_alpha = res['estimated_alpha']
print(f"Estimated alpha: {iter_alpha}")

# print alpha
print("\n")
print("Algorithm\tIteration\tTrue_alpha\tEst_alpha")