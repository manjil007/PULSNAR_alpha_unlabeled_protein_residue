import pandas as pd
from sklearn.decomposition import NMF

# Path to the master file
file_path = '/home/mpradhan/Intern_Research_Project/data/X_master_dense.csv'

# 1. Read the master file
X = pd.read_csv(file_path)

# 2. Apply Non-Negative Matrix Factorization (NMF)
# Remove non-numeric columns if any (optional)
X_numeric = X.select_dtypes(include=[float, int])

# Drop rows with any NaN values
X_numeric = X_numeric.dropna()

# Find the minimum value
min_val = X_numeric.min().min()

if min_val < 0:
    X_numeric = X_numeric + abs(min_val)
else:
    X_numeric = X_numeric
# Show the shape after dropping
print(X_numeric.shape)

# Create an NMF instance
nmf_model = NMF(n_components='auto', init='random', random_state=42)  # adjust n_components as needed

# Fit and transform
X_nmf = nmf_model.fit_transform(X_numeric)

# Create a DataFrame of the NMF output
X_nmf_df = pd.DataFrame(X_nmf, index=X_numeric.index)

# import ace_tools as tools; tools.display_dataframe_to_user(name="NMF_Result", dataframe=X_nmf_df)

X_nmf_df.to_csv('/home/mpradhan/Intern_Research_Project/data/nnmf_output.csv', index=False)