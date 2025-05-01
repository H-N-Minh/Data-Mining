import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests
import random
import warnings

# Load the dataset (replace 'dataset.csv' with your file path)
df = pd.read_csv('task2-dataset.csv', index_col=0)
print(f"Dataset shape: {df.shape}")

########################### Compute Pearson Correlation Matrix
pearson_corr_matrix = df.corr(method='pearson')

# Find Top 20 Pearson Correlations
pearson_strong_corrs = (
    pearson_corr_matrix
    .where(pearson_corr_matrix.abs() > 0.5)
    .where(pearson_corr_matrix != 1.0)
    .stack()
)
pearson_strong_corrs = pearson_strong_corrs.sort_values(ascending=False)
print("\nTop 20 Strong Pearson Correlations:")
unique_pairs = []
for (var1, var2), corr in pearson_strong_corrs[:20].items():
    print(f"{var1} to {var2}: Linear correlation = {corr:.4f}")
    if (var2, var1) not in unique_pairs:
        unique_pairs.append((var1, var2))



############################## Compute Spearman Correlation Matrix
spearman_corr_matrix = df.corr(method='spearman')

# Find Top 20 Spearman Correlations
spearman_strong_corrs = (
    spearman_corr_matrix
    .where(spearman_corr_matrix.abs() > 0.5)
    .where(spearman_corr_matrix != 1.0)
    .stack()
)
spearman_strong_corrs = spearman_strong_corrs.sort_values(ascending=False)
print("\nTop 20 Strong Spearman Correlations:")
for (var1, var2), corr in spearman_strong_corrs[:20].items():
    print(f"{var1} to {var2}: Monotonic correlation = {corr:.4f}")



############################### Compute Mutual Information
# Scale data for Mutual Information
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Sample 500 random pairs for MI computation
all_pairs = [(var1, var2) for i, var1 in enumerate(df.columns) for var2 in df.columns[i+1:]]
sampled_pairs = random.sample(all_pairs, min(500, len(all_pairs)))

mi_matrix = np.zeros((len(df.columns), len(df.columns)))
for var1, var2 in sampled_pairs:
    i, j = df.columns.get_loc(var1), df.columns.get_loc(var2)
    mi = mutual_info_regression(df_scaled[[var1]], df_scaled[var2])[0]
    mi_matrix[i, j] = mi
    mi_matrix[j, i] = mi

# Find Top 10 MI Pairs
mi_flat = [
    (df.columns[i], df.columns[j], mi_matrix[i, j])
    for i in range(len(df.columns))
    for j in range(i + 1, len(df.columns))
    if mi_matrix[i, j] > 0
]
mi_flat.sort(key=lambda x: x[2], reverse=True)
print("\nTop 10 Mutual Information Pairs:")
for var1, var2, mi in mi_flat[:10]:
    print(f"{var1} to {var2}: MI = {mi:.4f}")




################################ CAUSALITY TESTS ################################
#  Granger Causality Tests for Top 10 Unique Pearson Correlation Pairs
print("\nGranger Causality Tests for Top 20 Correlated Pairs:")
significant_causality = []
for var1, var2 in unique_pairs[:20]:
    if var1 in df.columns and var2 in df.columns:
        pair_data = df[[var2, var1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results = grangercausalitytests(pair_data, maxlag=3, verbose=False)
        for lag, result in results.items():
            p_value = result[0]['ssr_ftest'][1]
            if p_value < 0.05:
                significant_causality.append("msg")
        pair_data = df[[var1, var2]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results = grangercausalitytests(pair_data, maxlag=3, verbose=False)
        for lag, result in results.items():
            p_value = result[0]['ssr_ftest'][1]
            if p_value < 0.05:
                msg = f"{var2} may Granger-cause {var1} at lag {lag} (p-value = {p_value:.4f})"
                significant_causality.append("msg")


# Print Significant Causal Relationships
if significant_causality:
    print("\nSignificant Causal Relationships Found:")
    for causal in significant_causality:
        print(causal)
else:
    print("\nNo significant causal relationships found (all p-values > 0.05).")