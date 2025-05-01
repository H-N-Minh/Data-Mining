import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv('task4-dataset.csv')
# preparing the dataset (removing the first column and replacing missing values)
df = df.iloc[:, 1:]
df.replace(['N/A', 'unkwn', ''], np.nan, inplace=True)


# TASK a)

numeric_cols = ['semester', 'height', 'age', 'english_skills', 'books_per_year']
categorical_cols = ['gender', 'study', 'likes_pinapple_on_pizza', 'likes_chocolate']

print("\n\n##################### TASK A)#####################\n")
print("\nPearson correlation:")
corr_matrix = df[numeric_cols].corr(method='pearson')


# Get the top 3 highest absolute correlations (excluding self-correlation)
print("Top 3 pairs with highest absolute Pearson correlation:")
corr_pairs = corr_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1]  # Exclude self-correlation
high_corr = corr_pairs.sort_values(ascending=False).drop_duplicates()
top3 = high_corr.head(3)
for (col1, col2), value in top3.items():
    print(f"{col1} - {col2}: {corr_matrix.loc[col1, col2]:.3f}")

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Mutual information scores
# Compute mutual information for all features
df_mi = df.copy()

# Impute missing values (only using mean for now, later in D we will use regression)
for col in numeric_cols:
    df_mi[col] = df_mi[col].fillna(df_mi[col].mean())
for col in categorical_cols:
    df_mi[col] = df_mi[col].fillna(df_mi[col].mode()[0])

mi_scores = {}
for col in df_mi.columns:
    # Ensure target is numeric for regression or categorical for classification
    X = df_mi.drop(col, axis=1)
    y = df_mi[col]
    # Convert categorical columns to numeric codes if necessary
    if col in categorical_cols:
        y = y.astype('category').cat.codes
        mi = mutual_info_classif(X, y, random_state=42)
    else:
        mi = mutual_info_regression(X, y, random_state=42)
    mi_scores[col] = mi

print("\nMutual Information Scores: (only showing scores >= 0.1 to ignore weak relationships)")
columns = df_mi.columns.tolist()
for target, mi_arr in mi_scores.items():
    for i, mi in enumerate(mi_arr):
        feature = [col for col in columns if col != target][i]
        if mi >= 0.1:
            print(f"{feature} -> {target}: {mi:.4f}")



# TASK b)
print("\n\n##################### TASK b)#####################\n")
# Calculate percentage of missing values per column
print("Percentage of Missing Values for all collumns that has at least 1 missing value:")
missing_percent = df.isna().mean() * 100
for col, percent in missing_percent.items():
    if percent > 0: # Only print columns with missing values
        print(f"{col}: {percent:.2f}%")

# now we check if missingness is correlated with other features
# The print above shows 4 columns with missing values: height, likes_pinapple_on_pizza, likes_chocolate, and english_skills.
# we create new columns to indicate missingness for these features, so we can analyze their correlations with other features.
for col in ['height', 'likes_pinapple_on_pizza', 'likes_chocolate', 'english_skills']:
    df[f'missing_{col}'] = df[col].isna().astype(int)

# Analyze correlations between missingness indicators and numeric features

numeric_cols = ['semester', 'age', 'books_per_year']
print("\nCorrelations of Missingness Indicators with Numeric Features: (only showing correlations > 0.1 or < -0.1)")
for col in ['missing_height', 'missing_likes_pinapple_on_pizza', 'missing_likes_chocolate', 'missing_english_skills']:
    for num_col in numeric_cols:
        corr = df[col].corr(df[num_col])
        if abs(corr) > 0.1:
            print(f"{num_col} in correlation with {col}: {corr:.3f}")

# Analyze correlations between missingness indicators and  categorical/binary features
categorical_cols = ['gender', 'study']
print("\nCorrelations of Missingness Indicators with Categorical Features: (percentage of missing values for each category of each feature)")
for col in ['height', 'likes_pinapple_on_pizza', 'likes_chocolate', 'english_skills']:
    print(f"\nMissingness of {col}:")
    for cat_col in categorical_cols:
        missing_by_cat = df.groupby(cat_col)[col].apply(lambda x: x.isna().mean() * 100)
        print(f" - in feature {missing_by_cat}")

# clean up the dataset by removing the missingness indicators
df.drop(columns=['missing_height', 'missing_likes_pinapple_on_pizza', 'missing_likes_chocolate', 'missing_english_skills'], inplace=True)



print("\n\n##################### TASK d) #####################\n")
# Define columns
numeric_cols = ['semester', 'height', 'age', 'english_skills', 'books_per_year']
categorical_cols = ['gender', 'study', 'likes_pinapple_on_pizza', 'likes_chocolate']
cols_with_missing = ['height', 'likes_pinapple_on_pizza', 'likes_chocolate', 'english_skills']

# TASK d)
# Convert  columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce') 

# Compute means before imputation
print("Means Before Imputation:")
for col in cols_with_missing:
    mean = df[col].mean()
    print(f"{col}: {mean:.4f}" if not pd.isna(mean) else f"{col}: No valid data")

# start imputation:
df_imputed = df.copy()

# Height: Regression imputation using gender and age
mask_height = df_imputed['height'].notna()
X_height = df_imputed.loc[mask_height, ['gender', 'age']]
y_height = df_imputed.loc[mask_height, 'height']
height_reg = LinearRegression().fit(X_height, y_height)
missing_height = df_imputed['height'].isna()
X_missing_height = df_imputed.loc[missing_height, ['gender', 'age']]
df_imputed.loc[missing_height, 'height'] = height_reg.predict(X_missing_height)

# Likes_pinapple_on_pizza: Mode imputation
df_imputed['likes_pinapple_on_pizza'] = df_imputed['likes_pinapple_on_pizza'].fillna(df['likes_pinapple_on_pizza'].mode()[0])

# Likes_chocolate: Mode imputation
df_imputed['likes_chocolate'] = df_imputed['likes_chocolate'].fillna(df['likes_chocolate'].mode()[0])

# English_skills: Regression imputation using only study
mask_english = df_imputed['english_skills'].notna()
X_english = df_imputed.loc[mask_english, ['study']]
y_english = df_imputed.loc[mask_english, 'english_skills']
english_reg = LinearRegression().fit(X_english, y_english)
missing_english = df_imputed['english_skills'].isna()
X_missing_english = df_imputed.loc[missing_english, ['study']]
df_imputed.loc[missing_english, 'english_skills'] = english_reg.predict(X_missing_english)

# Compute means after imputation
print("\nMeans After Imputation:")
for col in cols_with_missing:
    mean = df_imputed[col].mean()
    print(f"{col}: {mean:.4f}" if not pd.isna(mean) else f"{col}: No valid data")
