import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- 1. Reproduce Features and Clusters from Previous Tasks ---
print("--- 1. Reproducing features and cluster assignments ---")

# This section is the same as before to ensure we have the necessary data
df = pd.read_csv('debates_2022.csv')
df['word_count'] = df['talk_text'].astype(str).apply(lambda x: len(x.split()))
preprocessed_df = df[df['word_count'] >= 10].copy()
custom_stop_words = list(text.ENGLISH_STOP_WORDS)
custom_stop_words.extend(['mr', 'mrs', 'madam', 'president'])
vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words=custom_stop_words,
    max_df=0.90,
    min_df=5,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(preprocessed_df['talk_text'])

# Rerun clustering with k=19 to get the labels
optimal_k = 19
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)

print("Features and clusters are ready.")


# --- 2. Perform Dimensionality Reduction with PCA ---
print("\n--- 2. Reducing dimensionality from 2000 to 2 with PCA ---")

# Initialize PCA to find the top 2 components
pca = PCA(n_components=2, random_state=42)

# Fit PCA on the TF-IDF data and transform it into 2D space
X_pca = pca.fit_transform(X.toarray()) # .toarray() is used as PCA works on dense matrices

print(f"Original feature shape: {X.shape}")
print(f"Reduced feature shape: {X_pca.shape}")

# Check how much variance the 2 components capture
explained_variance = pca.explained_variance_ratio_.sum()
print(f"Variance explained by the first 2 principal components: {explained_variance:.2%}")
print("Note: A low variance is expected for complex text data.")


# --- 3. Visualize the Clusters in 2D PCA Space ---
print("\n--- 3. Generating 2D scatter plot ---")

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 10))

# Create the scatter plot
scatter = plt.scatter(
    X_pca[:, 0],      # X-axis: Principal Component 1
    X_pca[:, 1],      # Y-axis: Principal Component 2
    c=cluster_labels, # Color by the cluster label
    cmap='tab20',     # A colormap with 20 distinct colors, suitable for our 19 clusters
    alpha=0.6,        # Set transparency to see overlapping points
    s=10              # Set marker size
)

# Add plot titles and labels
plt.title('2D PCA of Debate Speeches with K-Means Clusters', fontsize=18, pad=20)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# Add a color bar to show which color corresponds to which cluster
# The color bar will have ticks representing each cluster number from 0 to 18
cbar = plt.colorbar(scatter, ticks=range(optimal_k))
cbar.set_label('Cluster Label', fontsize=12)

plt.show()