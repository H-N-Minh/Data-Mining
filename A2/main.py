import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- 1. Reproduce Feature Extraction (from the previous task) ---
print("--- 1. Loading Data and Extracting Features ---")
df = pd.read_csv('debates_2022.csv')

# Preprocessing: Remove short talks
df['word_count'] = df['talk_text'].astype(str).apply(lambda x: len(x.split()))
preprocessed_df = df[df['word_count'] >= 10].copy()

# Define custom stop words
custom_stop_words = list(text.ENGLISH_STOP_WORDS)
custom_stop_words.extend(['mr', 'mrs', 'madam', 'president'])


# Initialize the vectorizer with the tuned parameters
vectorizer = TfidfVectorizer(
    max_features=2000,          # Limit to the top 2000 features
    stop_words=custom_stop_words, # Use our custom stop word list
    max_df=0.90,                # Ignore terms that appear in > 90% of documents
    min_df=5,                   # Ignore terms that appear in < 5 documents
    ngram_range=(1, 2)          # Include both words and two-word phrases
    # lowercase=True is the default and does not need to be set
)

X = vectorizer.fit_transform(preprocessed_df['talk_text'])
print(f"Feature matrix created with shape: {X.shape}")


# --- 2. Determine the Optimal Number of Clusters (k) ---
print("\n--- 2. Analyzing for Optimal 'k' (2 to 20 clusters) ---")
print("This may take a few minutes...")

# We will test a range of k values
k_range = range(2, 21) 
inertia_values = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    
    # Calculate inertia (for Elbow Method)
    inertia_values.append(kmeans.inertia_)
    
    # Calculate Silhouette Score
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"For k={k}, Silhouette Score = {score:.3f}, Inertia = {kmeans.inertia_:.2f}")

# --- 3. Plot the Results for Analysis ---
print("\n--- 3. Plotting Analysis Graphs ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Plot the Elbow Method
ax1.plot(k_range, inertia_values, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (WCSS)')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True)

# Plot the Silhouette Scores
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score for Optimal k')
ax2.grid(True)

plt.suptitle('K-Means Clustering Analysis', fontsize=16)
plt.show()

print("\n--- 4. Next Steps ---")
print("Please inspect the generated plots to decide on the best number of clusters 'k'.")
print("Look for the 'elbow' in the first plot and the peak in the second plot.")
print("Once you provide the chosen 'k', I can run the final clustering model and interpret the results.")


