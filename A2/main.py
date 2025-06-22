import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


df = pd.read_csv('debates_2022.csv')
df['word_count'] = df['talk_text'].astype(str).apply(lambda x: len(x.split()))

############################################# TASK 1 #############################################
# Preprocessing
preprocessed_df = df[df['word_count'] >= 10].copy()

# Tf-idf Vectorization
custom_stop_words = list(text.ENGLISH_STOP_WORDS)
custom_stop_words.extend([
    'mr', 'mrs', 'madam', 'president', 'speaker', 'colleagues', 'dear', 'ms',
    'session', 'meeting', 'sitting', 'agenda', 'minutes', 'request', 'discussion', 'debate',
    'statement', 'statements', 'written', 'point', 'card', 'cards', 'blue', 'question', 'answer',
    'vote', 'voting', 'votes', 'resolution', 'resolutions', 'protocol', 'procedure', 'rules',
    'adopted', 'accordance', 'negotiations', 'committees', 'committee', 'report', 'reports',
    'results', 'submitted', 'published', 'approved', 'announcement', 'remind', 'inform',
    'european', 'union', 'parliament', 'europe', 'eu', 'member', 'members', 'states', 'council', 'behalf', 'group',
    '2022', '2021', 'november', 'june', 'september', 'march', 'thursday', 'tomorrow', 'today', 'years',
    'ended', 'held', 'place', 'available', 'enter', 'floor', 'time', 'minute', 'minutes', '12', '15', 'a9', '171',
    'think', 'like', 'thank', 'want', 'just', 'don', 'say', 'know', 'comments', 'speak',
    'follows', 'closed', 'article', 'protocol', 'request', 'shall', 'word'
])
vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words=custom_stop_words,
    max_df=0.85,
    min_df=5,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(preprocessed_df['talk_text'])
print(f"Shape of the TF-IDF matrix: {X.shape}")


############################################### TASK 2 #############################################
# Find best k
k_range = range(2, 21)  # Silhouette score is undefined for k=1
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Best k is {optimal_k} with a Silhouette Score of {max(silhouette_scores):.4f}")

# Apply K-Means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)


# Interpretation of the top ten most important terms in each cluster
print("Top 10 words in each cluster:")
feature_names = vectorizer.get_feature_names_out()
centroids = kmeans.cluster_centers_
for i in range(optimal_k):
    print(f"\n--- Top Terms for Cluster {i} ---")
    top_term_indices = centroids[i].argsort()[-10:][::-1]
    top_terms = [feature_names[index] for index in top_term_indices]
    print(", ".join(top_terms))


################################################# TASK 3 #############################################

# Initialize PCA to find the top 2 components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())

# Check how much variance the 2 components capture
explained_variance = pca.explained_variance_ratio_.sum()
print(f"Variance explained by the first 2 principal components: {explained_variance:.2%}")

#  Visualize the Clusters
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    X_pca[:, 0],  
    X_pca[:, 1],  
    c=cluster_labels, 
    cmap='tab20',   
    alpha=0.6,      
    s=10        
)

plt.title('2D PCA of Debate Speeches with K-Means Clusters', fontsize=18, pad=20)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
cbar = plt.colorbar(scatter, ticks=range(optimal_k))
cbar.set_label('Cluster Label', fontsize=12)
plt.savefig('debate_clusters.png', dpi=300, bbox_inches='tight')
