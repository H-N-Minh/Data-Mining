import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 

print("--- Loading and Preprocessing Data ---")

# Load the dataset
df = pd.read_csv('debates_2022.csv')

# --- 1. Preprocessing ---
# Based on the analysis, we remove talks with fewer than 10 words.

# Calculate word count to filter based on it
df['word_count'] = df['talk_text'].astype(str).apply(lambda x: len(x.split()))

# Keep only the rows where word count is 10 or more
initial_rows = len(df)
preprocessed_df = df[df['word_count'] >= 10].copy()
removed_rows = initial_rows - len(preprocessed_df)

print(f"Original number of talks: {initial_rows}")
print(f"Removed {removed_rows} talks with fewer than 10 words.")
print(f"Number of talks to be vectorized: {len(preprocessed_df)}")


# --- 2. Feature Extraction with TfIdfVectorizer ---
print("\n--- Extracting Features using TfIdfVectorizer ---")

# The analysis showed that words like 'president', 'mr', etc., are very frequent.
# We will add them to the standard English stop words list.
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

# Fit the vectorizer to the data and transform the text into a feature matrix
X = vectorizer.fit_transform(preprocessed_df['talk_text'])

# --- 3. Display Results ---
print("\n--- Results ---")
print(f"Shape of the TF-IDF feature matrix: {X.shape}")
print(f"Successfully extracted {X.shape[1]} features from {X.shape[0]} documents.")

# To see what kind of features were created, we can print a few
feature_names = vectorizer.get_feature_names_out()
print("\nFirst 20 example features extracted:")
print(feature_names[:20])

print("\nLast 20 example features extracted:")
print(feature_names[-20:])