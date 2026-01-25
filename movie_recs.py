import pandas as pd
import kagglehub
import os
import glob
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel

# KDD Process Phase 1: Data Selection
# Goal: Identify and load the relevant data for analysis.
def load_data():
    """
    Downloads and loads the updated TMDB dataset (asaniczka/tmdb-movies-dataset-2023-930k-movies).
    Filters for top 10,000 popular movies to ensure performance.
    """
    print("Downloading updated dataset via kagglehub...")
    # [NEW] Major update to new daily-updated dataset
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    
    print("Path where dataset files are stored:", path)

    # Find the csv file
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
         raise FileNotFoundError("No CSV file found in downloaded dataset.")
             
    target_file = csv_files[0]
    print(f"Loading data from: {target_file}")
    
    # [OPTIMIZATION] Filter loading:
    # 1. We only need specific columns
    # 2. We filter for popularity/votes to keep the "best" 10k movies
    # This prevents memory issues with the 1M+ rows
    df = pd.read_csv(target_file)
    
    print(f"Total raw movies: {len(df)}")
    
    # Filter: Keep movies with at least 50 votes to remove noise
    df = df[df['vote_count'] > 50]
    
    # Sort by popularity and take top 10,000
    df = df.sort_values('popularity', ascending=False).head(10000)
    
    print(f"Filtered to top {len(df)} movies.")
    
    # Select columns
    # New dataset has 'genres' and 'keywords' as comma-separated strings already
    target_cols = ['id', 'title', 'overview', 'genres', 'keywords', 'poster_path']
    
    # Ensure all columns exist
    existing_cols = [c for c in target_cols if c in df.columns]
    df = df[existing_cols]
    
    
    # [FIX] Reset Index VITAL for alignment with TF-IDF MAtrix!
    # Without this, we use the original Index IDs (e.g. 825) to access matrix row 825,
    # which is WRONG if we filtered rows. We need row 8 matches item at index 8.
    df = df.reset_index(drop=True)
    
    # Data Cleaning
    df = df.dropna(subset=['overview', 'title'])
    
    # Ensure poster_path is valid string
    df['poster_path'] = df['poster_path'].fillna('')
    
    print(f"Final dataset size: {len(df)} movies.")
    return df

def parse_txt(text):
    """
    Parses comma-separated string (e.g. "Action, Drama") 
    Returns space-separated string (e.g. "Action Drama") for TF-IDF soup.
    """
    if not isinstance(text, str):
        return ""
    return text.replace(',', ' ')

def create_soup(row):
    """
    Combines overview, genres, and keywords into a single string.
    """
    return row['keywords_str'] + " " + row['genres_str'] + " " + row['cleaned_overview']

# KDD Process Phase 2: Preprocessing
# Goal: Clean and normalize text data for better analysis.
def preprocess_text(text):
    """
    NLP Pipeline:
    1. Lowercasing
    2. Tokenization
    3. Noise Removal (punctuation)
    4. Stop Words Removal
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # 3. Noise Removal: Keep only alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]
    
    # 4. Stop Words Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Reconstruction
    return " ".join(tokens)

# KDD Process Phase 3 & 4: Transformation & Data Mining
# Goal: Convert text to numbers and find hidden patterns (clusters).
def build_model(df):
    """
    Builds the TF-IDF matrix and K-Means clusters.
    """
    print("Transforming text data (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', min_df=3)
    tfidf_matrix = vectorizer.fit_transform(df['soup'])
    
    print("Mining data (K-Means Clustering)...")
    kmeans = KMeans(n_clusters=8, random_state=42)
    kmeans.fit(tfidf_matrix)
    
    return tfidf_matrix, kmeans, vectorizer

# KDD Process Phase 5: Pattern Evaluation
# Goal: Measure similarity between movies to provide recommendations.
def get_recommendations(title, df, tfidf_matrix):
    """
    Returns top 8 similar movies based on Cosine Similarity.
    """
    # Create a mapping of movie titles to indices
    # We strip and lower to ensure matches
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    
    if title.lower() not in indices:
        return None

    idx = indices[title.lower()]
    
    # Calculate Cosine Similarity
    cosine_scores = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    
    # Get the pair scores of all movies with that movie
    sim_scores = list(enumerate(cosine_scores))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 8 most similar movies (ignoring the first one which is itself)
    sim_scores = sim_scores[1:9]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 8 most similar movies
    return df.iloc[movie_indices]

def initialize_system():
    """
    Initializes the system by checking resources, loading data, and building models.
    Returns:
        tuple: (df, tfidf_matrix)
    """
    # Setup NLTK resources
    print("Checking NLTK resources...")
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        
    # 1. Initialize
    df = load_data()

    # Preprocess overviews
    print("Preprocessing text data...")
    df['cleaned_overview'] = df['overview'].apply(preprocess_text)
    
    # [IMPROVEMENT] Feature Engineering for better quality
    # New dataset has simple comma separated strings, so we use parse_txt instead of parse_json
    print("Extracting features (Genres & Keywords)...")
    df['genres_str'] = df['genres'].apply(parse_txt)
    df['keywords_str'] = df['keywords'].apply(parse_txt)
    
    print("Creating feature soup...")
    df['soup'] = df.apply(create_soup, axis=1)
    
    # Build models
    tfidf_matrix, kmeans, _ = build_model(df)
    
    # Assign clusters to DataFrame
    df['cluster'] = kmeans.labels_
    
    return df, tfidf_matrix

def main():
    try:
        df, tfidf_matrix = initialize_system()
    except Exception as e:
        print(f"Error initializing system: {e}")
        return

    print("\n--- System Ready ---")
    
    # 2. Interactive Loop
    while True:
        user_input = input("\nEnter a movie title (or 'q' to quit): ").strip()
        
        if user_input.lower() == 'q':
            print("Exiting. Goodbye!")
            break
            
        # Case-insensitive search match
        # We try to find the exact title first (case-insensitive)
        matches = df[df['title'].str.lower() == user_input.lower()]
        
        if matches.empty:
            print(f"Movie '{user_input}' not found. Please try another title.")
            continue
            
        # Get the 'official' title from the dataset
        original_title = matches.iloc[0]['title']
        movie_cluster = matches.iloc[0]['cluster']
        
        print(f"\nFound '{original_title}' in Cluster {movie_cluster}.")
        print("--- Recommendations based on Plot Analysis ---")
        
        recommendations = get_recommendations(original_title, df, tfidf_matrix)
        
        if recommendations is not None:
            # We enforce 1-indexing for the list display
            row_idx = 1
            for _, row in recommendations.iterrows():
                print(f"{row_idx}. {row['title']} (Cluster {row['cluster']})")
                row_idx += 1
        else:
            print("Could not generate recommendations.")

if __name__ == "__main__":
    main()

