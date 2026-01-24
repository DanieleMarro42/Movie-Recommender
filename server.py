from flask import Flask, render_template, request, jsonify
from movie_recs import initialize_system, get_recommendations
import pandas as pd

app = Flask(__name__)

# Global variables to hold data and models
df = None
tfidf_matrix = None

print("Initializing Recommendation System...")
try:
    df, tfidf_matrix = initialize_system()
    print("System initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not initialize system. {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search_movie():
    """
    Search for movies matching the query.
    """
    query = request.args.get('query', '').lower()
    if not query or df is None:
        return jsonify([])
    
    # Simple substring match limit to 10 results
    matches = df[df['title'].str.lower().str.contains(query)].head(10)
    
    results = []
    for _, row in matches.iterrows():
        results.append({
            'title': row['title'],
            'cluster': int(row['cluster']),
            'poster_path': row['poster_path'] if 'poster_path' in row and pd.notna(row['poster_path']) else None
        })
        
    return jsonify(results)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Get recommendations for a specific movie title.
    """
    data = request.json
    title = data.get('title')
    
    if not title or df is None:
        return jsonify({'error': 'Invalid request or system not ready'}), 400
        
    # Find exact match first (case insensitive)
    matches = df[df['title'].str.lower() == title.lower()]
    
    if matches.empty:
        return jsonify({'error': 'Movie not found'}), 404
        
    original_title = matches.iloc[0]['title']
    cluster = int(matches.iloc[0]['cluster'])
    overview = matches.iloc[0]['overview']
    poster_path = matches.iloc[0]['poster_path'] if 'poster_path' in matches.columns and pd.notna(matches.iloc[0]['poster_path']) else None
    
    recs = get_recommendations(original_title, df, tfidf_matrix)
    
    rec_list = []
    if recs is not None:
        for _, row in recs.iterrows():
            rec_list.append({
                'title': row['title'],
                'cluster': int(row['cluster']),
                'overview': row['overview'],
                'poster_path': row['poster_path'] if 'poster_path' in row and pd.notna(row['poster_path']) else None
            })
            
    return jsonify({
        'title': original_title,
        'cluster': cluster,
        'overview': overview,
        'poster_path': poster_path,
        'recommendations': rec_list
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
