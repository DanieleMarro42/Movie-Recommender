# 🎬 CineCluster: Content-Based Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0-green?style=for-the-badge&logo=flask&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**CineCluster** is a sophisticated movie recommendation engine developed as the final project for the **"Data Mining & Text Analytics"** university course. It leverages Natural Language Processing (NLP) and Machine Learning techniques to analyze movie plots and provide personalized suggestions.

---

## 📖 Abstract

In the era of information overload, finding the right movie can be challenging. CineCluster addresses this by implementing a **Content-Based Filtering** approach. Unlike collaborative filtering which relies on user history, this system analyzes the *content* of the movies themselves (plots, genres, keywords).

The core of the project demonstrates the **Knowledge Discovery in Databases (KDD)** process:
1.  **Data Selection**: Utilizing a high-quality dataset of 1,000,000+ movies (filtered to the top 10,000 for performance).
2.  **Preprocessing**: Advanced text cleaning (Tokenization, Stop-word removal).
3.  **Transformation**: `TF-IDF` Vectorization to convert text into mathematical space.
4.  **Data Mining**: `K-Means Clustering` to discover hidden genre groups.
5.  **Pattern Evaluation**: `Cosine Similarity` to rank and recommend relevant titles.

---

## ✨ Key Features

-   **🔍 Intelligent Search**: Real-time typeahead search with poster thumbnails.
-   **🧠 Powered Recommendations**: Analyzes the "DNA" of a movie (Plot + Genres + Keywords) to find its closest matches.
-   **📊 Cluster Analysis**: Automatically groups movies into 8 distinct thematic clusters (e.g., "Family & Fantasy", "Horror & Thriller").
-   **🖼️ Rich Visuals**: Fetches high-quality movie posters from TMDB for an immersive experience.
-   **⚡ High Performance**: optimized to load and query 10,000 movies primarily in-memory using Pandas and Scikit-Learn.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend** | Flask | Server logic and API endpoints. |
| **Data Processing** | Pandas | Data manipulation and filtering. |
| **Machine Learning** | Scikit-Learn | TF-IDF Vectorization, K-Means, Cosine Similarity. |
| **NLP** | NLTK | Text tokenization and cleaning. |
| **Frontend** | HTML5, CSS3, JS | Responsive UI with glassmorphism design. |
| **Data Source** | Kaggle (Asaniczka) | Daily updated TMDB dataset. |

---

## ⚙️ Installation & Usage

### Prerequisites
-   Python 3.8 or higher.
-   Internet connection (for initial data download).

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/CineCluster.git
    cd CineCluster
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python server.py
    ```
    *Note: The first run may take a few moments to download the dataset (~100MB) via KaggleHub.*

4.  **Access the Interface**
    Open your browser and navigate to:
    `http://localhost:5001`

---

## 🧠 Methodology Details

### 1. Feature Engineering ("The Soup")
To improve accuracy, we don't just look at the plot. We create a "metadata soup" for each movie:
```python
soup = keywords + genres + overview
```
This ensures that a movie like *Toy Story* matches other movies that are both "Animation" (Genre) and about "Toys" (Keyword/Plot).

### 2. TF-IDF Vectorization
We use **Term Frequency-Inverse Document Frequency** to convert this "soup" into a matrix.
-   **TF**: How often a word appears in a movie.
-   **IDF**: Reduces the weight of common words (like "movie", "film").
-   Result: A vector space where each movie is a point.

### 3. Cosine Similarity
To recommend movies, we calculate the angle between these points.
-   **Small angle (High Similarity)**: Movies share many unique keywords.
-   **Large angle (Low Similarity)**: Movies are unrelated.

---

## 📂 Project Structure

```
CineCluster/
├── movie_recs.py       # Core Logic (Data Loading, ML Models)
├── server.py           # Flask Web Server
├── requirements.txt    # Dimensions dependencies
├── static/
│   ├── style.css       # Custom styling
│   └── script.js       # Frontend logic (Search, API calls)
├── templates/
│   └── index.html      # Main user interface
└── README.md           # Project Documentation
```

---

## 👥 Credits & Acknowledgements

-   **Course**: Data Mining & Text Analytics, IULM University (2025-2026).
-   **Student**: Diego Casati, Enrico Forneris, Daniele Marro
-   **Data Provider**: [TMDB Movies Dataset 2024 (Asaniczka)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)

---

*For educational purposes only.*


