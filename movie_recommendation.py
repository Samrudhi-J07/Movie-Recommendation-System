import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print(movies.head())
print(ratings.head())

# Clean genres
movies['genres'] = movies['genres'].str.replace('|', ' ')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

print(tfidf_matrix.shape)

# Recommendation function (memory-safe)
def recommend_movie(movie_name, top_n=5):
    if movie_name not in movies['title'].values:
        print("Movie not found!")
        return

    index = movies[movies['title'] == movie_name].index[0]

    similarity_scores = cosine_similarity(
        tfidf_matrix[index], tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]

    print("\nRecommended Movies:")
    for i in similar_indices:
        print(movies.iloc[i]['title'])

# CALL THE FUNCTION (IMPORTANT!)
recommend_movie("Toy Story (1995)")
