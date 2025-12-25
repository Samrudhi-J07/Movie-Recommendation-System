from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].str.replace('|', ' ')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        user_genre = request.form["genre"]

        user_vec = tfidf.transform([user_genre])
        similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()

        movie_indices = similarity.argsort()[::-1][:10]
        recommendations = movies.iloc[movie_indices]["title"].tolist()

    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
