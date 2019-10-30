from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt')

from nltk.stem.snowball import SnowballStemmer

# Create an English language SnowballStemmer object
stemmer = SnowballStemmer("english")

np.random.seed(5)

movies_df = pd.read_csv("movies.csv")

movies_df["plot"] = movies_df["wiki_plot"].astype(str) + "\n" + movies_df["imdb_plot"].astype(str)


def tokenize_and_stem(text):
    # Tokenize by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize(
        text) for word in nltk.word_tokenize(sent)]
    # Filter out raw tokens to remove noise
    filtered_tokens = [
        token for token in tokens if re.search('[a-zA-Z]', token)]
    # Stem the filtered_tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.2, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem,
                                   ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["plot"]])

similarity = cosine_similarity(tfidf_matrix)

app = Flask(__name__)


@app.route('/get_similar/<movie>', methods=['GET'])
def get_similar(movie):
    global movies_df, tfidf_matrix, similarity

    index = movies_df[movies_df.title == movie]["rank"].values[0]

    similar_movies = list(enumerate(similarity[index]))

    sorted_similar_movies = sorted(
        similar_movies, key=lambda x: x[1], reverse=True)

    recommendations = []

    for element in sorted_similar_movies[1:11]:
        recommendations.append(movies_df["title"][element[0]])

    response = {'result': recommendations}

    return jsonify(response)


@app.route('/list_all', methods=['GET'])
def list_all():
    global movies_df

    res = []

    for i in movies_df["title"]:
        res.append(i)

    response = {'result': res}

    return jsonify(response)


@app.route('/')
def index():
    return "Welcome to Binge Watch! We waste your time beautifully!"

if __name__ == '__main__':
    app.run()
