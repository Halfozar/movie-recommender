#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------- Caching Data Loading & Processing -----------
@st.cache_data
def load_data():
    movies = pd.read_csv("C:\\Users\\HP\\Downloads\\Project\\tmdb_5000_movies.csv")
    credits = pd.read_csv("C:\\Users\\HP\\Downloads\\Project\\tmdb_5000_credits.csv")

    return movies, credits

@st.cache_data
def preprocess_data(movies_df, credits_df):
    # Merge datasets on title
    movies_df = movies_df.merge(credits_df[['title', 'cast']], on='title')

    # Parse columns
    def parse_column(col):
        return [i['name'] for i in ast.literal_eval(col)] if isinstance(col, str) else []

    movies_df['genres'] = movies_df['genres'].apply(parse_column)
    movies_df['keywords'] = movies_df['keywords'].apply(parse_column)
    movies_df['cast'] = movies_df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3] if isinstance(x, str) else [])
    movies_df['overview'] = movies_df['overview'].fillna('')

    # Combine text features
    movies_df['combined_features'] = movies_df['genres'] + movies_df['keywords'] + movies_df['cast']
    movies_df['combined_features'] = movies_df['combined_features'].apply(lambda x: ' '.join(x))
    movies_df['combined_features'] = movies_df['combined_features'] + ' ' + movies_df['overview']

    # Add lowercase title column for matching
    movies_df['title_lower'] = movies_df['title'].str.lower()

    # Vectorize and compute cosine similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies_df, cosine_sim

# ----------- Recommendation Logic -----------
def recommend_movie(title, df, sim_matrix):
    title = title.lower()
    if title not in df['title_lower'].values:
        return ["Movie not found."]

    idx = df[df['title_lower'] == title].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# ----------- Streamlit UI -----------
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommendation System")

# Load and preprocess data
movies_df, credits_df = load_data()
processed_df, cosine_sim = preprocess_data(movies_df, credits_df)

# Dropdown for movie selection
movie_name = st.selectbox("Choose a movie to get similar recommendations:", sorted(processed_df['title'].tolist()))

# Show recommendations
if movie_name:
    results = recommend_movie(movie_name, processed_df, cosine_sim)
    if results == ["Movie not found."]:
        st.error("🚫 Movie not found. Please try another title.")
    else:
        st.success("### 🎯 Recommended Movies:")
        for movie in results:
            st.write(f"🎥 {movie}")


# In[ ]:





# In[ ]:




