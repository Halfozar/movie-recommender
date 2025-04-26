Recommends movies based on similarity in genres, cast, keywords, and overview.

Uses TF-IDF vectorization and cosine similarity to measure relevance.

Clean, interactive web interface using Streamlit.

Processes and merges metadata from TMDB dataset.

 How It Works
Data Loading
Loads two datasets: tmdb_5000_movies.csv and tmdb_5000_credits.csv.

Data Preprocessing

Merges datasets on movie title.

Parses JSON-like strings in genres, cast, and keywords columns.

Combines parsed text with movie overview.

Applies TF-IDF vectorization to convert text into numeric form.

Computes cosine similarity between movies.

Recommendation Engine

When a user selects a movie, the system finds the most similar movies using cosine similarity.

User Interface

Built with Streamlit for easy interaction.

Dropdown menu to choose a movie and display top 5 recommendations.



 Example
If you select "The Dark Knight", the system might recommend similar action-packed or crime-themed movies like:

Batman Begins

The Dark Knight Rises

Inception

Iron Man

Man of Steel





