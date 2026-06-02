# Βιβλιοθήκες
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer #Μετατρέπει κείμενο σε αριθμητικά vectors
from sklearn.metrics.pairwise import cosine_similarity

# Χρόνος εκτέλεσης για το content-based σύστημα
content_time = 0
# Αποθήκευση χρόνου για το Personalized Recommender
personalized_time = 0

# Φάκελος project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Αρχεία dataset

movies_path = os.path.join(
    BASE_DIR,
    "ml-latest",
    "movies.csv"
)

ratings_path = os.path.join(
    BASE_DIR,
    "ml-latest",
    "ratings.csv"
)

tags_path = os.path.join(
    BASE_DIR,
    "ml-latest",
    "tags.csv"
)

# =========================
# LOAD DATASETS
# =========================
# Φόρτωση datasets
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)
tags = pd.read_csv(tags_path)

print("Datasets loaded successfully!")

# =========================
# SHOW DATA
# =========================
# Εμφάνιση δεδομένων
print("\nMovies Dataset:\n")
print(movies.head())

print("\nRatings Dataset:\n")
print(ratings.head())

# =========================
# CLEAN DATA
# =========================
# Καθαρισμός δεδομένων
movies['genres'] = movies['genres'].fillna('')

# =========================
# PROCESS TAGS
# =========================
# Καθαρισμός δεδομένων
tags = tags.groupby(
    'movieId'
)['tag'].apply(
    lambda x: ' '.join(
        x.dropna().astype(str)
    )
).reset_index()

#ένωση tags με movies dataset
movies = movies.merge(
    tags,
    on='movieId',
    how='left'
)

movies['tag'] = movies['tag'].fillna('')

# =========================
# COMBINE FEATURES
# =========================
# Συνδυασμός χαρακτηριστικών
movies['features'] = (
    movies['genres']
    + ' '
    + movies['tag']
)

# =========================
# TF-IDF VECTORIZATION
# =========================
# Δημιουργία TF-IDF vectors
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=10000
)

vectors = tfidf.fit_transform(
    movies['features']
)

# =========================
# TOP RATED MOVIES
# =========================
# Υπολογισμός ratings ταινιών
movie_stats = ratings.groupby('movieId').agg({
    'rating': ['mean', 'count']
})

movie_stats.columns = [
    'avg_rating',
    'rating_count'
]

movie_stats = movie_stats.reset_index()

top_rated = movies.merge(
    movie_stats,
    on='movieId'
)

# Ταινίες με αρκετές αξιολογήσεις
top_rated = top_rated[
    top_rated['rating_count'] > 100
]

# Ταξινόμηση με βάση το rating
top_rated = top_rated.sort_values(
    by='avg_rating',
    ascending=False
)

print("\n" + "=" * 50)
print("TOP RATED MOVIES")
print("=" * 50)

print(
    top_rated[
        ['title', 'avg_rating', 'rating_count']
    ].head(10)
)

# =========================
# DATASET STATISTICS
# =========================
# Βασικά στατιστικά δεδομένων
total_movies = movies.shape[0]
total_ratings = ratings.shape[0]
total_users = ratings['userId'].nunique()
average_rating = ratings['rating'].mean()

print("\nDataset Statistics:\n")

print(f"Total Movies: {total_movies}")
print(f"Total Ratings: {total_ratings}")
print(f"Total Users: {total_users}")
print(f"Average Rating: {average_rating:.2f}")

# =========================
# TOP GENRES ANALYSIS
# =========================
# Υπολογισμός genres
genre_counts = {}

for genres in movies['genres']:

    split_genres = genres.split('|')

    for genre in split_genres:

        if genre in genre_counts:

            genre_counts[genre] += 1

        else:

            genre_counts[genre] = 1

# Μετατροπή σε DataFrame
genre_df = pd.DataFrame(
    genre_counts.items(),
    columns=['Genre', 'Count']
)

# Ταξινόμηση genres
genre_df = genre_df.sort_values(
    by='Count',
    ascending=False
)

# top 10 genres
top_genres = genre_df.head(10)

# =========================
# PLOT TOP GENRES
# =========================
# Γράφημα top genres
plt.figure(figsize=(10, 6))

plt.bar(
    top_genres['Genre'],
    top_genres['Count'],
    color='#ce2feb'
)

plt.title('Top 10 Movie Genres')

plt.xlabel('Genre')

plt.ylabel('Number of Movies')

plt.grid(axis='y')

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# =========================
# RATING DISTRIBUTION
# =========================
# Γράφημα κατανομής ratings
plt.figure(figsize=(8, 5))

plt.hist(
    ratings['rating'],
    bins=10,
    color="#ef139b"
)

plt.title('Rating Distribution')

plt.xlabel('Rating')

plt.ylabel('Frequency')

plt.tight_layout()

plt.show()

# =========================
# CONTENT-BASED FILTERING
# =========================

def recommend(movie_title):
    # Έναρξη μέτρησης χρόνου
    start_time = time.time()
    
 # Αναζήτηση ταινίας
    matching_movies = movies[
        movies['title'].str.contains(
            movie_title,
            case=False,
            na=False
        )
    ]

    # Έλεγχος αν υπάρχει η ταινία
    if matching_movies.empty:

        print("\nMovie not found.")
        return

    movie_index = matching_movies.index[0]

    selected_movie = movies.iloc[movie_index].title

    print(f"\nSelected Movie: {selected_movie}")

    # Υπολογισμός ομοιότητας
    similarity_scores = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    # Ταξινόμηση προτάσεων
    similar_movies = sorted(
        list(enumerate(similarity_scores)),
        reverse=True,
        key=lambda x: x[1]
    )

    print("\nContent-Based Recommendations:\n")

    count = 0

    for movie in similar_movies[1:]:

        title = movies.iloc[movie[0]].title
        movie_id = movies.iloc[movie[0]].movieId

        movie_rating_data = movie_stats[
            movie_stats['movieId'] == movie_id
        ]

        if movie_rating_data.empty:
            continue

        rating_count = movie_rating_data[
            'rating_count'
        ].values[0]

        avg_rating = movie_rating_data[
            'avg_rating'
        ].values[0]

        # Φιλτράρισμα αδύναμων ταινιών
        if rating_count < 500:
            continue

        if avg_rating < 3.5:
            continue

        print(title)

        count += 1

        if count == 5:
            break
        
    # Υπολογισμός χρόνου εκτέλεσης
    execution_time = time.time() - start_time

    global content_time

    content_time = execution_time

    print(
        f"\nExecution Time: {execution_time:.4f} seconds"
    )

# =========================
# PERSONALIZED RECOMMENDER
# =========================

def personalized_recommendations(user_preferences):
# Έναρξη μέτρησης χρόνου
    start_time = time.time()

    liked_movies = []

    # Αναζήτηση αγαπημένων ταινιών
    for movie, rating in user_preferences.items():

        if rating >= 4:

            matching_movies = movies[
                movies['title'].str.contains(
                    movie,
                    case=False,
                    na=False
                )
            ]

            if not matching_movies.empty:

                liked_movies.append(
                    matching_movies.index[0]
                )
    # Έλεγχος αν βρέθηκαν ταινίες
    if len(liked_movies) == 0:

        print("\nNo highly rated movies found.")
        return

    # Υπολογισμός μέσης ομοιότητας
    similarity_scores = np.zeros(
        vectors.shape[0]
    )

    for movie_index in liked_movies:

        scores = cosine_similarity(
            vectors[movie_index],
            vectors
        ).flatten()

        similarity_scores += scores

    similarity_scores = (
        similarity_scores / len(liked_movies)
    )

    # Ταξινόμηση προτάσεων
    recommended = sorted(
        list(enumerate(similarity_scores)),
        reverse=True,
        key=lambda x: x[1]
    )

    print("\nPersonalized Recommendations:\n")

    count = 0

    for movie in recommended:

        title = movies.iloc[movie[0]].title
        movie_id = movies.iloc[movie[0]].movieId

        # Παράλειψη ήδη επιλεγμένων ταινιών
        already_rated = False

        for rated_movie in user_preferences.keys():

            if rated_movie.lower() in title.lower():

                already_rated = True

        if already_rated:
            continue

       # Έλεγχος ratings
        movie_rating_data = movie_stats[
            movie_stats['movieId'] == movie_id
        ]

        if movie_rating_data.empty:
            continue

        rating_count = movie_rating_data[
            'rating_count'
        ].values[0]

        avg_rating = movie_rating_data[
            'avg_rating'
        ].values[0]

        # Φιλτράρισμα αδύναμων ταινιών
        if rating_count < 500:
            continue

        if avg_rating < 3.5:
            continue

        print(f"{count}. {title}")

        count += 1

        if count == 10:
            break
    
    # Υπολογισμός χρόνου εκτέλεσης
    execution_time = time.time() - start_time

    global personalized_time

    personalized_time = execution_time

    print(
        f"\nExecution Time: {execution_time:.4f} seconds"
    )

# =========================
# CONTENT-BASED INPUT
# =========================
# Είσοδος ταινίας από χρήστη
movie_name = input(
    "\nEnter a movie name: "
)
# Εκτέλεση content-based συστήματος
recommend(movie_name)

# =========================
# PERSONALIZED INPUT
# =========================
# Εισαγωγή αγαπημένων ταινιών
favorite1 = input(
    "\nEnter favorite movie 1: "
)

favorite2 = input(
    "Enter favorite movie 2: "
)

favorite3 = input(
    "Enter favorite movie 3: "
)

# Προτιμήσεις χρήστη
user_preferences = {
    favorite1: 5,
    favorite2: 5,
    favorite3: 5
}

# =========================
# RUN PERSONALIZED SYSTEM
# =========================
# Εκτέλεση personalized συστήματος
personalized_recommendations(
    user_preferences
)

# =========================
# PERFORMANCE COMPARISON
# =========================
# Σύγκριση χρόνου εκτέλεσης
methods = [
    'Content-Based',
    'Personalized'
]

times = [
    content_time,
    personalized_time
]

# Γράφημα χρόνου εκτέλεσης
plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    times,
    color=['#55f265', '#f28f55']
)


plt.title(
    'Recommendation Method Execution Time'
)

plt.ylabel(
    'Execution Time (seconds)'
)
plt.grid(axis='y')

plt.tight_layout()

plt.show()