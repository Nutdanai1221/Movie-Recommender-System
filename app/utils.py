import pandas as pd
import numpy as np
import config

def process_data(data_path) :
    dataframe = pd.read_csv(data_path)
    user_ids = dataframe["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie_ids = dataframe["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    dataframe["user"] = dataframe["userId"].map(user2user_encoded)
    dataframe["movie"] = dataframe["movieId"].map(movie2movie_encoded)
    dataframe["rating"] = dataframe["rating"].values.astype(np.float32)
    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    return dataframe, num_users, num_movies, movie2movie_encoded, user2user_encoded, movie_encoded2movie

def get_recomendation(model, rating_df, movie_df, user_id, movie2movie_encoded, user2user_encoded, movie_encoded2movie) : 
    movies_watched_by_user = rating_df[rating_df.userId == user_id]
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-config.n_recommen:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    return recommended_movie_ids

def get_user_history(user_id):
    ratings = pd.read_csv('data/process/cleaned_ratings.csv')
    user_ratings = ratings[ratings['userId'] == int(user_id)]
    user_history = user_ratings['movieId'].astype(str).tolist()
    return user_history[:config.n_history]

    