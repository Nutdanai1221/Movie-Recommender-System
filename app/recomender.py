import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_movies(user_id):
    # Load the rating data
    ratings = pd.read_csv('data/process/cleaned_ratings.csv')

    # Pivot the ratings data into a user-item matrix
    user_data = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_mean_ratings = user_data.mean(axis=1)
    user_data_mean_centered = user_data.sub(user_mean_ratings, axis=0)

    user_similarity = cosine_similarity(user_data_mean_centered)
    user_similarity[np.isnan(user_similarity)] = 0
    dummy = ratings.copy()
    
    dummy['rating'] = dummy['rating'].apply(lambda x: 0 if x > 0 else 1)
    dummy = dummy.pivot(index='userId', columns='movieId', values='rating').fillna(1)
    user_predicted_ratings = np.dot(user_similarity, user_data_mean_centered)
    user_final_ratings = np.multiply(user_predicted_ratings, dummy)

    # Convert the user_final_ratings to a DataFrame for easier handling
    user_final_ratings_df = pd.DataFrame(user_final_ratings, index=user_data.index, columns=user_data.columns)

    # Get the top N recommended movie IDs for the given user
    user_ratings = user_final_ratings_df.loc[int(user_id)]
    top_movie_ids = user_ratings.sort_values(ascending=False).index.tolist()

    return top_movie_ids[:10]  # Return top 10 movie IDs

if __name__ == "__main__":
    print(recommend_movies(610))
