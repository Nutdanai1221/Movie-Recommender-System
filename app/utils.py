import pandas as pd
from recomender import recommend_movies

def get_user_history(user_id):
    ratings = pd.read_csv('data/process/cleaned_ratings.csv')
    user_ratings = ratings[ratings['userId'] == int(user_id)]
    user_history = user_ratings['movieId'].astype(str).tolist()
    return user_history[:10]

def get_recommendations(user_id, return_metadata):
    movies = pd.read_csv('data/movielens/movies.csv')
    recommended_movie_ids = recommend_movies(user_id)

    if return_metadata:
        recommendations = movies[movies['movieId'].isin(recommended_movie_ids)]
        recommendations = recommendations[['movieId', 'title', 'genres']]
        recommendations.columns = ['id', 'title', 'genres']
        recommendations['genres'] = recommendations['genres'].apply(lambda x: x.split('|'))
        return {'items': recommendations.to_dict('records')}
    else:
        return {'items': [{'id': str(movie_id)} for movie_id in recommended_movie_ids]}

def get_features(user_id):
    user_history = get_user_history(user_id)
    return {'features': [{'histories': user_history}]}

if __name__ == "__main__" :
    print(get_features(18))
