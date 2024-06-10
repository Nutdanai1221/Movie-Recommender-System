import pandas as pd

def get_user_history(user_id):

    ratings = pd.read_csv('data/process/cleaned_ratings.csv')
    user_ratings = ratings[ratings['userId'] == int(user_id)]
    user_history = user_ratings['movieId'].astype(str).tolist()

    return user_history[:10]