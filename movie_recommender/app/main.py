from flask import Flask, request, jsonify
from utils import get_user_history, process_data, get_recomendation
from model import RecommenderNet
import pandas as pd
import logging
import config
import os


app = Flask(__name__)

# Configure logging Refference from this : https://docs.python.org/3/howto/logging-cookbook.html
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
MOVIES_FILE_PATH = os.getenv('MOVIES_FILE_PATH', '/app/movie_recommender/data/movielens/movies.csv')
CLEANED_RATINGS_FILE_PATH = os.getenv('CLEANED_RATINGS_FILE_PATH', '/app/movie_recommender/data/process/cleaned_ratings.csv')

# Load the movie data
try:
    movies = pd.read_csv(MOVIES_FILE_PATH)
except FileNotFoundError:
    logger.error(f'Movies file not found: {MOVIES_FILE_PATH}')
    movies = pd.DataFrame(columns=['movieId', 'title', 'genres'])

# Load the ratings data
try:
    ratings = pd.read_csv(CLEANED_RATINGS_FILE_PATH)
except FileNotFoundError:
    logger.error(f'Rating file not found: {CLEANED_RATINGS_FILE_PATH}')
    ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

data_f,n_user,n_movie, movie2movie_encoded, user2user_encoded, movie_encoded2movie = process_data(ratings)
recomendation_model = RecommenderNet(n_user, n_movie, 50)
recomendation_model.load_weights('movie_recommender/data/models/recommender_weights.weights.h5')

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    return_metadata = request.args.get('returnMetadata', 'false').lower() == 'true'

    if not user_id or not user_id.isdigit():
        return jsonify({'error': 'Valid user_id is required'}), 400

    try:
        recommended_movie_ids = get_recomendation(recomendation_model,
                                                  data_f,
                                                  movies,
                                                  int(user_id),
                                                  movie2movie_encoded, 
                                                  user2user_encoded, 
                                                  movie_encoded2movie)
    except Exception as e:
        logger.error(f'Error getting recommendations: {e}')
        return jsonify({'error': 'Error getting recommendations'}), 500

    if return_metadata:

        # Join with the movie metadata
        recommendations = movies[movies['movieId'].isin(recommended_movie_ids)]
        recommendations = recommendations[['movieId', 'title', 'genres']]
        recommendations.columns = ['id', 'title', 'genres']
        recommendations['genres'] = recommendations['genres'].apply(lambda x: x.split('|'))
        recommendations = recommendations.to_dict(orient='records')
        recommendations = sorted(recommendations, key=lambda x: recommended_movie_ids.index(x['id']))

    else:
        # Return just the movie IDs
        recommendations = [{'id': str(movie_id)} for movie_id in recommended_movie_ids]

    return jsonify({'items': recommendations})

@app.route('/features', methods=['GET'])
def get_features():
    user_id = request.args.get('user_id')

    if not user_id or not user_id.isdigit():
        return jsonify({'error': 'Valid user_id is required'}), 400

    try:
        user_history = get_user_history(user_id, ratings)
    except Exception as e:
        logger.error(f'Error getting user history: {e}')
        return jsonify({'error': 'Error getting user history'}), 500

    return jsonify({'features': [{'histories': user_history}]})

if __name__ == '__main__':
    app.run(host=config.service_host, port=config.service_port,debug=False)
