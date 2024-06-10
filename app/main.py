from flask import Flask, request, jsonify
from utils import *
from recomender import *

app = Flask(__name__)

# Load the movie data
movies = pd.read_csv('data/movielens/movies.csv')

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    return_metadata = request.args.get('returnMetadata', False)

    # Call the recommendation function
    recommended_movie_ids = recommend_movies(user_id)

    if return_metadata:
        # Join with the movie metadata
        recommendations = movies[movies['movieId'].isin(recommended_movie_ids)]
        recommendations = recommendations[['movieId', 'title', 'genres']]
        recommendations.columns = ['id', 'title', 'genres']
        recommendations = recommendations.to_dict('records')
    else:
        # Return just the movie IDs
        recommendations = [{'id': str(movie_id)} for movie_id in recommended_movie_ids]

    return jsonify({'items': recommendations})

@app.route('/features', methods=['GET'])
def get_features():
    user_id = request.args.get('user_id')

    # Get the user's history (e.g., from a database)
    user_history = get_user_history(user_id)

    return jsonify({'features': [{'histories': user_history}]})

if __name__ == '__main__':
    app.run(debug=True)