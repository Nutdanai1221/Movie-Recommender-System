import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from io import StringIO
from unittest.mock import Mock
from config import n_history, n_recommen
from model import RecommenderNet  # Replace with the actual module name
from utils import process_data, get_recomendation, get_user_history

class TestRecommenderNet(unittest.TestCase):

    def setUp(self):
        self.num_users = 100
        self.num_movies = 100
        self.embedding_size = 32
        self.model = RecommenderNet(self.num_users, self.num_movies, self.embedding_size)

    def test_initialization(self):
        # Test that the embeddings and biases are created with the correct shapes
        self.assertEqual(self.model.user_embedding.input_dim, self.num_users)
        self.assertEqual(self.model.user_embedding.output_dim, self.embedding_size)
        self.assertEqual(self.model.user_bias.input_dim, self.num_users)
        self.assertEqual(self.model.user_bias.output_dim, 1)
        self.assertEqual(self.model.movie_embedding.input_dim, self.num_movies)
        self.assertEqual(self.model.movie_embedding.output_dim, self.embedding_size)
        self.assertEqual(self.model.movie_bias.input_dim, self.num_movies)
        self.assertEqual(self.model.movie_bias.output_dim, 1)

    def test_call(self):
        # Create a sample input
        inputs = np.array([[1, 2], [3, 4]])
        # Run the call method
        outputs = self.model(inputs)
        # Check the shape of the output
        self.assertEqual(outputs.shape, (2, 1))
        # Check that the output is between 0 and 1
        self.assertTrue(np.all(outputs >= 0) and np.all(outputs <= 1))


class TestUtilsFunctions(unittest.TestCase):

    def test_process_data(self):
        csv_data = """userId,movieId,rating
                      1,10,4.0
                      1,20,5.0
                      2,30,3.0
                      1,40,2.5
                      2,10,3.5
                      2,20,4.5
                      3,30,2.0
                      3,40,3.0
                      4,10,4.0
                      4,20,5.0"""
        ratings = pd.read_csv(StringIO(csv_data))
        print(ratings)
        dataframe, num_users, num_movies, movie2movie_encoded, user2user_encoded, movie_encoded2movie = process_data(ratings)  # Pass DataFrame instead of StringIO

        # Check dataframe shape and contents
        self.assertEqual(dataframe.shape, ratings.shape)  # Update with your expected shape
        self.assertIn('userId', dataframe.columns)
        self.assertIn('movieId', dataframe.columns)

        print(num_users, num_movies, movie2movie_encoded, user2user_encoded, movie_encoded2movie)
        # print(len(set(ratings['userId'])))
        # # Check user and movie counts
        self.assertEqual(num_users, len(set(ratings['userId'])))
        self.assertEqual(num_movies, len(set(ratings['movieId'])))
        
        # # Check encoding dictionaries
        self.assertEqual(movie2movie_encoded, {10: 0, 20: 1, 30: 2, 40:3})
        self.assertEqual(user2user_encoded, {1: 0, 2: 1, 3:2, 4:3})
        self.assertEqual(movie_encoded2movie, {0:10, 1:20, 2:30, 3:40})

    def test_get_recommendation(self):
        # Sample data
        rating_df = pd.DataFrame({
            "userId": [1, 1, 2, 2, 1],
            "movieId": [1, 3, 2, 5, 4],
            "rating": [4.0, 4.0, 4.0, 5.0, 5.0]
        })

        movie_df = pd.DataFrame({
            "movieId": [1, 2, 3, 4, 5],
            "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)", "Waiting to Exhale (1995)", "Father of the Bride Part II (1995)"],
            "genres": ["Adventure|Animation|Children|Comedy|Fantasy", "Adventure|Children|Fantasy", "Comedy|Romance", "Comedy|Drama|Romance", "Comedy"]
        })

        # Mocking the model
        model = MagicMock()
        model.predict.return_value = np.array([4.5, 3.0, 2.0])  # Sample prediction

        movie2movie_encoded = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6:5}  # Sample encoding
        user2user_encoded = {1: 0, 2:1}  # Sample encoding
        movie_encoded2movie = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5:6}  # Sample encoding
        user_id = 2

        # Call the function
        recommended_movie_ids = get_recomendation(model, rating_df, movie_df, user_id, movie2movie_encoded, user2user_encoded, movie_encoded2movie)

        # Assertions
        self.assertEqual(len(recommended_movie_ids), 3)


    def test_get_user_history(self):
        csv_data = """userId,movieId,rating
                      1,10,4.0
                      1,20,5.0
                      2,30,3.0
                      1,40,2.5
                      2,10,3.5
                      2,20,4.5
                      3,30,2.0
                      3,40,3.0
                      4,10,4.0
                      4,20,5.0"""
        ratings = pd.read_csv(StringIO(csv_data))

        user_id = 1

        history = get_user_history(user_id, ratings)

        # Check history
        self.assertEqual(history, ['10', '20', '40'])

if __name__ == '__main__':
    unittest.main()
