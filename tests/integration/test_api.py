import unittest
import requests

class TestRecommendationsEndpoint(unittest.TestCase):
    def setUp(self):
        # Set up any test data or configurations needed
        pass

    def tearDown(self):
        # Clean up after each test if needed
        pass

    def test_recommendations_endpoint_without_metadata(self):
        # Send a GET request to the recommendations endpoint without metadata
        response = requests.get('http://127.0.0.1:5000/recommendations?user_id=18')

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected JSON structure
        data = response.json()
        self.assertTrue('items' in data)
        self.assertIsInstance(data['items'], list)
        for item in data['items']:
            self.assertIsInstance(item, dict)
            self.assertIn('id', item)

    def test_recommendations_endpoint_with_metadata(self):
        # Send a GET request to the recommendations endpoint with metadata
        response = requests.get('http://127.0.0.1:5000/recommendations?user_id=18&returnMetadata=true')

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected JSON structure
        data = response.json()
        self.assertTrue('items' in data)
        self.assertIsInstance(data['items'], list)
        for item in data['items']:
            self.assertIsInstance(item, dict)
            self.assertIn('id', item)
            self.assertIn('title', item)
            self.assertIn('genres', item)

    def test_features_endpoint(self):
        # Send a GET request to the features endpoint
        response = requests.get('http://127.0.0.1:5000/features?user_id=18')

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check if the response contains the expected JSON structure
        data = response.json()
        self.assertTrue('features' in data)
        self.assertIsInstance(data['features'], list)
        self.assertEqual(len(data['features']), 1)
        for feature in data['features']:
            self.assertIsInstance(feature, dict)
            self.assertIn('histories', feature)
            self.assertIsInstance(feature['histories'], list)

if __name__ == '__main__':
    unittest.main()
