# Movie Recommender System 🎬

A movie recommender system suggests movies to users based on their preferences and viewing history. The goal is to provide personalized recommendations that match the user's interests, enhancing their overall viewing experience.

## Assignment

Content discovery helps users explore the content they would like to consume. Currently, we manually
curate the content into each row then present it to the users. However, we are now growing as a
business. Users come and experience our services every day, so we need your help!


Let's build a recommender system based on users' behaviors to recommend the content. As a Machine
Learning Engineer, in this project, you need to develop a recommender system, starting from data
preparation, model development, model deployment, and API service.

### Example
Given User A, who loves to watch Action movies,
When User A sends a request to an API ({"user_id": "A"}),
Then the recommender system should return a list of movies related to User A.
### Datasets

The dataset used is based on the MovieLens small datasets, available [here](https://github.com/lukkiddd-tdg/movielens-small).

### Requirements for API Service

| Path                      | Query Parameter        | Response                                                                                             |
|---------------------------|------------------------|------------------------------------------------------------------------------------------------------|
| GET /recommendations      | ?user_id=18            | { "items": [{ "id": "74510" }, { "id": "76175" }] }                                                  |
| GET /recommendations      | ?user_id=18&returnMetadata=true | { "items": [{ "id":"74510", "title": "Girl Who Played with Fire, The (2009)", "genres": ["Action", "Crime", "Drama", "Mystery", "Thriller"] }, { "id":"76175", "title": "Clash of the Titans (2010)", "genres": ["Action", "Adventure", "Drama", "Fantasy"] }] } |
| GET /features             | ?user_id=18            | { "features": [{ "histories": ["185135", "180777", "180095", "177593"] }] }                          |

## How Does It Work

In this section, you'll discover the inner workings of Neural Collaborative Filtering (NCF), encompassing data preparation, model architecture, and the training process.

### Data Preparation
For the full data preparation, see the notebook [here](https://github.com/Nutdanai1221/Movie-Recommender-System/blob/master/movie_recommender/notebooks/movie-lens-recomendation.ipynb).

1. **Load Data**: The ratings data is loaded into a DataFrame.
    ```python
    df = pd.read_csv('../data/process/cleaned_ratings.csv')
    ```

2. **Encode Users and Movies**: Unique user and movie IDs are converted into numerical indices.
    ```python
    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["movie"] = df["movieId"].map(movie2movie_encoded)
    ```

3. **Normalize Ratings**: The ratings are normalized to the range [0, 1] to facilitate training.
    ```python
    df["rating"] = df["rating"].values.astype(np.float32)
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    ```

4. **Train-Validation Split**: The data is shuffled and split into training and validation sets.
    ```python
    df = df.sample(frac=1, random_state=42)
    x = df[["user", "movie"]].values
    train_indices = int(0.7 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    ```

### Model Architecture

The NCF model consists of embedding layers for users and movies, followed by a neural network to learn their interactions.

![NCF Architecture](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*aP-Mx266ExwoWZPSdHtYpA.png)

*Image Source: [Towards Data Science](https://towardsdatascience.com/neural-collaborative-filtering-96cef1009401)*

1. **Embedding Layers**: These layers convert user and movie indices into dense vectors of fixed size (`EMBEDDING_SIZE`).
    ```python
    EMBEDDING_SIZE = 50

    class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_movies, embedding_size, **kwargs):
            super().__init__(**kwargs)
            self.user_embedding = layers.Embedding(
                num_users,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.movie_embedding = layers.Embedding(
                num_movies,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.movie_bias = layers.Embedding(num_movies, 1)
    ```

2. **Forward Pass**: During the forward pass, user and movie embeddings are combined through a dot product, and biases are added to predict the rating.
    ```python
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = ops.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return x
    ```

### Training

The model is trained to minimize the difference between predicted and actual ratings using a suitable loss function.

## How to Feed Input and Get Output
In this section, you'll find instructions on setting up the environment and commands to request recommendations and retrieve responses from the API.
To set up the environment and make requests to the API for recommendations, follow these steps:

To run the project using Docker, follow these steps:

1. **Install Docker**:
   - If you haven't already installed Docker on your system, follow the instructions for your operating system:
     - [Install Docker on Windows](https://docs.docker.com/desktop/install/windows-install/)
     - [Install Docker on macOS](https://docs.docker.com/desktop/install/mac-install/)
     - [Install Docker on Linux](https://docs.docker.com/desktop/install/linux-install/)

2. **Run Docker Compose**:
   - Once Docker is installed, navigate to the project directory in your terminal.
   - Run the following command to build and start the containers:
     ```
     docker-compose up
     ```
3. **Requesting Recommendations with cURL**:
   - After Docker Compose has started the containers, open a new terminal window.
   - Use the following cURL command to request recommendations for a specific user:
     ```
     curl -X GET http://localhost:5000/recommendations?user_id=<USER_ID>
     ```
   - Replace `<USER_ID>` with the ID of the user for whom you want recommendations.
   - Example Request:
     ```
     curl -X GET http://localhost:5000/recommendations?user_id=18
     ```
    - **Example Response**:
     ```json
     {
       "items": [
         { "id": "142056" },
         { "id": "43912" }
       ]
     }
     ```
4. **Requesting Recommendations with Return Metadata**:
   - To include additional metadata about the recommended movies, add the `returnMetadata=true` parameter to the request.
   - Example Request:
     ```
     curl -X GET http://localhost:5000/recommendations?user_id=18&returnMetadata=true
     ```
   - **Example Response**:
     ```json
     {
       "items": [
         {
           "id": 142056,
           "title": "Iron Man & Hulk: Heroes United (2013)",
           "genres": ["Action", "Adventure", "Animation"]
         },
         {
           "id": 43912,
           "title": "Freedomland (2006)",
           "genres": ["Crime", "Drama"]
         }
       ]
     }
     ```
5. **Fetching User Features**:
   - Use the following `curl` command to retrieve the movie IDs that a specific user has already seen:
     ```
     curl -X GET http://localhost:5000/features?user_id=<USER_ID>
     ```
   - **Example Request**:
     ```
     curl -X GET http://localhost:5000/features?user_id=18
     ```
   - **Example Response**:
     ```json
     {
       "features": [
         { "histories": ["1", "2", "6", "16"] }
       ]
     }
     ```
see the example JSON output file [here](https://github.com/Nutdanai1221/Movie-Recommender-System/blob/master/movie_recommender/data/output_json)
## How to Improve in the Future
1. **Incorporate Additional User Features**: Enhance the recommender system by incorporating a wider range of user features such as user_id, sex, occupation, and age_group. By including these additional features, the system can provide more personalized recommendations tailored to individual user preferences and characteristics.

2. **Explore More Complex Models**: Consider exploring more advanced recommendation models beyond Neural Collaborative Filtering (NCF). For example, investigate Transformer-based recommendation systems, which have shown promising results in capturing intricate patterns in user behavior and preferences. By leveraging more complex models, the system can potentially improve recommendation accuracy and relevance.

3. **Utilize Diverse Data Sources**: Expand the scope of data used by the recommender system to include diverse sources such as social media activity, browsing history, and demographic information. Integrating these additional data sources can provide richer insights into user preferences and behavior, enabling the system to deliver more relevant and personalized recommendations.
