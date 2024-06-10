from flask import Flask, request, jsonify

app = Flask(__name__)

# Dummy data for recommendations and features
recommendations_data = {
    "18": [
        {"id": "74510"},
        {"id": "76175"}
    ]
}

recommendations_metadata = {
    "18": [
        {
            "id": "74510",
            "title": "Girl Who Played with Fire, The (Flickan som lekte med elden) (2009)",
            "genres": ["Action", "Crime", "Drama", "Mystery", "Thriller"]
        },
        {
            "id": "76175",
            "title": "Clash of the Titans (2010)",
            "genres": ["Action", "Adventure", "Drama", "Fantasy"]
        }
    ]
}

features_data = {
    "18": {
        "features": [
            {
                "histories": ["185135", "180777", "180095", "177593"]
            }
        ]
    }
}

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    return_metadata = request.args.get('returnMetadata', 'false').lower() == 'true'
    
    if return_metadata:
        return jsonify({"items": recommendations_metadata.get(user_id, [])})
    else:
        return jsonify({"items": recommendations_data.get(user_id, [])})

@app.route('/features', methods=['GET'])
def get_features():
    user_id = request.args.get('user_id')
    return jsonify(features_data.get(user_id, {"features": []}))

if __name__ == '__main__':
    app.run(debug=True)
