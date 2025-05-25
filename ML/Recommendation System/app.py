from flask import Flask, request, jsonify, render_template
from recommend import recommend

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json()
    movie_name = data.get('movie')

    if not movie_name:
        return jsonify({'error': 'Please provide a movie title.'}), 400

    recommendations = recommend(movie_name)

    if not recommendations:
        return jsonify({'error': f'No recommendations found for "{movie_name}".'}), 404

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
