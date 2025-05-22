from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_rent():
    data = request.form
    rent = util.get_predicted_rent(
        float(data['facility']),
        int(data['services']),
        int(data['members']),
        int(data['size']),
        data['type'],
        int(data['appearance_score']),
        int(data['attached_washroom']),
        data['location'],
        int(data['electricity']),
        int(data['wifi']),
        int(data['security_rating'])
    )
    return f"<h2>Estimated Rent: ₹{round(rent, 2)}</h2><br><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
