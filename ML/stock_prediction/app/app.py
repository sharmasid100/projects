from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model and scaler
model = load_model('predictions_model.keras')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            data = pd.read_csv(filepath)
            if data.shape[1] != 1:
                prediction = "❌ CSV must have only 1 column of numerical data."
            elif len(data) < 100:
                prediction = "❌ CSV must have at least 100 rows."
            else:
                values = data.iloc[:, 0].values.reshape(-1, 1)
                scaled = scaler.transform(values)
                input_data = np.reshape(scaled[-100:], (1, 100, 1))
                pred_scaled = model.predict(input_data)[0][0]
                pred_actual = scaler.inverse_transform([[pred_scaled]])[0][0]
                prediction = f"✅ Predicted next value: {pred_actual:.4f}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
