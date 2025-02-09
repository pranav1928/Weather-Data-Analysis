from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("../models/linear_regression.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    humidity = float(data['humidity'])
    wind_speed = float(data['wind_speed'])

    # Predict using Linear Regression Model
    prediction = model.predict([[humidity, wind_speed]])[0]
    return jsonify({'predicted_temperature': prediction})

if __name__ == '__main__':
    app.run(debug=True)
