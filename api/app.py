import pickle
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained models
linear_model = pickle.load(open("models/linear_model.pkl", "rb"))
lstm_model = load_model("models/lstm_model.h5")

@app.route("/")
def home():
    return "Weather Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    humidity = data["humidity"]
    pressure = data["pressure"]

    # Linear Model Prediction
    timestamp = np.array([[pd.Timestamp.now().timestamp(), humidity, pressure]])
    temp_linear = linear_model.predict(timestamp)[0]

    # LSTM Model Prediction
    timestamp_lstm = timestamp.reshape(1, 1, 3)
    temp_lstm = lstm_model.predict(timestamp_lstm)[0][0]

    return jsonify({
        "linear_regression_temperature": temp_linear,
        "lstm_temperature": temp_lstm
    })

if __name__ == "__main__":
    app.run(debug=True)
