import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load dataset
df = pd.read_csv("../data/weather_data.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Prepare data for LSTM
sequence_length = 10
features = ['temperature', 'humidity', 'wind_speed']

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict temperature
    return np.array(X), np.array(y)

data = df[features].values
X, y = create_sequences(data, sequence_length)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))),
    LSTM(50),
    Dense(25),
    Dense(1)
])

# Compile and Train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Save Model
model.save("../models/lstm_model.h5")
