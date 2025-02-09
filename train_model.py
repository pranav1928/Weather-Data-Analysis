import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("data/weather_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"] = df["timestamp"].astype(int) / 10**9  # Convert timestamp to numeric

# Features & Target
X = df[["timestamp", "humidity", "pressure"]]
y = df["temperature"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
pickle.dump(linear_model, open("models/linear_model.pkl", "wb"))

# Train LSTM Model
X_train_lstm = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])

lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[1])),
    LSTM(50, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=16, verbose=1)
lstm_model.save("models/lstm_model.h5")
