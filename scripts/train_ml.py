import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("../data/weather_data.csv")

# Features & Target
X = df[['humidity', 'wind_speed']]
y = df['temperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae:.2f}")

# Save model
pickle.dump(model, open("../models/linear_regression.pkl", "wb"))
