# calorie_prediction_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_calorie_prediction_model(data):
    # Assume 'data' has features like steps, heart_rate, activity_duration, etc.
    X = data[['steps', 'heart_rate', 'activity_duration']]
    y = data['calories_burned']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

    return model

# Example usage
data = pd.read_csv('fitness_data.csv')  # Example dataset
model = train_calorie_prediction_model(data)