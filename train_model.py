import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def predict_fitness(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert data to numeric
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Define features and target
    X = df[["Steps", "HeartRate", "CaloriesBurned"]]
    y = df["SleepHours"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict for last entry
    last_entry = X.iloc[-1:].values
    prediction = model.predict(last_entry)[0]

    print(f"\nPredicted sleep hours based on your activity: {prediction:.2f}")