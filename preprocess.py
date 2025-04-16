import pandas as pd
import os

# Define file paths
RAW_DATA_FILE = "data/exercise.csv"
PROCESSED_DATA_FILE = "data/calories.csv"

def preprocess_data():
    """Preprocesses fitness data by handling missing values, converting data types, and normalizing."""
    
    if not os.path.exists(RAW_DATA_FILE):
        print("No raw data found. Please log some data first.")
        return

    # Load raw data
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Convert data to numeric (handling errors)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values (fill with median values)
    df.fillna(df.median(), inplace=True)

    # Normalize data (scaling between 0 and 1)
    df["Steps"] = df["Steps"] / df["Steps"].max()
    df["HeartRate"] = df["HeartRate"] / df["HeartRate"].max()
    df["CaloriesBurned"] = df["CaloriesBurned"] / df["CaloriesBurned"].max()
    df["SleepHours"] = df["SleepHours"] / df["SleepHours"].max()

    # Save processed data
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print("Data preprocessing complete. Processed data saved.")

if __name__ == "__main__":
    preprocess_data()