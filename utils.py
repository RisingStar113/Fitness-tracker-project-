import pandas as pd
import os

def log_fitness_data(file_path):
    # Collect user input
    steps = input("Enter steps walked: ")
    heart_rate = input("Enter heart rate (bpm): ")
    calories = input("Enter calories burned: ")
    sleep_hours = input("Enter sleep hours: ")

    # Prepare data dictionary
    new_entry = {
        "Steps": steps,
        "HeartRate": heart_rate,
        "CaloriesBurned": calories,
        "SleepHours": sleep_hours
    }

    # Convert to DataFrame
    df_new = pd.DataFrame([new_entry])

    # Append to file
    if os.path.exists(file_path):
        df_new.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(file_path, index=False)
    
    print("\nData logged successfully!")