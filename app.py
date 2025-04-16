import pandas as pd
import os
from src.train_model import predict_fitness
from app.utils import log_fitness_data

# Define the CSV file path
DATA_FILE = "data/exercise.csv"

def main():
    print("Welcome to the Personal Fitness Tracker!")
    
    while True:
        print("\nChoose an option:")
        print("1. Log new fitness data")
        print("2. View fitness data")
        print("3. Predict fitness level")
        print("4. Exit")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            log_fitness_data(DATA_FILE)
        
        elif choice == "2":
            if os.path.exists(DATA_FILE):
                df = pd.read_csv(DATA_FILE)
                print("\nFitness Data:\n", df.tail(10))  # Show last 10 records
            else:
                print("\nNo fitness data available. Please log some data first.")

        elif choice == "3":
            if os.path.exists(DATA_FILE):
                predict_fitness(DATA_FILE)
            else:
                print("\nNo data available to make predictions.")
        
        elif choice == "4":
            print("\nExiting the application. Stay fit!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__== "__main__":
    main()