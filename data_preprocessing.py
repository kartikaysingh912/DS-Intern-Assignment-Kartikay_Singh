import pandas as pd
import numpy as np

def preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Convert numeric columns from object to float
    numeric_cols = df.columns.drop('timestamp')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check missing values after conversion
    missing_before = df.isnull().sum()
    
    # Handle missing values: for simplicity, drop rows with missing target or timestamp
    df = df.dropna(subset=['timestamp', 'equipment_energy_consumption'])
    
    # For other missing values, fill with column mean
    df = df.fillna(df.mean())
    
    missing_after = df.isnull().sum()
    
    print("Missing values before cleaning:")
    print(missing_before)
    print("\nMissing values after cleaning:")
    print(missing_after)
    
    return df

if __name__ == "__main__":
    data_file = "data/data.csv"
    df_clean = preprocess_data(data_file)
    print("\nCleaned data sample:")
    print(df_clean.head())
