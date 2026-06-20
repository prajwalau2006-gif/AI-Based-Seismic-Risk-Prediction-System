import pandas as pd
import os

def clean_data():
    raw_path = "d:/Earthquake_Project/japan_earthquake_dataset.csv"
    cleaned_path = "d:/Earthquake_Project/cleaned_japan_earthquake_dataset.csv"
    
    print("Loading raw dataset...")
    df = pd.read_csv(raw_path)
    print(f"Raw dataset shape: {df.shape}")
    
    # 1. Keep only earthquake records
    print("Filtering out non-earthquake events...")
    df = df[df['type'] == 'earthquake']
    print(f"Shape after type filtering: {df.shape}")
    
    # 2. Convert 'time' to datetime format
    print("Converting 'time' to datetime...")
    df['time'] = pd.to_datetime(df['time'], errors='raise')
    
    # 3. Create year, month, day columns
    print("Extracting year, month, and day...")
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    
    # 4. Keep specified columns
    keep_columns = [
        'time',
        'year',
        'month',
        'day',
        'latitude',
        'longitude',
        'depth',
        'mag',
        'place'
    ]
    df_cleaned = df[keep_columns].copy()
    
    # 5. Check for missing values in cleaned columns
    print("Checking for missing values...")
    missing_summary = df_cleaned.isnull().sum()
    print(missing_summary)
    
    # 6. Save the cleaned dataset
    print(f"Saving cleaned dataset to {cleaned_path}...")
    df_cleaned.to_csv(cleaned_path, index=False)
    print("Cleaning complete!")
    
    # 7. Print verification details
    print("\n--- CLEANED DATASET SUMMARY ---")
    print(f"Final shape: {df_cleaned.shape}")
    print(f"Columns: {list(df_cleaned.columns)}")
    print("\nMissing values:")
    print(df_cleaned.isnull().sum())
    print("\nStatistical summary:")
    print(df_cleaned.describe(include='all'))

if __name__ == "__main__":
    clean_data()
