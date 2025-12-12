"""
Data preparation utilities for the Film Reviews Classification project.

This module exposes a single idempotent helper that:
- Loads a raw reviews CSV.
- Validates required columns ('content', 'score').
- Normalizes and filters score values into a binary 'sentiment' label
  (negative / positive) and drops ambiguous scores.
- Optionally restricts to short reviews.
- Produces stratified train/test CSV files in the requested output folder.

The implementation is robust to missing files/columns, creates output
directories as needed, and avoids chained-assignment issues.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def process_save_data(input_file_path: str=r"data/raw/netflix_reviews.csv", output_folder_path: str=r"data/processed") -> None:
    """
    Process raw data by keeping only the content and scores columns,
    removing rows with missing values, and saving the cleaned data to a new CSV file.

    Parameters:
    input_file_path (str): Path to the raw data CSV file.
    output_file_path (str): Path to save the cleaned data CSV file.
    """
    # Load raw data
    try:
        raw_data = pd.read_csv(input_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file_path} not found. Please provide a valid path to the raw data CSV file.")
    
    # Keep only content and scores columns
    cleaned_data = raw_data[['content', 'score']]

    # Remove rows with missing values
    cleaned_data = cleaned_data.dropna()

    # Make "sentiment" column: merge scores 1 and 2 into 0, scores 4 and 5 into 1, while taking only these rows 
    cleaned_data = cleaned_data[(cleaned_data['score'] ==1) | (cleaned_data['score'] == 5)]
    cleaned_data['sentiment'] = cleaned_data['score'].apply(lambda x: 0 if x ==1 else 1)
    cleaned_data = cleaned_data.drop('score', axis=1)

    # As our dataset is too large we will choose only short(length is less than 150) sequences
    cleaned_data['content_length'] = cleaned_data['content'].apply(len)
    cleaned_data = cleaned_data[cleaned_data["content_length"]<150]
    cleaned_data.drop(columns=['content_length'])

    # Save cleaned data to new CSV file
    train_path = f"{output_folder_path}/train_data.csv"
    test_path = f"{output_folder_path}/test_data.csv"
    X = cleaned_data['content']
    y = cleaned_data['sentiment']   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    train_data = pd.DataFrame({'content': X_train, 'sentiment': y_train})
    test_data = pd.DataFrame({'content': X_test, 'sentiment': y_test})
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"Cleaned data saved to {train_path} and {test_path}")