"""
Data processing script for text sentiment analysis.

This script handles:
1. Loading raw data
2. Text preprocessing
3. Feature engineering
4. Saving processed data
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Optional
from datasets import load_dataset

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def download_and_prepare_imdb():
    """
    Downloads the IMDB dataset and prepares it for training.
    Saves train and test sets as CSV files.
    """
    print("Downloading and preparing IMDB dataset...")
    
    # Create directories if they don't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download dataset using Hugging Face datasets
    imdb = load_dataset("imdb")
    
    # Convert to pandas and save as CSV
    train_df = pd.DataFrame(imdb["train"])
    test_df = pd.DataFrame(imdb["test"])
    
    # Save raw data
    train_df.to_csv(RAW_DATA_DIR / "imdb_train.csv", index=False)
    test_df.to_csv(RAW_DATA_DIR / "imdb_test.csv", index=False)
    
    # Basic preprocessing - rename columns to standard names
    train_df = train_df.rename(columns={"text": "text", "label": "sentiment"})
    test_df = test_df.rename(columns={"text": "text", "label": "sentiment"})
    
    # Save processed data
    train_df.to_csv(PROCESSED_DATA_DIR / "imdb_train_processed.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "imdb_test_processed.csv", index=False)
    
    # Save some dataset metadata
    metadata = {
        "dataset": "IMDB Movie Reviews",
        "task": "Sentiment Analysis",
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df),
        "columns": list(train_df.columns),
        "label_mapping": {0: "negative", 1: "positive"}
    }
    
    with open(PROCESSED_DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data preparation complete. {len(train_df)} training samples, {len(test_df)} test samples.")
    print(f"Files saved to {PROCESSED_DATA_DIR}/")

def load_data(file_name: str) -> Optional[pd.DataFrame]:
    """
    Load data from the raw data directory.
    
    Args:
        file_name: Name of the file to load
        
    Returns:
        DataFrame containing the loaded data or None if file doesn't exist
    """
    file_path = RAW_DATA_DIR / file_name
    if not file_path.exists():
        print(f"Error: File {file_name} not found in {RAW_DATA_DIR}")
        return None
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file {file_name}: {str(e)}")
        return None

def preprocess_text(text: str) -> str:
    """
    Preprocess text data.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    # TODO: Implement text preprocessing steps
    # 1. Convert to lowercase
    # 2. Remove special characters
    # 3. Remove extra whitespace
    # 4. Handle contractions
    # 5. Remove stopwords
    return text

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Processed DataFrame
    """
    # TODO: Implement data processing steps
    # 1. Apply text preprocessing
    # 2. Create features
    # 3. Handle missing values
    # 4. Encode categorical variables
    return df

def save_processed_data(df: pd.DataFrame, output_file: str) -> bool:
    """
    Save processed data to the processed data directory.
    
    Args:
        df: DataFrame to save
        output_file: Name of the output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create processed data directory if it doesn't exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save the processed data
        output_path = PROCESSED_DATA_DIR / output_file
        df.to_csv(output_path, index=False)
        print(f"Successfully saved processed data to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving processed data: {str(e)}")
        return False

def main():
    """Main function to run the data processing pipeline."""
    # Download and prepare IMDB dataset
    download_and_prepare_imdb()
    
    # Example usage for custom data processing
    input_file = "imdb_train.csv"  # Using the downloaded IMDB data
    output_file = "imdb_train_further_processed.csv"
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return
    
    # Process data
    processed_df = process_data(df)
    
    # Save processed data
    save_processed_data(processed_df, output_file)

if __name__ == "__main__":
    main() 