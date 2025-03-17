import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'data_processing_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        raise

# Data loading function
def load_data(file_path):
    """
    Load data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# Data cleaning function
def clean_text(text):
    """
    Clean text data by removing special characters, numbers, etc.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

# Text preprocessing function
def preprocess_text(text, remove_stopwords=True):
    """
    Preprocess text by tokenizing, removing stopwords, and lemmatizing
    
    Args:
        text (str): Text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str) or text == "":
        return ""
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Extract company names and stock symbols
def extract_companies(text, company_list):
    """
    Extract company names and stock symbols from text
    
    Args:
        text (str): Text to extract from
        company_list (list): List of company names and symbols to look for
        
    Returns:
        list: List of found companies
    """
    if not isinstance(text, str) or text == "":
        return []
    
    found_companies = []
    text_lower = text.lower()
    
    for company in company_list:
        if company['name'].lower() in text_lower or company['symbol'].lower() in text_lower:
            found_companies.append(company['symbol'])
    
    return found_companies

# Process data
def process_data(df, company_list=None):
    """
    Process the data by cleaning and preprocessing text, extracting companies
    
    Args:
        df (pd.DataFrame): Data to process
        company_list (list): List of company dictionaries with 'name' and 'symbol' keys
        
    Returns:
        pd.DataFrame: Processed data
    """
    logger.info("Starting data processing")
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean and preprocess text
    logger.info("Cleaning and preprocessing news text")
    processed_df['cleaned_text'] = processed_df['news'].apply(clean_text)
    processed_df['preprocessed_text'] = processed_df['cleaned_text'].apply(preprocess_text)
    
    # Extract companies if company list is provided
    if company_list:
        logger.info("Extracting company mentions from news")
        processed_df['mentioned_companies'] = processed_df['news'].apply(
            lambda x: extract_companies(x, company_list)
        )
    
    logger.info("Data processing completed")
    return processed_df

# Save processed data
def save_processed_data(df, output_path):
    """
    Save processed data to CSV
    
    Args:
        df (pd.DataFrame): Data to save
        output_path (str): Path to save the data
    """
    try:
        logger.info(f"Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

# Main function to run the data processing pipeline
def run_data_processing(input_path, output_path, company_list_path=None):
    """
    Run the complete data processing pipeline
    
    Args:
        input_path (str): Path to the input data
        output_path (str): Path to save the processed data
        company_list_path (str): Path to the company list JSON file
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting data processing pipeline")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    df = load_data(input_path)
    
    # Load company list if provided
    company_list = None
    if company_list_path:
        try:
            company_list = pd.read_json(company_list_path).to_dict('records')
        except Exception as e:
            logger.warning(f"Could not load company list: {e}. Continuing without company extraction.")
    
    # Process data
    processed_df = process_data(df, company_list)
    
    # Save processed data
    save_processed_data(processed_df, output_path)
    
    logger.info("Data processing pipeline completed")
    return processed_df

if __name__ == "__main__":
    # Example usage
    input_path = "data/raw/news.csv"
    output_path = "data/processed/processed_news.csv"
    company_list_path = "data/raw/company_list.json"  # Optional
    
    run_data_processing(input_path, output_path, company_list_path) 