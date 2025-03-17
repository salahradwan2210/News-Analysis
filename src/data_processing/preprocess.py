"""
Data preprocessing module for financial news analysis.
"""
import os
import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "news.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_news.csv"
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "raw" / "sample_news.csv"

# Download NLTK resources if not already downloaded
def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        logger.info("NLTK resources downloaded successfully.")

# Text preprocessing functions
def clean_text(text):
    """Clean and normalize text data."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """Tokenize text into words."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Remove stopwords from tokenized text."""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    """Lemmatize tokens to their root form."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    """Apply full preprocessing pipeline to text."""
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

def extract_stock_symbols(text):
    """Extract potential stock symbols from text."""
    # Simple pattern for stock symbols (uppercase letters, 1-5 characters)
    pattern = r'\b[A-Z]{1,5}\b'
    symbols = re.findall(pattern, text)
    return symbols

def add_financial_features(df):
    """Add financial-specific features to the dataframe."""
    # Extract symbols from content if not already present
    if 'symbol' not in df.columns:
        df['extracted_symbols'] = df['news'].apply(
            lambda x: ','.join(extract_stock_symbols(x)) if isinstance(x, str) else ''
        )
    
    # Count financial terms
    financial_terms = [
        'profit', 'loss', 'revenue', 'earnings', 'dividend', 'stock', 
        'share', 'market', 'investor', 'trading', 'growth', 'decline',
        'increase', 'decrease', 'quarterly', 'annual', 'fiscal'
    ]
    
    for term in financial_terms:
        df[f'{term}_count'] = df['news'].apply(
            lambda x: x.lower().count(term) if isinstance(x, str) else 0
        )
    
    # Add total financial terms count
    df['financial_terms_count'] = df[[f'{term}_count' for term in financial_terms]].sum(axis=1)
    
    return df

def process_data(use_sample=False, data_path=None, data_percentage=100.0):
    """
    Process the financial news dataset.
    
    Args:
        use_sample (bool): Whether to use the sample dataset instead of the full dataset.
        data_path (str or Path, optional): Path to the data file to process. If None, use default paths.
        data_percentage (float): Percentage of data to use (1-100).
        
    Returns:
        pd.DataFrame: Processed dataframe.
    """
    # Download NLTK resources
    download_nltk_resources()
    
    # Determine which dataset to use
    if data_path is not None:
        data_path = Path(data_path)
    else:
        data_path = SAMPLE_DATA_PATH if use_sample else RAW_DATA_PATH
    
    # Check if the file exists
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        if not use_sample and os.path.exists(SAMPLE_DATA_PATH):
            logger.info(f"Using sample data instead: {SAMPLE_DATA_PATH}")
            data_path = SAMPLE_DATA_PATH
        else:
            raise FileNotFoundError(f"No data file found at {data_path}")
    
    # Load the dataset
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Sample data if percentage is less than 100
    if data_percentage < 100:
        sample_size = int(len(df) * (data_percentage / 100))
        if sample_size < 1:
            sample_size = 1
        logger.info(f"Sampling {data_percentage}% of data ({sample_size} out of {len(df)} records)")
        df = df.sample(n=sample_size, random_state=42)
    
    # Basic data cleaning
    logger.info("Performing basic data cleaning")
    
    # Handle missing values
    df = df.dropna(subset=['content' if 'content' in df.columns else 'news'])
    
    # Rename columns if needed
    if 'content' in df.columns and 'news' not in df.columns:
        df = df.rename(columns={'content': 'news'})
    
    # Process text data
    logger.info("Processing text data")
    df['processed_content'] = df['news'].apply(preprocess_text)
    
    # Add financial features
    logger.info("Adding financial features")
    df = add_financial_features(df)
    
    # Create date features if date column exists
    if 'date' in df.columns:
        logger.info("Creating date features")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
    
    # Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    logger.info(f"Data processing completed. Processed {len(df)} news items.")
    return df

def split_data(df, test_size=0.2, val_size=0.1, random_state=42, target_col=None):
    """
    Split the data into training, validation, and test sets.
    
    Args:
        df (pd.DataFrame): The processed dataframe.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of training data to use for validation.
        random_state (int): Random seed for reproducibility.
        target_col (str, optional): Target column name. If None, will try to determine automatically.
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Ensure we have the necessary columns
    if 'processed_content' not in df.columns:
        logger.warning("'processed_content' column not found. Using 'news' column instead.")
        text_col = 'news'
    else:
        text_col = 'processed_content'
    
    # Determine target column if not provided
    if target_col is None:
        if 'sentiment' in df.columns:
            target_col = 'sentiment'
        elif 'compound' in df.columns:
            # Create sentiment labels from compound scores
            df['sentiment_label'] = df['compound'].apply(
                lambda x: 'POSITIVE' if x > 0.05 else ('NEGATIVE' if x < -0.05 else 'NEUTRAL')
            )
            target_col = 'sentiment_label'
        else:
            logger.error("No sentiment column found in the dataframe.")
            raise ValueError("No sentiment column found in the dataframe.")
    
    # Split into features and target
    X = df[text_col]
    y = df[target_col]
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
    )
    
    # Second split: training vs validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, 
        random_state=random_state, stratify=y_train_val if len(y_train_val.unique()) > 1 else None
    )
    
    logger.info(f"Data split completed. Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    try:
        # Process the data
        df = process_data(use_sample=True)
        
        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        
        # Print some statistics
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
    
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise 