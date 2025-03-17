<<<<<<< HEAD
"""
Feature engineering module for financial news analysis.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
import gensim
from gensim.models import Word2Vec
import json
=======
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import logging
import os
from datetime import datetime
import joblib
>>>>>>> 42c49fa71cbd94cde7e31240fce9a881867433a4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
<<<<<<< HEAD
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_news.csv"
FEATURES_DIR = ROOT_DIR / "data" / "processed" / "features"
ALL_FEATURES_PATH = FEATURES_DIR / "all_features.csv"
TOPIC_KEYWORDS_PATH = FEATURES_DIR / "topic_keywords.json"
WORD2VEC_MODEL_PATH = FEATURES_DIR / "word2vec_model.bin"

# Ensure features directory exists
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def extract_sentiment_features(df):
    """
    Extract sentiment-related features from the dataframe.
    
    Args:
        df (pd.DataFrame): Processed dataframe with sentiment columns.
        
    Returns:
        pd.DataFrame: Dataframe with sentiment features.
    """
    logger.info("Extracting sentiment features")
    
    # Create a copy of the dataframe with only the necessary columns
    result_df = df.copy()
    
    # Ensure we have the necessary columns
    if 'processed_content' not in result_df.columns:
        logger.warning("'processed_content' column not found. Using 'news' column instead.")
        if 'news' in result_df.columns:
            result_df['processed_content'] = result_df['news']
        else:
            logger.error("Neither 'processed_content' nor 'news' column found.")
            raise ValueError("Text content column not found in the dataframe.")
    
    # Handle missing values
    result_df['processed_content'] = result_df['processed_content'].fillna('')
    
    # Extract sentiment features if available
    sentiment_cols = ['neg', 'neu', 'pos', 'compound', 'sentiment']
    available_cols = [col for col in sentiment_cols if col in result_df.columns]
    
    if not available_cols:
        logger.warning("No sentiment columns found in the dataframe.")
    else:
        logger.info(f"Found sentiment columns: {available_cols}")
    
    # Create sentiment polarity features if compound score is available
    if 'compound' in result_df.columns:
        result_df['sentiment_polarity'] = result_df['compound'].apply(
            lambda x: 1 if x > 0.05 else (-1 if x < -0.05 else 0)
        )
    
    # Create sentiment category if not available
    if 'sentiment' not in result_df.columns and 'compound' in result_df.columns:
        result_df['sentiment_category'] = result_df['compound'].apply(
            lambda x: 'POSITIVE' if x > 0.05 else ('NEGATIVE' if x < -0.05 else 'NEUTRAL')
        )
    
    return result_df

def extract_text_statistics(df):
    """
    Extract statistical features from text data.
    
    Args:
        df (pd.DataFrame): Processed dataframe with text content.
        
    Returns:
        pd.DataFrame: Dataframe with text statistics features.
    """
    logger.info("Extracting text statistics features")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Ensure we have the necessary columns
    text_col = 'processed_content' if 'processed_content' in result_df.columns else 'news'
    
    if text_col not in result_df.columns:
        logger.error(f"Text column '{text_col}' not found in the dataframe.")
        raise ValueError(f"Text column '{text_col}' not found in the dataframe.")
    
    # Handle missing values
    result_df[text_col] = result_df[text_col].fillna('')
    
    # Extract text statistics
    result_df['text_length'] = result_df[text_col].apply(len)
    result_df['word_count'] = result_df[text_col].apply(lambda x: len(x.split()))
    result_df['avg_word_length'] = result_df[text_col].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    result_df['unique_word_ratio'] = result_df[text_col].apply(
        lambda x: len(set(x.split())) / len(x.split()) if x and len(x.split()) > 0 else 0
    )
    
    return result_df

def extract_topic_features(df, n_topics=5, n_features=100, n_top_words=10):
    """
    Extract topic-related features using Latent Dirichlet Allocation (LDA).
    
    Args:
        df (pd.DataFrame): Processed dataframe with text content.
        n_topics (int): Number of topics to extract.
        n_features (int): Number of features to use in the vectorizer.
        n_top_words (int): Number of top words to save for each topic.
        
    Returns:
        pd.DataFrame: Dataframe with topic features.
    """
    logger.info(f"Extracting topic features with {n_topics} topics")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Ensure we have the necessary columns
    text_col = 'processed_content' if 'processed_content' in result_df.columns else 'news'
    
    if text_col not in result_df.columns:
        logger.error(f"Text column '{text_col}' not found in the dataframe.")
        raise ValueError(f"Text column '{text_col}' not found in the dataframe.")
    
    # Handle missing values
    result_df[text_col] = result_df[text_col].fillna('')
    
    # Create a document-term matrix
    # For small datasets, use min_df=1 to avoid pruning all terms
    vectorizer = CountVectorizer(
        max_features=n_features,
        stop_words='english',
        min_df=1  # Changed from 5 to 1 for small datasets
    )
    
    dtm = vectorizer.fit_transform(result_df[text_col])
    feature_names = vectorizer.get_feature_names_out()
    
    # Train LDA model
    # For small datasets, use smaller number of topics
    lda = LatentDirichletAllocation(
        n_components=min(n_topics, len(result_df)),  # Ensure n_topics doesn't exceed number of documents
        random_state=42,
        n_jobs=-1,
        learning_method='online'  # Added for better performance on small datasets
    )
    
    # Transform the document-term matrix to topic space
    topic_distributions = lda.fit_transform(dtm)
    
    # Add topic distribution features to the dataframe
    for i in range(min(n_topics, len(result_df))):
        result_df[f'topic_{i}'] = topic_distributions[:, i]
    
    # Add dominant topic for each document
    result_df['dominant_topic'] = np.argmax(topic_distributions, axis=1)
    
    # Save topic keywords
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_keywords[f'topic_{topic_idx}'] = top_words
    
    # Save topic keywords to a JSON file
    with open(TOPIC_KEYWORDS_PATH, 'w') as f:
        json.dump(topic_keywords, f, indent=2)
    
    logger.info(f"Topic keywords saved to {TOPIC_KEYWORDS_PATH}")
    
    return result_df

def train_word2vec(df, vector_size=50, window=3, min_count=1):
    """
    Train a Word2Vec model on the processed text data.
    
    Args:
        df (pd.DataFrame): Processed dataframe with text content.
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word.
        min_count (int): Minimum count of words to consider.
        
    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    logger.info("Training Word2Vec model")
    
    # Ensure we have the necessary columns
    text_col = 'processed_content' if 'processed_content' in df.columns else 'news'
    
    if text_col not in df.columns:
        logger.error(f"Text column '{text_col}' not found in the dataframe.")
        raise ValueError(f"Text column '{text_col}' not found in the dataframe.")
    
    # Handle missing values
    df[text_col] = df[text_col].fillna('')
    
    # Tokenize the text
    tokenized_texts = [text.split() for text in df[text_col] if text]
    
    # Train Word2Vec model
    # Adjusted parameters for small datasets
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,  # Reduced from 100
        window=window,  # Reduced from 5
        min_count=min_count,  # Changed from 5 to 1 for small datasets
        workers=4,
        epochs=20  # Increased from default for better training on small datasets
    )
    
    # Save the model
    model.save(str(WORD2VEC_MODEL_PATH))
    logger.info(f"Word2Vec model saved to {WORD2VEC_MODEL_PATH}")
    
    return model

def get_document_embedding(text, word2vec_model):
    """
    Get document embedding by averaging word vectors.
    
    Args:
        text (str): Processed text.
        word2vec_model (gensim.models.Word2Vec): Trained Word2Vec model.
        
    Returns:
        np.ndarray: Document embedding vector.
    """
    if not text:
        return np.zeros(word2vec_model.vector_size)
    
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    
    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)
    
    return np.mean(word_vectors, axis=0)

def add_word2vec_features(df, word2vec_model=None):
    """
    Add Word2Vec document embeddings as features.
    
    Args:
        df (pd.DataFrame): Processed dataframe with text content.
        word2vec_model (gensim.models.Word2Vec, optional): Trained Word2Vec model.
        
    Returns:
        pd.DataFrame: Dataframe with Word2Vec features.
    """
    logger.info("Adding Word2Vec features")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Ensure we have the necessary columns
    text_col = 'processed_content' if 'processed_content' in result_df.columns else 'news'
    
    if text_col not in result_df.columns:
        logger.error(f"Text column '{text_col}' not found in the dataframe.")
        raise ValueError(f"Text column '{text_col}' not found in the dataframe.")
    
    # Handle missing values
    result_df[text_col] = result_df[text_col].fillna('')
    
    # Load or train Word2Vec model
    if word2vec_model is None:
        if os.path.exists(WORD2VEC_MODEL_PATH):
            logger.info(f"Loading Word2Vec model from {WORD2VEC_MODEL_PATH}")
            word2vec_model = Word2Vec.load(str(WORD2VEC_MODEL_PATH))
        else:
            logger.info("Training new Word2Vec model")
            word2vec_model = train_word2vec(result_df)
    
    # Get document embeddings
    logger.info("Generating document embeddings")
    doc_embeddings = result_df[text_col].apply(
        lambda x: get_document_embedding(x, word2vec_model)
    )
    
    # Convert embeddings to dataframe columns
    embedding_df = pd.DataFrame(
        doc_embeddings.tolist(),
        columns=[f'w2v_{i}' for i in range(word2vec_model.vector_size)]
    )
    
    # Concatenate with original dataframe
    result_df = pd.concat([result_df, embedding_df], axis=1)
    
    return result_df

def extract_stock_symbols(text):
    """
    Extract stock symbols from text.
    
    Args:
        text (str): Text to extract symbols from.
        
    Returns:
        list: Extracted stock symbols.
    """
    import re
    
    if not isinstance(text, str):
        return []
    
    # Simple pattern for stock symbols (uppercase letters, 1-5 characters)
    pattern = r'\b[A-Z]{1,5}\b'
    symbols = re.findall(pattern, text)
    
    # Filter out common words that might be mistaken for symbols
    common_words = {'A', 'I', 'AM', 'PM', 'CEO', 'CFO', 'CTO', 'COO', 'THE', 'FOR', 'AND', 'OR', 'IT', 'IS', 'BE'}
    symbols = [s for s in symbols if s not in common_words]
    
    return symbols

def extract_all_features(df=None):
    """
    Extract all features from the processed data.
    
    Args:
        df (pd.DataFrame, optional): Processed dataframe. If None, load from file.
        
    Returns:
        pd.DataFrame: Dataframe with all features.
    """
    # Load processed data if not provided
    if df is None:
        if not os.path.exists(PROCESSED_DATA_PATH):
            logger.error(f"Processed data file not found: {PROCESSED_DATA_PATH}")
            raise FileNotFoundError(f"Processed data file not found: {PROCESSED_DATA_PATH}")
        
        logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}")
        df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Extract features
    logger.info("Starting feature extraction")
    
    # Extract sentiment features
    df = extract_sentiment_features(df)
    
    # Extract text statistics
    df = extract_text_statistics(df)
    
    # Extract topic features
    df = extract_topic_features(df)
    
    # Train Word2Vec model and add features
    df = add_word2vec_features(df)
    
    # Save all features
    logger.info(f"Saving all features to {ALL_FEATURES_PATH}")
    df.to_csv(ALL_FEATURES_PATH, index=False)
    
    logger.info(f"Feature extraction completed. Extracted features for {len(df)} news items.")
    return df

if __name__ == "__main__":
    try:
        # Extract all features
        df = extract_all_features()
        
        # Print some statistics
        logger.info(f"Total samples with features: {len(df)}")
        logger.info(f"Feature columns: {df.columns.tolist()}")
    
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise 
=======
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'feature_engineering_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_tfidf_features(df, text_column, max_features=1000, ngram_range=(1, 2)):
    """
    Create TF-IDF features from text data
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data
        text_column (str): Name of the column containing the text
        max_features (int): Maximum number of features to extract
        ngram_range (tuple): Range of n-grams to consider
        
    Returns:
        tuple: (Sparse matrix with TF-IDF features, TF-IDF vectorizer)
    """
    logger.info(f"Creating TF-IDF features from {text_column} column")
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        dtype=np.float32  # Use float32 instead of float64 to reduce memory usage
    )
    
    # Fit and transform the text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column].fillna(''))
    
    logger.info(f"Created {tfidf_matrix.shape[1]} TF-IDF features")
    
    return tfidf_matrix, tfidf_vectorizer

def reduce_dimensions(tfidf_matrix, n_components=100):
    """
    Reduce dimensions of TF-IDF features using TruncatedSVD
    
    Args:
        tfidf_matrix (scipy.sparse.csr_matrix): Sparse matrix containing TF-IDF features
        n_components (int): Number of components to keep
        
    Returns:
        tuple: (DataFrame with reduced features, SVD model)
    """
    logger.info(f"Reducing dimensions from {tfidf_matrix.shape[1]} to {n_components}")
    
    # Initialize TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # Fit and transform the TF-IDF features
    svd_features = svd.fit_transform(tfidf_matrix)
    
    # Convert to DataFrame
    svd_df = pd.DataFrame(
        svd_features,
        columns=[f'svd_{i}' for i in range(n_components)]
    )
    
    logger.info(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    
    return svd_df, svd

def create_sentiment_features(df):
    """
    Create additional features from sentiment scores
    
    Args:
        df (pd.DataFrame): DataFrame containing sentiment scores
        
    Returns:
        pd.DataFrame: DataFrame with additional sentiment features
    """
    logger.info("Creating additional sentiment features")
    
    # Create a copy to avoid modifying the original
    sentiment_df = pd.DataFrame()
    
    # Sentiment polarity (positive - negative)
    sentiment_df['sentiment_polarity'] = df['pos'] - df['neg']
    
    # Sentiment magnitude (positive + negative)
    sentiment_df['sentiment_magnitude'] = df['pos'] + df['neg']
    
    # Sentiment ratio (positive / negative)
    sentiment_df['sentiment_ratio'] = df['pos'] / df['neg'].replace(0, 0.001)
    
    # Sentiment certainty (1 - neutral)
    sentiment_df['sentiment_certainty'] = 1 - df['neu']
    
    logger.info("Created 4 additional sentiment features")
    
    return sentiment_df

def create_time_features(df, date_column='date'):
    """
    Create time-based features from date column
    
    Args:
        df (pd.DataFrame): DataFrame containing the date column
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with time-based features
    """
    logger.info(f"Creating time-based features from {date_column} column")
    
    # Create a copy to avoid modifying the original
    time_df = pd.DataFrame()
    
    # Convert date column to datetime if it's not already
    dates = pd.to_datetime(df[date_column])
    
    # Extract time components
    time_df['year'] = dates.dt.year
    time_df['month'] = dates.dt.month
    time_df['day'] = dates.dt.day
    time_df['day_of_week'] = dates.dt.dayofweek
    time_df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
    time_df['quarter'] = dates.dt.quarter
    
    logger.info("Created 6 time-based features")
    
    return time_df

def create_company_features(df, company_column='mentioned_companies'):
    """
    Create features based on mentioned companies
    
    Args:
        df (pd.DataFrame): DataFrame containing the company column
        company_column (str): Name of the column containing mentioned companies
        
    Returns:
        pd.DataFrame: DataFrame with company-based features
    """
    if company_column not in df.columns:
        logger.warning(f"Column {company_column} not found. Skipping company features.")
        return pd.DataFrame(index=df.index)
    
    logger.info(f"Creating company-based features from {company_column} column")
    
    # Create a copy to avoid modifying the original
    company_df = pd.DataFrame()
    
    # Number of companies mentioned
    company_df['num_companies'] = df[company_column].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # One-hot encoding for top companies (if needed)
    # This would require additional processing to identify top companies
    
    logger.info("Created 1 company-based feature")
    
    return company_df

def combine_features(df, feature_dfs):
    """
    Combine all feature DataFrames with the original DataFrame
    
    Args:
        df (pd.DataFrame): Original DataFrame
        feature_dfs (list): List of feature DataFrames to combine
        
    Returns:
        pd.DataFrame: Combined DataFrame with all features
    """
    logger.info("Combining all features")
    
    # Create a copy of the original DataFrame
    combined_df = df.copy()
    
    # Add each feature DataFrame
    for feature_df in feature_dfs:
        if not feature_df.empty:
            # Reset index to ensure proper alignment
            feature_df = feature_df.reset_index(drop=True)
            combined_df = pd.concat([combined_df, feature_df], axis=1)
    
    logger.info(f"Final DataFrame shape: {combined_df.shape}")
    
    return combined_df

def save_features(df, output_path):
    """
    Save feature DataFrame to CSV
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the DataFrame
    """
    try:
        logger.info(f"Saving features to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Features saved successfully")
    except Exception as e:
        logger.error(f"Error saving features: {e}")
        raise

def save_models(models, output_dir):
    """
    Save feature engineering models
    
    Args:
        models (dict): Dictionary of models to save
        output_dir (str): Directory to save the models
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving models to {output_dir}")
        
        for name, model in models.items():
            model_path = os.path.join(output_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Model {name} saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        raise

def run_feature_engineering(input_path, output_path, models_dir):
    """
    Run the complete feature engineering pipeline
    
    Args:
        input_path (str): Path to the processed data
        output_path (str): Path to save the feature data
        models_dir (str): Directory to save the feature engineering models
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting feature engineering pipeline")
    
    # Load processed data
    try:
        logger.info(f"Loading processed data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # Create TF-IDF features
    tfidf_matrix, tfidf_vectorizer = create_tfidf_features(df, 'preprocessed_text')
    
    # Reduce dimensions
    svd_df, svd_model = reduce_dimensions(tfidf_matrix)
    
    # Create sentiment features
    sentiment_df = create_sentiment_features(df)
    
    # Create time features
    time_df = create_time_features(df)
    
    # Create company features if available
    company_df = create_company_features(df)
    
    # Combine all features
    feature_dfs = [svd_df, sentiment_df, time_df, company_df]
    combined_df = combine_features(df, feature_dfs)
    
    # Save features
    save_features(combined_df, output_path)
    
    # Save models
    models = {
        'tfidf_vectorizer': tfidf_vectorizer,
        'svd_model': svd_model
    }
    save_models(models, models_dir)
    
    logger.info("Feature engineering pipeline completed")
    return combined_df

if __name__ == "__main__":
    # Example usage
    input_path = "data/processed/processed_news.csv"
    output_path = "data/interim/features.csv"
    models_dir = "models/feature_engineering"
    
    run_feature_engineering(input_path, output_path, models_dir) 
>>>>>>> 42c49fa71cbd94cde7e31240fce9a881867433a4
