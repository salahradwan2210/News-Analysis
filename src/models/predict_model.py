import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'prediction_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_models(sentiment_model_path, impact_model_path):
    """
    Load trained models
    
    Args:
        sentiment_model_path (str): Path to the sentiment model
        impact_model_path (str): Path to the impact model
        
    Returns:
        tuple: (sentiment_model, impact_model)
    """
    try:
        logger.info(f"Loading sentiment model from {sentiment_model_path}")
        sentiment_model = joblib.load(sentiment_model_path)
        
        logger.info(f"Loading impact model from {impact_model_path}")
        impact_model = joblib.load(impact_model_path)
        
        logger.info("Models loaded successfully")
        return sentiment_model, impact_model
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def load_feature_engineering_models(models_dir):
    """
    Load feature engineering models
    
    Args:
        models_dir (str): Directory containing the feature engineering models
        
    Returns:
        dict: Dictionary of feature engineering models
    """
    try:
        logger.info(f"Loading feature engineering models from {models_dir}")
        
        tfidf_vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        svd_model_path = os.path.join(models_dir, "svd_model.joblib")
        
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        svd_model = joblib.load(svd_model_path)
        
        models = {
            'tfidf_vectorizer': tfidf_vectorizer,
            'svd_model': svd_model
        }
        
        logger.info("Feature engineering models loaded successfully")
        return models
    except Exception as e:
        logger.error(f"Error loading feature engineering models: {e}")
        raise

def preprocess_news(news_data, feature_engineering_models):
    """
    Preprocess news data for prediction
    
    Args:
        news_data (pd.DataFrame): DataFrame containing news data
        feature_engineering_models (dict): Dictionary of feature engineering models
        
    Returns:
        pd.DataFrame: Preprocessed data with features
    """
    logger.info("Preprocessing news data for prediction")
    
    # Extract feature engineering models
    tfidf_vectorizer = feature_engineering_models['tfidf_vectorizer']
    svd_model = feature_engineering_models['svd_model']
    
    # Create TF-IDF features
    tfidf_matrix = tfidf_vectorizer.transform(news_data['preprocessed_text'].fillna(''))
    
    # Reduce dimensions using SVD
    svd_features = svd_model.transform(tfidf_matrix)
    
    # Convert to DataFrame
    svd_df = pd.DataFrame(
        svd_features,
        columns=[f'svd_{i}' for i in range(svd_features.shape[1])]
    )
    
    # Create sentiment features
    sentiment_df = pd.DataFrame()
    sentiment_df['sentiment_polarity'] = news_data['pos'] - news_data['neg']
    sentiment_df['sentiment_magnitude'] = news_data['pos'] + news_data['neg']
    sentiment_df['sentiment_ratio'] = news_data['pos'] / news_data['neg'].replace(0, 0.001)
    sentiment_df['sentiment_certainty'] = 1 - news_data['neu']
    
    # Create time features if date column exists
    if 'date' in news_data.columns:
        time_df = pd.DataFrame()
        dates = pd.to_datetime(news_data['date'])
        time_df['year'] = dates.dt.year
        time_df['month'] = dates.dt.month
        time_df['day'] = dates.dt.day
        time_df['day_of_week'] = dates.dt.dayofweek
        time_df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        time_df['quarter'] = dates.dt.quarter
    else:
        time_df = pd.DataFrame()
    
    # Combine all features
    feature_dfs = [svd_df, sentiment_df]
    if not time_df.empty:
        feature_dfs.append(time_df)
    
    # Combine features
    features = pd.concat(feature_dfs, axis=1)
    
    logger.info(f"Preprocessed features shape: {features.shape}")
    
    return features

def predict_sentiment(features, sentiment_model):
    """
    Predict sentiment from features
    
    Args:
        features (pd.DataFrame): Features for prediction
        sentiment_model (object): Trained sentiment model
        
    Returns:
        tuple: (sentiment_predictions, sentiment_probabilities)
    """
    logger.info("Predicting sentiment")
    
    # Make predictions
    sentiment_pred = sentiment_model.predict(features)
    sentiment_proba = sentiment_model.predict_proba(features)
    
    # Map numeric predictions to sentiment labels
    sentiment_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
    sentiment_labels = [sentiment_map[pred] for pred in sentiment_pred]
    
    logger.info(f"Sentiment predictions: {len(sentiment_labels)} items")
    
    return sentiment_labels, sentiment_proba

def predict_impact(features, impact_model):
    """
    Predict impact score from features
    
    Args:
        features (pd.DataFrame): Features for prediction
        impact_model (object): Trained impact model
        
    Returns:
        np.ndarray: Impact score predictions
    """
    logger.info("Predicting impact scores")
    
    # Make predictions
    impact_scores = impact_model.predict(features)
    
    # Ensure scores are within 0-100 range
    impact_scores = np.clip(impact_scores, 0, 100)
    
    logger.info(f"Impact score predictions: {len(impact_scores)} items")
    
    return impact_scores

def rank_news_by_impact(news_data, impact_scores):
    """
    Rank news by impact score
    
    Args:
        news_data (pd.DataFrame): DataFrame containing news data
        impact_scores (np.ndarray): Impact score predictions
        
    Returns:
        pd.DataFrame: Ranked news data
    """
    logger.info("Ranking news by impact score")
    
    # Add impact scores to news data
    ranked_news = news_data.copy()
    ranked_news['impact_score'] = impact_scores
    
    # Sort by impact score in descending order
    ranked_news = ranked_news.sort_values('impact_score', ascending=False)
    
    logger.info("News ranked successfully")
    
    return ranked_news

def rank_stocks_by_impact(ranked_news, top_n=10):
    """
    Rank stocks by aggregated impact scores
    
    Args:
        ranked_news (pd.DataFrame): Ranked news data
        top_n (int): Number of top stocks to return
        
    Returns:
        pd.DataFrame: Ranked stocks
    """
    logger.info("Ranking stocks by aggregated impact scores")
    
    # Check if mentioned_companies column exists
    if 'mentioned_companies' not in ranked_news.columns:
        logger.warning("No mentioned_companies column found. Cannot rank stocks.")
        return pd.DataFrame()
    
    # Explode the mentioned_companies lists to get one row per company mention
    exploded_df = ranked_news.explode('mentioned_companies')
    
    # Remove rows with no company mentions
    exploded_df = exploded_df[exploded_df['mentioned_companies'].notna()]
    
    # Group by company and aggregate impact scores
    stock_rankings = exploded_df.groupby('mentioned_companies').agg({
        'impact_score': ['mean', 'count'],
        'sentiment': lambda x: x.value_counts().index[0]  # Most common sentiment
    })
    
    # Flatten multi-level columns
    stock_rankings.columns = ['avg_impact_score', 'mention_count', 'most_common_sentiment']
    
    # Calculate a combined score (impact score * mention count)
    stock_rankings['combined_score'] = stock_rankings['avg_impact_score'] * np.log1p(stock_rankings['mention_count'])
    
    # Sort by combined score in descending order
    stock_rankings = stock_rankings.sort_values('combined_score', ascending=False)
    
    # Get top N stocks
    top_stocks = stock_rankings.head(top_n)
    
    logger.info(f"Top {len(top_stocks)} stocks ranked by impact")
    
    return top_stocks

def format_predictions(news_data, sentiment_labels, impact_scores):
    """
    Format predictions into a structured output
    
    Args:
        news_data (pd.DataFrame): Original news data
        sentiment_labels (list): Predicted sentiment labels
        impact_scores (np.ndarray): Predicted impact scores
        
    Returns:
        pd.DataFrame: Formatted predictions
    """
    logger.info("Formatting predictions")
    
    # Create a copy of the news data
    predictions = news_data.copy()
    
    # Add predictions
    predictions['predicted_sentiment'] = sentiment_labels
    predictions['impact_score'] = impact_scores
    
    # Select relevant columns
    if 'date' in predictions.columns:
        output_cols = ['date', 'news', 'predicted_sentiment', 'impact_score']
    else:
        output_cols = ['news', 'predicted_sentiment', 'impact_score']
    
    # Add mentioned_companies if available
    if 'mentioned_companies' in predictions.columns:
        output_cols.append('mentioned_companies')
    
    # Select columns
    formatted_predictions = predictions[output_cols]
    
    logger.info("Predictions formatted successfully")
    
    return formatted_predictions

def save_predictions(predictions, output_path):
    """
    Save predictions to CSV
    
    Args:
        predictions (pd.DataFrame): Formatted predictions
        output_path (str): Path to save the predictions
    """
    try:
        logger.info(f"Saving predictions to {output_path}")
        predictions.to_csv(output_path, index=False)
        logger.info("Predictions saved successfully")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise

def save_stock_rankings(stock_rankings, output_path):
    """
    Save stock rankings to CSV
    
    Args:
        stock_rankings (pd.DataFrame): Ranked stocks
        output_path (str): Path to save the rankings
    """
    try:
        logger.info(f"Saving stock rankings to {output_path}")
        stock_rankings.to_csv(output_path, index=True)
        logger.info("Stock rankings saved successfully")
    except Exception as e:
        logger.error(f"Error saving stock rankings: {e}")
        raise

def run_prediction(news_data_path, sentiment_model_path, impact_model_path, 
                  feature_models_dir, predictions_output_path, rankings_output_path):
    """
    Run the complete prediction pipeline
    
    Args:
        news_data_path (str): Path to the news data
        sentiment_model_path (str): Path to the sentiment model
        impact_model_path (str): Path to the impact model
        feature_models_dir (str): Directory containing feature engineering models
        predictions_output_path (str): Path to save the predictions
        rankings_output_path (str): Path to save the stock rankings
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting prediction pipeline")
    
    # Load models
    sentiment_model, impact_model = load_models(sentiment_model_path, impact_model_path)
    feature_engineering_models = load_feature_engineering_models(feature_models_dir)
    
    # Load news data
    try:
        logger.info(f"Loading news data from {news_data_path}")
        news_data = pd.read_csv(news_data_path)
        logger.info(f"News data loaded successfully. Shape: {news_data.shape}")
    except Exception as e:
        logger.error(f"Error loading news data: {e}")
        raise
    
    # Preprocess news data
    features = preprocess_news(news_data, feature_engineering_models)
    
    # Predict sentiment
    sentiment_labels, sentiment_proba = predict_sentiment(features, sentiment_model)
    
    # Predict impact
    impact_scores = predict_impact(features, impact_model)
    
    # Format predictions
    predictions = format_predictions(news_data, sentiment_labels, impact_scores)
    
    # Rank news by impact
    ranked_news = rank_news_by_impact(predictions, impact_scores)
    
    # Rank stocks by impact
    stock_rankings = rank_stocks_by_impact(ranked_news)
    
    # Save predictions
    save_predictions(ranked_news, predictions_output_path)
    
    # Save stock rankings if available
    if not stock_rankings.empty:
        save_stock_rankings(stock_rankings, rankings_output_path)
    
    logger.info("Prediction pipeline completed")
    
    return ranked_news, stock_rankings

if __name__ == "__main__":
    # Example usage
    news_data_path = "data/processed/processed_news.csv"
    sentiment_model_path = "models/sentiment_model.joblib"
    impact_model_path = "models/impact_model.joblib"
    feature_models_dir = "models/feature_engineering"
    predictions_output_path = "data/processed/predictions.csv"
    rankings_output_path = "data/processed/stock_rankings.csv"
    
    run_prediction(news_data_path, sentiment_model_path, impact_model_path,
                  feature_models_dir, predictions_output_path, rankings_output_path) 