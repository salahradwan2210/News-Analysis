import os
import sys
import logging
from datetime import datetime
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import pandas as pd
import yfinance as yf

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from src.data.data_processing import run_data_processing
from src.features.feature_engineering import run_feature_engineering
from src.models.train_model import run_model_training
from src.models.predict_model import run_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'pipeline_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@task
def fetch_latest_news(symbols=None, days=7):
    """
    Fetch latest financial news for given stock symbols
    
    Args:
        symbols (list): List of stock symbols to fetch news for
        days (int): Number of days to look back
        
    Returns:
        pd.DataFrame: DataFrame containing latest news
    """
    logger.info(f"Fetching latest news for {len(symbols) if symbols else 'all'} symbols")
    
    # This is a placeholder. In a real implementation, you would:
    # 1. Use a news API (e.g., Alpha Vantage, News API, etc.)
    # 2. Fetch news for the specified symbols
    # 3. Process and return as a DataFrame
    
    # For demonstration, we'll use the existing dataset
    try:
        news_df = pd.read_csv("news.csv")
        logger.info(f"Loaded {len(news_df)} news items from existing dataset")
        return news_df
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        # Create a minimal dataset for demonstration
        return pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')],
            'news': ['This is a placeholder news item for demonstration purposes.'],
            'neg': [0.1],
            'neu': [0.5],
            'pos': [0.4],
            'compound': [0.3],
            'sentiment': ['POSITIVE']
        })

@task
def fetch_stock_data(symbols, days=30):
    """
    Fetch stock price data for given symbols
    
    Args:
        symbols (list): List of stock symbols to fetch data for
        days (int): Number of days to look back
        
    Returns:
        dict: Dictionary mapping symbols to price data
    """
    logger.info(f"Fetching stock data for {len(symbols)} symbols")
    
    stock_data = {}
    
    for symbol in symbols:
        try:
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if not hist.empty:
                stock_data[symbol] = hist
                logger.info(f"Fetched data for {symbol}: {len(hist)} days")
            else:
                logger.warning(f"No data found for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return stock_data

@task
def process_data(news_df, raw_path, processed_path, company_list_path=None):
    """
    Process the news data
    
    Args:
        news_df (pd.DataFrame): News data
        raw_path (str): Path to save raw data
        processed_path (str): Path to save processed data
        company_list_path (str): Path to company list
        
    Returns:
        pd.DataFrame: Processed data
    """
    logger.info("Processing news data")
    
    # Save raw data
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    news_df.to_csv(raw_path, index=False)
    
    # Process data
    processed_df = run_data_processing(raw_path, processed_path, company_list_path)
    
    return processed_df

@task
def engineer_features(processed_path, features_path, models_dir):
    """
    Engineer features from processed data
    
    Args:
        processed_path (str): Path to processed data
        features_path (str): Path to save features
        models_dir (str): Directory to save feature engineering models
        
    Returns:
        pd.DataFrame: Feature data
    """
    logger.info("Engineering features")
    
    # Run feature engineering
    feature_df = run_feature_engineering(processed_path, features_path, models_dir)
    
    return feature_df

@task
def train_models(features_path, sentiment_model_path, impact_model_path):
    """
    Train sentiment and impact models
    
    Args:
        features_path (str): Path to feature data
        sentiment_model_path (str): Path to save sentiment model
        impact_model_path (str): Path to save impact model
        
    Returns:
        tuple: (sentiment_model, impact_model)
    """
    logger.info("Training models")
    
    # Run model training
    sentiment_model, impact_model = run_model_training(
        features_path, sentiment_model_path, impact_model_path
    )
    
    return sentiment_model, impact_model

@task
def make_predictions(processed_path, sentiment_model_path, impact_model_path, 
                    feature_models_dir, predictions_path, rankings_path):
    """
    Make predictions using trained models
    
    Args:
        processed_path (str): Path to processed data
        sentiment_model_path (str): Path to sentiment model
        impact_model_path (str): Path to impact model
        feature_models_dir (str): Directory containing feature engineering models
        predictions_path (str): Path to save predictions
        rankings_path (str): Path to save rankings
        
    Returns:
        tuple: (predictions, rankings)
    """
    logger.info("Making predictions")
    
    # Run prediction
    predictions, rankings = run_prediction(
        processed_path, sentiment_model_path, impact_model_path,
        feature_models_dir, predictions_path, rankings_path
    )
    
    return predictions, rankings

@task
def generate_investment_recommendations(rankings_path, stock_data, output_path):
    """
    Generate investment recommendations based on rankings and stock data
    
    Args:
        rankings_path (str): Path to stock rankings
        stock_data (dict): Dictionary of stock price data
        output_path (str): Path to save recommendations
        
    Returns:
        pd.DataFrame: Investment recommendations
    """
    logger.info("Generating investment recommendations")
    
    try:
        # Load rankings
        rankings = pd.read_csv(rankings_path)
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame()
        
        # Add rankings data
        recommendations['symbol'] = rankings['mentioned_companies']
        recommendations['impact_score'] = rankings['avg_impact_score']
        recommendations['sentiment'] = rankings['most_common_sentiment']
        
        # Add price data
        recommendations['current_price'] = None
        recommendations['price_change_pct'] = None
        
        for i, symbol in enumerate(recommendations['symbol']):
            if symbol in stock_data:
                price_data = stock_data[symbol]
                if not price_data.empty:
                    current_price = price_data['Close'].iloc[-1]
                    prev_price = price_data['Close'].iloc[0]
                    price_change = (current_price - prev_price) / prev_price * 100
                    
                    recommendations.at[i, 'current_price'] = current_price
                    recommendations.at[i, 'price_change_pct'] = price_change
        
        # Calculate recommendation score
        # Higher impact score and positive sentiment increase recommendation score
        recommendations['recommendation_score'] = recommendations['impact_score']
        recommendations.loc[recommendations['sentiment'] == 'NEGATIVE', 'recommendation_score'] *= 0.5
        
        # Sort by recommendation score
        recommendations = recommendations.sort_values('recommendation_score', ascending=False)
        
        # Add recommendation category
        recommendations['recommendation'] = 'HOLD'
        recommendations.loc[recommendations['recommendation_score'] > 75, 'recommendation'] = 'STRONG BUY'
        recommendations.loc[(recommendations['recommendation_score'] > 60) & 
                           (recommendations['recommendation_score'] <= 75), 'recommendation'] = 'BUY'
        recommendations.loc[recommendations['recommendation_score'] < 40, 'recommendation'] = 'SELL'
        recommendations.loc[recommendations['recommendation_score'] < 25, 'recommendation'] = 'STRONG SELL'
        
        # Save recommendations
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        recommendations.to_csv(output_path, index=False)
        
        logger.info(f"Generated {len(recommendations)} investment recommendations")
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()

@flow(task_runner=SequentialTaskRunner())
def financial_news_analysis_pipeline(symbols=None, retrain_models=True):
    """
    Run the complete financial news analysis pipeline
    
    Args:
        symbols (list): List of stock symbols to analyze
        retrain_models (bool): Whether to retrain models or use existing ones
    """
    logger.info("Starting financial news analysis pipeline")
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/interim', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/feature_engineering', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Define paths
    raw_path = "data/raw/latest_news.csv"
    processed_path = "data/processed/processed_news.csv"
    features_path = "data/interim/features.csv"
    sentiment_model_path = "models/sentiment_model.joblib"
    impact_model_path = "models/impact_model.joblib"
    feature_models_dir = "models/feature_engineering"
    predictions_path = "data/processed/predictions.csv"
    rankings_path = "data/processed/stock_rankings.csv"
    recommendations_path = "reports/investment_recommendations.csv"
    
    # Fetch latest news
    news_df = fetch_latest_news(symbols)
    
    # Process data
    processed_df = process_data(news_df, raw_path, processed_path)
    
    # Engineer features
    feature_df = engineer_features(processed_path, features_path, feature_models_dir)
    
    # Train or load models
    if retrain_models:
        sentiment_model, impact_model = train_models(
            features_path, sentiment_model_path, impact_model_path
        )
    
    # Make predictions
    predictions, rankings = make_predictions(
        processed_path, sentiment_model_path, impact_model_path,
        feature_models_dir, predictions_path, rankings_path
    )
    
    # Extract symbols from rankings
    if not rankings.empty:
        ranked_symbols = rankings.index.tolist()
    else:
        ranked_symbols = symbols if symbols else []
    
    # Fetch stock data
    stock_data = fetch_stock_data(ranked_symbols)
    
    # Generate investment recommendations
    recommendations = generate_investment_recommendations(
        rankings_path, stock_data, recommendations_path
    )
    
    logger.info("Financial news analysis pipeline completed")
    
    return recommendations

if __name__ == "__main__":
    # Example usage
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    financial_news_analysis_pipeline(symbols=symbols, retrain_models=True) 