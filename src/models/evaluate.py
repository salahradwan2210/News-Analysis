"""
Model evaluation module for financial news analysis.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "processed_news.csv"
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features" / "all_features.csv"
MODELS_DIR = ROOT_DIR / "models"
EVALUATION_DIR = ROOT_DIR / "evaluation"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_lstm.h5"
IMPACT_MODEL_PATH = MODELS_DIR / "impact_gru.h5"

# Ensure evaluation directory exists
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_sentiment_model(model=None, df=None, model_type="lstm", use_sample=False):
    """
    Evaluate a sentiment analysis model.
    
    Args:
        model: Trained sentiment model. If None, load from file.
        df (pd.DataFrame, optional): Processed dataframe. If None, load from file.
        model_type (str): Type of model to evaluate ('lstm', 'gru', or 'cnn_rnn').
        use_sample (bool): Whether to use a sample of the data for evaluation.
        
    Returns:
        dict: Evaluation metrics.
    """
    logger.info(f"Evaluating sentiment model ({model_type})")
    
    # Load processed data if not provided
    if df is None:
        if not os.path.exists(FEATURES_PATH):
            logger.error(f"Features file not found: {FEATURES_PATH}")
            raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
        
        logger.info(f"Loading features from {FEATURES_PATH}")
        df = pd.read_csv(FEATURES_PATH)
    
    # Load model if not provided
    if model is None:
        from src.models.deep_learning import load_trained_model
        model_path = MODELS_DIR / f"sentiment_{model_type}.h5"
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Training new model.")
            from src.models.train import train_sentiment_model
            train_sentiment_model(model_type=model_type, use_sample=use_sample)
        
        logger.info(f"Loading model from {model_path}")
        model, tokenizer, label_encoder = load_trained_model(f"sentiment_{model_type}")
    else:
        # Extract tokenizer and label_encoder from model
        from src.models.deep_learning import load_trained_model
        _, tokenizer, label_encoder = load_trained_model(f"sentiment_{model_type}")
    
    # Determine text column
    text_col = 'processed_content' if 'processed_content' in df.columns else 'news'
    
    # Determine target column
    if 'sentiment' in df.columns:
        target_col = 'sentiment'
    elif 'sentiment_category' in df.columns:
        target_col = 'sentiment_category'
    else:
        logger.error("No sentiment column found in the dataframe.")
        raise ValueError("No sentiment column found in the dataframe.")
    
    # Split data
    from src.data_processing.preprocess import split_data
    _, _, X_test, _, _, y_test = split_data(df, target_col=target_col)
    
    # Make predictions
    from src.models.deep_learning import predict
    y_pred_proba, y_pred = predict(model, X_test.tolist(), tokenizer, label_encoder)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    if len(np.unique(y_test)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Sentiment Model ({model_type})')
    
    # Save confusion matrix
    cm_path = EVALUATION_DIR / f"sentiment_{model_type}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    # Save evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix_path': str(cm_path)
    }
    
    metrics_path = EVALUATION_DIR / f"sentiment_{model_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Sentiment model evaluation completed and saved to {metrics_path}")
    logger.info(f"Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
    
    return metrics

def evaluate_impact_model(model=None, df=None, model_type="gru", use_sample=False):
    """
    Evaluate an impact prediction model.
    
    Args:
        model: Trained impact model. If None, load from file.
        df (pd.DataFrame, optional): Processed dataframe. If None, load from file.
        model_type (str): Type of model to evaluate ('lstm', 'gru', or 'cnn_rnn').
        use_sample (bool): Whether to use a sample of the data for evaluation.
        
    Returns:
        dict: Evaluation metrics.
    """
    logger.info(f"Evaluating impact model ({model_type})")
    
    # Load processed data if not provided
    if df is None:
        if not os.path.exists(FEATURES_PATH):
            logger.error(f"Features file not found: {FEATURES_PATH}")
            raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
        
        logger.info(f"Loading features from {FEATURES_PATH}")
        df = pd.read_csv(FEATURES_PATH)
    
    # Create impact categories if they don't exist
    if 'compound' in df.columns and 'impact_category' not in df.columns:
        logger.info("Creating impact categories from compound scores")
        
        # Use compound score as a proxy for impact
        df['impact_score'] = df['compound'].apply(lambda x: abs(x) * 100)
        
        # Create impact categories
        df['impact_category'] = pd.cut(
            df['impact_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['very low', 'low', 'medium', 'high', 'very high']
        )
    
    # Ensure we have impact categories
    if 'impact_category' not in df.columns:
        logger.error("No impact category column found and couldn't create one.")
        raise ValueError("No impact category column found and couldn't create one.")
    
    # Load model if not provided
    if model is None:
        from src.models.deep_learning import load_trained_model
        model_path = MODELS_DIR / f"impact_{model_type}.h5"
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Training new model.")
            from src.models.train import train_impact_model
            train_impact_model(model_type=model_type, use_sample=use_sample)
        
        logger.info(f"Loading model from {model_path}")
        model, tokenizer, label_encoder = load_trained_model(f"impact_{model_type}")
    else:
        # Extract tokenizer and label_encoder from model
        from src.models.deep_learning import load_trained_model
        _, tokenizer, label_encoder = load_trained_model(f"impact_{model_type}")
    
    # Determine text column
    text_col = 'processed_content' if 'processed_content' in df.columns else 'news'
    
    # Split data
    from src.data_processing.preprocess import split_data
    _, _, X_test, _, _, y_test = split_data(df, target_col='impact_category')
    
    # Make predictions
    from src.models.deep_learning import predict
    y_pred_proba, y_pred = predict(model, X_test.tolist(), tokenizer, label_encoder)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Impact Model ({model_type})')
    
    # Save confusion matrix
    cm_path = EVALUATION_DIR / f"impact_{model_type}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    # Save evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix_path': str(cm_path)
    }
    
    metrics_path = EVALUATION_DIR / f"impact_{model_type}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Impact model evaluation completed and saved to {metrics_path}")
    logger.info(f"Accuracy: {accuracy:.4f}, F1 score: {f1:.4f}")
    
    return metrics

def evaluate_ranking_model(df=None, ranking_file=None):
    """
    Evaluate the stock ranking model.
    
    Args:
        df (pd.DataFrame, optional): Processed dataframe. If None, load from file.
        ranking_file (str, optional): Path to the ranking file. If None, use default.
        
    Returns:
        dict: Evaluation metrics.
    """
    logger.info("Evaluating stock ranking model")
    
    # Load processed data if not provided
    if df is None:
        if not os.path.exists(FEATURES_PATH):
            logger.error(f"Features file not found: {FEATURES_PATH}")
            raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
        
        logger.info(f"Loading features from {FEATURES_PATH}")
        df = pd.read_csv(FEATURES_PATH)
    
    # Load ranking file if not provided
    if ranking_file is None:
        ranking_file = MODELS_DIR / "stock_ranking.json"
        if not os.path.exists(ranking_file):
            logger.error(f"Ranking file not found: {ranking_file}")
            raise FileNotFoundError(f"Ranking file not found: {ranking_file}")
    
    logger.info(f"Loading ranking file from {ranking_file}")
    with open(ranking_file, 'r') as f:
        ranking_data = json.load(f)
    
    # Extract ranked stocks
    ranked_stocks = ranking_data.get('ranked_stocks', [])
    
    if not ranked_stocks:
        logger.error("No ranked stocks found in the ranking file.")
        raise ValueError("No ranked stocks found in the ranking file.")
    
    # Calculate metrics
    num_stocks = len(ranked_stocks)
    avg_news_count = np.mean([stock.get('news_count', 0) for stock in ranked_stocks])
    avg_sentiment = np.mean([stock.get('overall_sentiment_mean', 0) for stock in ranked_stocks])
    avg_impact = np.mean([stock.get('impact_score_mean', 0) for stock in ranked_stocks])
    
    # Calculate sentiment distribution
    positive_stocks = sum(1 for stock in ranked_stocks if stock.get('overall_sentiment_mean', 0) > 0.1)
    negative_stocks = sum(1 for stock in ranked_stocks if stock.get('overall_sentiment_mean', 0) < -0.1)
    neutral_stocks = num_stocks - positive_stocks - negative_stocks
    
    sentiment_distribution = {
        'positive': positive_stocks / num_stocks if num_stocks > 0 else 0,
        'negative': negative_stocks / num_stocks if num_stocks > 0 else 0,
        'neutral': neutral_stocks / num_stocks if num_stocks > 0 else 0
    }
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    plt.bar(['Positive', 'Neutral', 'Negative'], 
            [sentiment_distribution['positive'], 
             sentiment_distribution['neutral'], 
             sentiment_distribution['negative']])
    plt.title('Sentiment Distribution of Ranked Stocks')
    plt.ylabel('Proportion')
    
    # Save sentiment distribution
    sentiment_dist_path = EVALUATION_DIR / "ranking_sentiment_distribution.png"
    plt.savefig(sentiment_dist_path)
    plt.close()
    
    # Plot top 10 stocks by ranking score
    top_stocks = sorted(ranked_stocks, key=lambda x: x.get('ranking_score', 0), reverse=True)[:10]
    
    plt.figure(figsize=(12, 8))
    plt.barh([stock.get('symbol', f"Stock {i}") for i, stock in enumerate(top_stocks)],
             [stock.get('ranking_score', 0) for stock in top_stocks])
    plt.title('Top 10 Stocks by Ranking Score')
    plt.xlabel('Ranking Score')
    plt.gca().invert_yaxis()  # Highest score at the top
    
    # Save top stocks plot
    top_stocks_path = EVALUATION_DIR / "top_stocks_ranking.png"
    plt.savefig(top_stocks_path)
    plt.close()
    
    # Save evaluation metrics
    metrics = {
        'num_stocks': num_stocks,
        'avg_news_count': avg_news_count,
        'avg_sentiment': avg_sentiment,
        'avg_impact': avg_impact,
        'sentiment_distribution': sentiment_distribution,
        'sentiment_distribution_path': str(sentiment_dist_path),
        'top_stocks_path': str(top_stocks_path)
    }
    
    metrics_path = EVALUATION_DIR / "ranking_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Ranking model evaluation completed and saved to {metrics_path}")
    logger.info(f"Number of ranked stocks: {num_stocks}")
    
    return metrics

def evaluate_all_models(df=None, use_sample=False):
    """
    Evaluate all models.
    
    Args:
        df (pd.DataFrame, optional): Processed dataframe. If None, load from file.
        use_sample (bool): Whether to use a sample of the data for evaluation.
        
    Returns:
        dict: Evaluation metrics for all models.
    """
    logger.info("Evaluating all models")
    
    # Load processed data if not provided
    if df is None:
        if not os.path.exists(FEATURES_PATH):
            logger.info("Features not found. Processing data and extracting features...")
            
            # Process data
            from src.data_processing.preprocess import process_data
            df = process_data(use_sample=use_sample)
            
            # Extract features
            from src.features.feature_engineering import extract_all_features
            df = extract_all_features(df)
        else:
            logger.info(f"Loading features from {FEATURES_PATH}")
            df = pd.read_csv(FEATURES_PATH)
    
    # Evaluate sentiment model
    try:
        sentiment_metrics = evaluate_sentiment_model(df=df, model_type="lstm", use_sample=use_sample)
    except Exception as e:
        logger.error(f"Error evaluating sentiment model: {str(e)}")
        sentiment_metrics = {"error": str(e)}
    
    # Evaluate impact model
    try:
        impact_metrics = evaluate_impact_model(df=df, model_type="gru", use_sample=use_sample)
    except Exception as e:
        logger.error(f"Error evaluating impact model: {str(e)}")
        impact_metrics = {"error": str(e)}
    
    # Evaluate ranking model
    try:
        ranking_metrics = evaluate_ranking_model(df=df)
    except Exception as e:
        logger.error(f"Error evaluating ranking model: {str(e)}")
        ranking_metrics = {"error": str(e)}
    
    # Combine all metrics
    all_metrics = {
        'sentiment_model': sentiment_metrics,
        'impact_model': impact_metrics,
        'ranking_model': ranking_metrics
    }
    
    # Save all metrics
    all_metrics_path = EVALUATION_DIR / "all_models_metrics.json"
    with open(all_metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    logger.info(f"All models evaluation completed and saved to {all_metrics_path}")
    
    return all_metrics

if __name__ == "__main__":
    try:
        # Evaluate all models
        metrics = evaluate_all_models(use_sample=False)
        
        # Print metrics summary
        for model_name, model_metrics in metrics.items():
            if "error" in model_metrics:
                logger.error(f"{model_name}: {model_metrics['error']}")
            else:
                if model_name == 'sentiment_model' or model_name == 'impact_model':
                    logger.info(f"{model_name}: Accuracy = {model_metrics.get('accuracy', 'N/A'):.4f}, "
                               f"F1 = {model_metrics.get('f1_score', 'N/A'):.4f}")
                elif model_name == 'ranking_model':
                    logger.info(f"{model_name}: Ranked {model_metrics.get('num_stocks', 'N/A')} stocks")
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise 