import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'model_training_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_features(features_path):
    """
    Load features from CSV file
    
    Args:
        features_path (str): Path to the features CSV file
        
    Returns:
        pd.DataFrame: Loaded features
    """
    try:
        logger.info(f"Loading features from {features_path}")
        df = pd.read_csv(features_path)
        logger.info(f"Features loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        raise

def prepare_data(df, target_col='sentiment', test_size=0.2, random_state=42):
    """
    Prepare data for model training by splitting into train and test sets
    
    Args:
        df (pd.DataFrame): DataFrame containing features and target
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing data for model training")
    
    # Convert target to numeric if it's categorical
    if df[target_col].dtype == 'object':
        # Map sentiment labels to numeric values
        sentiment_map = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
        y = df[target_col].map(sentiment_map)
    else:
        y = df[target_col]
    
    # Select feature columns (exclude non-feature columns)
    non_feature_cols = ['date', 'news', 'cleaned_text', 'preprocessed_text', 
                        'mentioned_companies', 'sentiment', 'neg', 'neu', 'pos', 'compound']
    feature_cols = [col for col in df.columns if col not in non_feature_cols and col != target_col]
    
    X = df[feature_cols]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logger.info(f"Data prepared. Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_sentiment_model(X_train, y_train):
    """
    Train a sentiment classification model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        object: Trained model
    """
    logger.info("Training sentiment classification model")
    
    # Start MLflow run
    with mlflow.start_run(run_name="sentiment_classification"):
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Define model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        # Perform grid search
        logger.info("Performing grid search for hyperparameter tuning")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "sentiment_model")
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return best_model

def train_impact_model(X_train, y_train):
    """
    Train an impact regression model
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        object: Trained model
    """
    logger.info("Training impact regression model")
    
    # Start MLflow run
    with mlflow.start_run(run_name="impact_regression"):
        # Log parameters
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        
        # Define model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        # Perform grid search
        logger.info("Performing grid search for hyperparameter tuning")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "impact_model")
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {-grid_search.best_score_:.4f}")
        
        return best_model

def evaluate_sentiment_model(model, X_test, y_test):
    """
    Evaluate sentiment classification model
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    """
    logger.info("Evaluating sentiment classification model")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log metrics with MLflow
    with mlflow.start_run(run_name="sentiment_evaluation"):
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("precision_weighted", report['weighted avg']['precision'])
        mlflow.log_metric("recall_weighted", report['weighted avg']['recall'])
        mlflow.log_metric("f1_weighted", report['weighted avg']['f1-score'])
    
    # Print report
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

def evaluate_impact_model(model, X_test, y_test):
    """
    Evaluate impact regression model
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    """
    logger.info("Evaluating impact regression model")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics with MLflow
    with mlflow.start_run(run_name="impact_evaluation"):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
    
    # Print metrics
    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info(f"Root Mean Squared Error: {rmse:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"R² Score: {r2:.4f}")

def save_model(model, model_path):
    """
    Save model to disk
    
    Args:
        model (object): Model to save
        model_path (str): Path to save the model
    """
    try:
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def create_impact_score(df, sentiment_model, feature_cols):
    """
    Create impact score based on sentiment probabilities and other features
    
    Args:
        df (pd.DataFrame): DataFrame containing features
        sentiment_model (object): Trained sentiment model
        feature_cols (list): List of feature columns
        
    Returns:
        pd.Series: Impact scores
    """
    logger.info("Creating impact scores")
    
    # Get sentiment probabilities
    sentiment_probs = sentiment_model.predict_proba(df[feature_cols])
    
    # Calculate impact score
    # Higher probability of positive sentiment and lower probability of negative sentiment
    # increases the impact score
    impact_score = sentiment_probs[:, 2] - sentiment_probs[:, 0]
    
    # Scale to 0-100 range
    impact_score = (impact_score + 1) * 50
    
    logger.info("Impact scores created")
    
    return impact_score

def run_model_training(features_path, sentiment_model_path, impact_model_path):
    """
    Run the complete model training pipeline
    
    Args:
        features_path (str): Path to the features CSV file
        sentiment_model_path (str): Path to save the sentiment model
        impact_model_path (str): Path to save the impact model
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("financial_news_analysis")
    
    logger.info("Starting model training pipeline")
    
    # Load features
    df = load_features(features_path)
    
    # Prepare data for sentiment classification
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, target_col='sentiment')
    
    # Train sentiment model
    sentiment_model = train_sentiment_model(X_train, y_train)
    
    # Evaluate sentiment model
    evaluate_sentiment_model(sentiment_model, X_test, y_test)
    
    # Save sentiment model
    save_model(sentiment_model, sentiment_model_path)
    
    # Create impact score
    df['impact_score'] = create_impact_score(df, sentiment_model, feature_cols)
    
    # Prepare data for impact regression
    X_train, X_test, y_train, y_test, _ = prepare_data(df, target_col='impact_score')
    
    # Train impact model
    impact_model = train_impact_model(X_train, y_train)
    
    # Evaluate impact model
    evaluate_impact_model(impact_model, X_test, y_test)
    
    # Save impact model
    save_model(impact_model, impact_model_path)
    
    logger.info("Model training pipeline completed")
    
    return sentiment_model, impact_model

if __name__ == "__main__":
    # Example usage
    features_path = "data/interim/features.csv"
    sentiment_model_path = "models/sentiment_model.joblib"
    impact_model_path = "models/impact_model.joblib"
    
    run_model_training(features_path, sentiment_model_path, impact_model_path) 