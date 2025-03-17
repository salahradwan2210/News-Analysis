"""
API module for financial news analysis.
"""
import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
import torch
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

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
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_lstm.pt"
IMPACT_MODEL_PATH = MODELS_DIR / "impact_gru.pt"
RANKING_MODEL_PATH = MODELS_DIR / "stock_ranking.json"

# Initialize FastAPI app
app = FastAPI(
    title="Financial News Analysis API",
    description="API for analyzing financial news and predicting sentiment and impact on stocks.",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define models
class NewsItem(BaseModel):
    """
    News item model.
    """
    title: str
    content: str
    date: Optional[str] = None
    source: Optional[str] = None
    symbol: Optional[str] = None

class NewsAnalysis(BaseModel):
    """
    News analysis model.
    """
    sentiment: str
    sentiment_probability: float
    impact: str
    impact_probability: float
    extracted_symbols: List[str]

class StockRanking(BaseModel):
    """
    Stock ranking model.
    """
    symbol: str
    ranking_score: float
    impact_score_mean: float
    impact_score_max: float
    overall_sentiment_mean: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    news_count: int

class RankingResponse(BaseModel):
    """
    Ranking response model.
    """
    timestamp: str
    ranking_formula: Dict[str, float] = {
        "sentiment_weight": 0.6,
        "impact_weight": 0.4
    }
    ranked_stocks: List[StockRanking]

class StatusResponse(BaseModel):
    """
    Status response model.
    """
    status: str

# Dummy model classes for when real models are not available
class DummySentimentModel:
    def __init__(self):
        self.classes_ = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
    def eval(self):
        return self
        
    def to(self, device):
        return self
        
    def __call__(self, x):
        # Return random predictions
        batch_size = x.shape[0]
        return torch.rand(batch_size, 3)

class DummyImpactModel:
    def __init__(self):
        self.classes_ = ["very low", "low", "medium", "high", "very high"]
        
    def eval(self):
        return self
        
    def to(self, device):
        return self
        
    def __call__(self, x):
        # Return random predictions
        batch_size = x.shape[0]
        return torch.rand(batch_size, 5)

class DummyTokenizer:
    def __init__(self):
        pass
        
    def encode_plus(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        # Return dummy input_ids and attention_mask
        return {
            "input_ids": torch.ones((1, 100), dtype=torch.long),
            "attention_mask": torch.ones((1, 100), dtype=torch.long)
        }

class DummyLabelEncoder:
    def __init__(self, classes=None):
        self.classes_ = classes or ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
    def inverse_transform(self, y):
        # For dummy predictions, return a mix of classes
        if isinstance(y, np.ndarray) and len(y) > 0:
            return np.random.choice(self.classes_, size=len(y))
        else:
            return np.random.choice(self.classes_)

# Global variables for models
sentiment_model = None
impact_model = None
sentiment_tokenizer = None
impact_tokenizer = None
sentiment_label_encoder = None
impact_label_encoder = None
ranking_data = None
models_available = False

def load_models():
    """
    Load trained models.
    
    Returns:
        bool: True if models were loaded successfully, False otherwise.
    """
    global sentiment_model, impact_model, sentiment_tokenizer, impact_tokenizer
    global sentiment_label_encoder, impact_label_encoder, ranking_data, models_available
    
    # Check if models should be loaded
    models_available_env = os.environ.get("MODELS_AVAILABLE", "true").lower()
    if models_available_env == "false":
        logger.warning("Models are disabled by environment variable. Using dummy models.")
        
        # Initialize dummy models
        sentiment_model = DummySentimentModel()
        impact_model = DummyImpactModel()
        sentiment_tokenizer = DummyTokenizer()
        impact_tokenizer = DummyTokenizer()
        sentiment_label_encoder = DummyLabelEncoder(["NEGATIVE", "NEUTRAL", "POSITIVE"])
        impact_label_encoder = DummyLabelEncoder(["very low", "low", "medium", "high", "very high"])
        
        # Create dummy ranking data
        ranking_data = {
            "timestamp": datetime.now().isoformat(),
            "ranking_formula": {
                "sentiment_weight": 0.6,
                "impact_weight": 0.4
            },
            "ranked_stocks": []
        }
        
        models_available = False
        return False
    
    try:
        # Check if model files exist
        if not os.path.exists(SENTIMENT_MODEL_PATH):
            logger.warning(f"Sentiment model not found at {SENTIMENT_MODEL_PATH}. Using dummy model.")
            sentiment_model = DummySentimentModel()
            sentiment_tokenizer = DummyTokenizer()
            sentiment_label_encoder = DummyLabelEncoder(["NEGATIVE", "NEUTRAL", "POSITIVE"])
        else:
            # Load sentiment model
            logger.info(f"Loading sentiment model from {SENTIMENT_MODEL_PATH}")
            from src.models.deep_learning import load_trained_model
            sentiment_model, sentiment_tokenizer, sentiment_label_encoder = load_trained_model("sentiment_lstm")
        
        if not os.path.exists(IMPACT_MODEL_PATH):
            logger.warning(f"Impact model not found at {IMPACT_MODEL_PATH}. Using dummy model.")
            impact_model = DummyImpactModel()
            impact_tokenizer = DummyTokenizer()
            impact_label_encoder = DummyLabelEncoder(["very low", "low", "medium", "high", "very high"])
        else:
            # Load impact model
            logger.info(f"Loading impact model from {IMPACT_MODEL_PATH}")
            from src.models.deep_learning import load_trained_model
            impact_model, impact_tokenizer, impact_label_encoder = load_trained_model("impact_gru")
        
        if not os.path.exists(RANKING_MODEL_PATH):
            logger.warning(f"Ranking model not found at {RANKING_MODEL_PATH}. Using dummy ranking data.")
            # Create dummy ranking data
            ranking_data = {
                "timestamp": datetime.now().isoformat(),
                "ranking_formula": {
                    "sentiment_weight": 0.6,
                    "impact_weight": 0.4
                },
                "ranked_stocks": []
            }
        else:
            # Load ranking data
            logger.info(f"Loading ranking data from {RANKING_MODEL_PATH}")
            with open(RANKING_MODEL_PATH, 'r') as f:
                ranking_data = json.load(f)
        
        # Set models_available flag based on whether all models were loaded
        models_available = (
            sentiment_model is not None and 
            impact_model is not None and 
            ranking_data is not None
        )
        
        return models_available
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        
        # Initialize dummy models
        sentiment_model = DummySentimentModel()
        impact_model = DummyImpactModel()
        sentiment_tokenizer = DummyTokenizer()
        impact_tokenizer = DummyTokenizer()
        sentiment_label_encoder = DummyLabelEncoder(["NEGATIVE", "NEUTRAL", "POSITIVE"])
        impact_label_encoder = DummyLabelEncoder(["very low", "low", "medium", "high", "very high"])
        
        # Create dummy ranking data
        ranking_data = {
            "timestamp": datetime.now().isoformat(),
            "ranking_formula": {
                "sentiment_weight": 0.6,
                "impact_weight": 0.4
            },
            "ranked_stocks": []
        }
        
        models_available = False
        return False

def extract_symbols(text):
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

def predict_sentiment(text):
    """
    Predict sentiment for a text.
    
    Args:
        text (str): Text to predict sentiment for.
        
    Returns:
        tuple: (sentiment, probability)
    """
    global sentiment_model, sentiment_tokenizer, sentiment_label_encoder, models_available
    
    if not models_available:
        # Return random sentiment for dummy model
        sentiments = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        sentiment = np.random.choice(sentiments)
        probability = np.random.uniform(0.6, 0.9)
        return sentiment, probability
    
    try:
        # Prepare text data
        from src.models.deep_learning import prepare_text_data
        
        # Set model to evaluation mode
        sentiment_model.eval()
        
        # Tokenize text
        inputs = sentiment_tokenizer.encode_plus(
            text,
            max_length=100,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Make prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentiment_model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sentiment_model(inputs["input_ids"])
            if len(sentiment_label_encoder.classes_) > 2:
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                probability = probabilities[predicted_class]
            else:
                probability = torch.sigmoid(outputs.view(-1)).cpu().numpy()[0]
                predicted_class = int(probability > 0.5)
                probability = probability if predicted_class == 1 else 1 - probability
        
        # Convert prediction to label
        sentiment = sentiment_label_encoder.inverse_transform([predicted_class])[0]
        
        return sentiment, float(probability)
    
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        
        # Return default sentiment in case of error
        return "NEUTRAL", 0.5

def predict_impact(text):
    """
    Predict impact for a text.
    
    Args:
        text (str): Text to predict impact for.
        
    Returns:
        tuple: (impact, probability)
    """
    global impact_model, impact_tokenizer, impact_label_encoder, models_available
    
    if not models_available:
        # Return random impact for dummy model
        impacts = ["very low", "low", "medium", "high", "very high"]
        impact = np.random.choice(impacts)
        probability = np.random.uniform(0.6, 0.9)
        return impact, probability
    
    try:
        # Set model to evaluation mode
        impact_model.eval()
        
        # Tokenize text
        inputs = impact_tokenizer.encode_plus(
            text,
            max_length=100,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Make prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        impact_model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = impact_model(inputs["input_ids"])
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            probability = probabilities[predicted_class]
        
        # Convert prediction to label
        impact = impact_label_encoder.inverse_transform([predicted_class])[0]
        
        return impact, float(probability)
    
    except Exception as e:
        logger.error(f"Error predicting impact: {str(e)}")
        
        # Return default impact in case of error
        return "medium", 0.5

def analyze_news(news_item):
    """
    Analyze a news item.
    
    Args:
        news_item (NewsItem): News item to analyze.
        
    Returns:
        dict: Analysis results.
    """
    # Extract text from news item
    title = news_item.title
    content = news_item.content
    
    # Combine title and content for analysis
    text = f"{title} {content}"
    
    # Extract stock symbols
    extracted_symbols = extract_symbols(text)
    
    # Add symbol from news item if provided
    if news_item.symbol and news_item.symbol not in extracted_symbols:
        extracted_symbols.append(news_item.symbol)
    
    # Predict sentiment
    sentiment, sentiment_probability = predict_sentiment(text)
    
    # Predict impact
    impact, impact_probability = predict_impact(text)
    
    # Return analysis results
    return {
        "sentiment": sentiment,
        "sentiment_probability": sentiment_probability,
        "impact": impact,
        "impact_probability": impact_probability,
        "extracted_symbols": extracted_symbols
    }

# API endpoints
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    """
    logger.info("Starting API server")
    load_models()

@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        dict: API information.
    """
    return {
        "name": "Financial News Analysis API",
        "version": "1.0.0",
        "description": "API for analyzing financial news and predicting sentiment and impact on stocks.",
        "models_available": models_available,
        "endpoints": [
            "/analyze",
            "/batch-analyze",
            "/rankings",
            "/rankings/{symbol}",
            "/update-rankings"
        ]
    }

@app.post("/analyze", response_model=NewsAnalysis)
async def analyze(news_item: NewsItem):
    """
    Analyze a news item.
    
    Args:
        news_item (NewsItem): News item to analyze.
        
    Returns:
        NewsAnalysis: Analysis results.
    """
    logger.info(f"Analyzing news: {news_item.title}")
    
    # Analyze news
    analysis = analyze_news(news_item)
    
    return analysis

@app.post("/batch-analyze", response_model=List[NewsAnalysis])
async def batch_analyze(news_items: List[NewsItem]):
    """
    Analyze multiple news items.
    
    Args:
        news_items (List[NewsItem]): News items to analyze.
        
    Returns:
        List[NewsAnalysis]: Analysis results for each news item.
    """
    logger.info(f"Batch analyzing {len(news_items)} news items")
    
    # Analyze each news item
    results = []
    for news_item in news_items:
        analysis = analyze_news(news_item)
        results.append(analysis)
    
    return results

@app.get("/rankings", response_model=RankingResponse)
async def get_rankings(limit: int = 10):
    """
    Get stock rankings.
    
    Args:
        limit (int): Maximum number of stocks to return.
        
    Returns:
        RankingResponse: Stock rankings.
    """
    global ranking_data
    
    logger.info(f"Getting stock rankings (limit={limit})")
    
    if not ranking_data:
        # Return empty rankings if no ranking data is available
        return {
            "timestamp": datetime.now().isoformat(),
            "ranking_formula": {
                "sentiment_weight": 0.6,
                "impact_weight": 0.4
            },
            "ranked_stocks": []
        }
    
    # Limit the number of stocks
    ranked_stocks = ranking_data.get("ranked_stocks", [])[:limit]
    
    # Return rankings
    return {
        "timestamp": ranking_data.get("timestamp", datetime.now().isoformat()),
        "ranking_formula": ranking_data.get("ranking_formula", {}),
        "ranked_stocks": ranked_stocks
    }

@app.get("/rankings/{symbol}", response_model=StockRanking)
async def get_ranking(symbol: str):
    """
    Get ranking for a specific stock.
    
    Args:
        symbol (str): Stock symbol.
        
    Returns:
        StockRanking: Stock ranking.
    """
    global ranking_data
    
    logger.info(f"Getting ranking for stock: {symbol}")
    
    if not ranking_data:
        # Return empty ranking if no ranking data is available
        return {
            "symbol": symbol,
            "ranking_score": 0.0,
            "impact_score_mean": 0.0,
            "impact_score_max": 0.0,
            "overall_sentiment_mean": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 0.0,
            "news_count": 0
        }
    
    # Find the stock in the rankings
    ranked_stocks = ranking_data.get("ranked_stocks", [])
    for stock in ranked_stocks:
        if stock.get("symbol") == symbol:
            return stock
    
    # Return 404 if stock not found
    raise HTTPException(status_code=404, detail=f"Stock {symbol} not found in rankings")

@app.post("/update-rankings", response_model=StatusResponse)
async def update_rankings(force: bool = False):
    """
    Update stock rankings.
    
    Args:
        force (bool): Force update even if rankings exist.
        
    Returns:
        StatusResponse: Status message.
    """
    global ranking_data
    
    logger.info(f"Updating stock rankings (force={force})")
    
    if not force and ranking_data and ranking_data.get("ranked_stocks"):
        return {"status": "Rankings already exist. Use force=true to update."}
    
    try:
        # Load processed data
        if os.path.exists(FEATURES_PATH):
            df = pd.read_csv(FEATURES_PATH)
            
            # Generate rankings
            from src.models.train import generate_stock_ranking
            ranking_path = generate_stock_ranking(df)
            
            # Load updated rankings
            with open(ranking_path, 'r') as f:
                ranking_data = json.load(f)
            
            return {"status": "Rankings updated successfully"}
        else:
            return {"status": "Features file not found. Process data and extract features first."}
    
    except Exception as e:
        logger.error(f"Error updating rankings: {str(e)}")
        return {"status": f"Error updating rankings: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
