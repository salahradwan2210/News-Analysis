import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Financial News Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Define paths
PREDICTIONS_PATH = "data/processed/predictions.csv"
RANKINGS_PATH = "data/processed/stock_rankings.csv"
RECOMMENDATIONS_PATH = "reports/investment_recommendations.csv"

def load_data():
    """Load data from files if they exist"""
    data = {}
    
    try:
        if os.path.exists(PREDICTIONS_PATH):
            data['predictions'] = pd.read_csv(PREDICTIONS_PATH)
        else:
            data['predictions'] = None
            
        if os.path.exists(RANKINGS_PATH):
            data['rankings'] = pd.read_csv(RANKINGS_PATH)
        else:
            data['rankings'] = None
            
        if os.path.exists(RECOMMENDATIONS_PATH):
            data['recommendations'] = pd.read_csv(RECOMMENDATIONS_PATH)
        else:
            data['recommendations'] = None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        
    return data

def get_api_status():
    """Get API status"""
    try:
        response = requests.get(f"{API_URL}/status")
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return {"status": "error", "models_loaded": False, "training_status": "unknown"}

def train_models():
    """Train models via API"""
    try:
        response = requests.post(f"{API_URL}/train")
        return response.json()
    except Exception as e:
        st.error(f"Error training models: {e}")
        return {"message": f"Error: {str(e)}"}

def get_training_status():
    """Get training status from API"""
    try:
        response = requests.get(f"{API_URL}/training-status")
        return response.json()
    except Exception as e:
        st.error(f"Error getting training status: {e}")
        return {"status": "unknown", "message": str(e)}

def analyze_news(news_items):
    """Analyze news via API"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"news_items": news_items}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error analyzing news: {e}")
        return None

def get_stock_rankings(top_n=10):
    """Get stock rankings from API"""
    try:
        response = requests.get(f"{API_URL}/stocks?top_n={top_n}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error getting stock rankings: {e}")
        return []

def fetch_stock_chart(symbol, period='1mo'):
    """Fetch stock chart data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

def plot_stock_chart(symbol, data):
    """Plot stock price chart"""
    if data is None or data.empty:
        st.warning(f"No data available for {symbol}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data['Close'], label=f"{symbol} Close Price")
    
    # Add moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    ax.plot(data.index, data['MA5'], label='5-day MA', alpha=0.7)
    ax.plot(data.index, data['MA20'], label='20-day MA', alpha=0.7)
    
    ax.set_title(f"{symbol} Stock Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_sentiment_distribution(predictions):
    """Plot sentiment distribution"""
    if predictions is None or predictions.empty:
        st.warning("No prediction data available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Count sentiment values
    sentiment_counts = predictions['predicted_sentiment'].value_counts()
    
    # Create pie chart
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
           colors=['#ff9999','#66b3ff','#99ff99'], startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    ax.set_title("Sentiment Distribution")
    
    return fig

def plot_impact_distribution(predictions):
    """Plot impact score distribution"""
    if predictions is None or predictions.empty:
        st.warning("No prediction data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    sns.histplot(predictions['impact_score'], bins=20, kde=True, ax=ax)
    
    ax.set_title("Impact Score Distribution")
    ax.set_xlabel("Impact Score")
    ax.set_ylabel("Frequency")
    
    return fig

def plot_top_stocks(recommendations, n=10):
    """Plot top stocks by recommendation score"""
    if recommendations is None or recommendations.empty:
        st.warning("No recommendation data available")
        return
    
    # Get top N stocks
    top_stocks = recommendations.head(n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(top_stocks['symbol'], top_stocks['recommendation_score'], 
                  color=sns.color_palette("viridis", len(top_stocks)))
    
    # Add labels
    ax.set_title(f"Top {n} Stocks by Recommendation Score")
    ax.set_xlabel("Recommendation Score")
    ax.set_ylabel("Stock Symbol")
    
    # Add value labels
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f"{top_stocks['recommendation_score'].iloc[i]:.1f}", 
                va='center')
    
    return fig

def main():
    """Main function for the Streamlit app"""
    # Sidebar
    st.sidebar.title("Financial News Analysis")
    
    # Check API status
    api_status = get_api_status()
    
    # Display API status
    status_color = "green" if api_status.get("status") == "operational" else "red"
    st.sidebar.markdown(f"API Status: <span style='color:{status_color}'>{api_status.get('status', 'unknown')}</span>", unsafe_allow_html=True)
    
    models_loaded = api_status.get("models_loaded", False)
    models_status = "✅ Models loaded" if models_loaded else "❌ Models not loaded"
    st.sidebar.markdown(models_status)
    
    # Training status
    training_status = api_status.get("training_status", "unknown")
    training_message = api_status.get("training_message", "")
    
    if training_status == "running":
        st.sidebar.markdown("🔄 Training in progress...")
        st.sidebar.markdown(f"Status: {training_message}")
    elif training_status == "completed":
        st.sidebar.markdown("✅ Training completed")
    elif training_status == "failed":
        st.sidebar.markdown("❌ Training failed")
        st.sidebar.markdown(f"Error: {training_message}")
    
    # Load data
    data = load_data()
    
    # Sidebar options
    st.sidebar.header("Options")
    
    # Input for stock symbols
    default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    symbols_input = st.sidebar.text_area("Enter stock symbols (one per line)", 
                                        "\n".join(default_symbols))
    symbols = [s.strip() for s in symbols_input.split("\n") if s.strip()]
    
    # Train models button
    if st.sidebar.button("Train Models"):
        with st.sidebar:
            with st.spinner("Starting model training..."):
                result = train_models()
                st.success(result.get("message", "Training started"))
    
    # Refresh training status button
    if st.sidebar.button("Refresh Status"):
        st.experimental_rerun()
    
    # Main content
    st.title("Financial News Analysis Dashboard")
    
    # Overview section
    st.header("Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("News Articles Analyzed", 
                 len(data['predictions']) if data['predictions'] is not None else 0)
    
    with col2:
        st.metric("Stocks Ranked", 
                 len(data['rankings']) if data['rankings'] is not None else 0)
    
    with col3:
        if data['recommendations'] is not None and not data['recommendations'].empty:
            st.metric("Top Recommendation", 
                     data['recommendations']['symbol'].iloc[0],
                     data['recommendations']['recommendation'].iloc[0] if 'recommendation' in data['recommendations'].columns else "")
        else:
            st.metric("Top Recommendation", "N/A", "")
    
    # News Analysis section
    st.header("News Analysis")
    
    # News input
    with st.expander("Analyze New Financial News", expanded=not models_loaded):
        if not models_loaded:
            st.warning("Models are not loaded. Please train the models first.")
        
        news_text = st.text_area("Enter financial news to analyze", 
                                "Apple reported record quarterly revenue of $111.4 billion, up 21 percent year over year.")
        news_date = st.date_input("News date", datetime.now())
        
        if st.button("Analyze News"):
            if not models_loaded:
                st.error("Models are not loaded. Please train the models first.")
            else:
                with st.spinner("Analyzing news..."):
                    # Prepare news item
                    news_item = {
                        "date": news_date.strftime("%Y-%m-%d"),
                        "news": news_text,
                        "neg": 0.0,
                        "neu": 0.0,
                        "pos": 0.0,
                        "compound": 0.0
                    }
                    
                    # Call API to analyze news
                    result = analyze_news([news_item])
                    
                    if result:
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Display prediction
                        if result["predictions"]:
                            prediction = result["predictions"][0]
                            st.markdown(f"**Sentiment:** {prediction['sentiment']}")
                            st.markdown(f"**Impact Score:** {prediction['impact_score']:.2f}")
                            
                            if "mentioned_companies" in prediction and prediction["mentioned_companies"]:
                                st.markdown(f"**Mentioned Companies:** {', '.join(prediction['mentioned_companies'])}")
                        
                        # Display stock rankings
                        if result["stock_rankings"]:
                            st.subheader("Stock Rankings")
                            rankings_df = pd.DataFrame(result["stock_rankings"])
                            st.dataframe(rankings_df)
    
    # Sentiment Analysis section
    st.header("Sentiment Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if data['predictions'] is not None:
            fig = plot_sentiment_distribution(data['predictions'])
            if fig:
                st.pyplot(fig)
    
    with col2:
        if data['predictions'] is not None:
            fig = plot_impact_distribution(data['predictions'])
            if fig:
                st.pyplot(fig)
    
    # Top News section
    st.header("Top News by Impact")
    if data['predictions'] is not None and not data['predictions'].empty:
        # Sort by impact score
        top_news = data['predictions'].sort_values('impact_score', ascending=False).head(5)
        
        for i, (_, row) in enumerate(top_news.iterrows()):
            with st.expander(f"{i+1}. {row['news'][:100]}...", expanded=i==0):
                st.write(f"**Date:** {row['date']}")
                st.write(f"**Full News:** {row['news']}")
                st.write(f"**Sentiment:** {row['predicted_sentiment']}")
                st.write(f"**Impact Score:** {row['impact_score']:.2f}")
                
                if 'mentioned_companies' in row and row['mentioned_companies']:
                    st.write(f"**Mentioned Companies:** {row['mentioned_companies']}")
    else:
        st.info("No news data available. Run the analysis to see results.")
    
    # Stock Rankings section
    st.header("Stock Rankings")
    
    # Get stock rankings from API
    api_rankings = get_stock_rankings()
    
    if api_rankings:
        # Convert to DataFrame
        rankings_df = pd.DataFrame(api_rankings)
        
        # Plot top stocks
        fig = plot_top_stocks(rankings_df)
        if fig:
            st.pyplot(fig)
        
        # Display recommendations table
        st.subheader("Investment Recommendations")
        st.dataframe(rankings_df)
    elif data['recommendations'] is not None and not data['recommendations'].empty:
        # Use local data if API fails
        fig = plot_top_stocks(data['recommendations'])
        if fig:
            st.pyplot(fig)
        
        # Display recommendations table
        st.subheader("Investment Recommendations")
        st.dataframe(data['recommendations'])
    else:
        st.info("No stock rankings available. Run the analysis to see results.")
    
    # Stock Details section
    st.header("Stock Details")
    
    # Get available stocks
    available_stocks = []
    if api_rankings:
        available_stocks = [item["symbol"] for item in api_rankings]
    elif data['recommendations'] is not None and not data['recommendations'].empty:
        available_stocks = data['recommendations']['symbol'].tolist()
    
    if available_stocks:
        # Select stock
        selected_stock = st.selectbox("Select a stock to view details", available_stocks)
        
        if selected_stock:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Fetch and plot stock data
                stock_data = fetch_stock_chart(selected_stock)
                fig = plot_stock_chart(selected_stock, stock_data)
                if fig:
                    st.pyplot(fig)
            
            with col2:
                # Display stock details
                if api_rankings:
                    stock_info = next((item for item in api_rankings if item["symbol"] == selected_stock), None)
                elif data['recommendations'] is not None and not data['recommendations'].empty:
                    stock_info = data['recommendations'][data['recommendations']['symbol'] == selected_stock].iloc[0]
                else:
                    stock_info = None
                
                if stock_info:
                    st.subheader(f"{selected_stock} Details")
                    
                    if isinstance(stock_info, dict):
                        st.write(f"**Impact Score:** {stock_info['avg_impact_score']:.2f}")
                        st.write(f"**Sentiment:** {stock_info['most_common_sentiment']}")
                        st.write(f"**Mentions:** {stock_info['mention_count']}")
                        st.write(f"**Combined Score:** {stock_info['combined_score']:.2f}")
                    else:
                        st.write(f"**Impact Score:** {stock_info['avg_impact_score']:.2f}")
                        st.write(f"**Sentiment:** {stock_info['most_common_sentiment']}")
                        st.write(f"**Mentions:** {stock_info['mention_count']}")
                        
                        if 'recommendation' in stock_info:
                            st.write(f"**Recommendation:** {stock_info['recommendation']}")
                        
                        if 'current_price' in stock_info and pd.notna(stock_info['current_price']):
                            st.write(f"**Current Price:** ${stock_info['current_price']:.2f}")
                        
                        if 'price_change_pct' in stock_info and pd.notna(stock_info['price_change_pct']):
                            st.write(f"**Price Change (30d):** {stock_info['price_change_pct']:.2f}%")
    else:
        st.info("No stock data available. Run the analysis to see results.")

if __name__ == "__main__":
    main() 