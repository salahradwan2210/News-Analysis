"""
Main pipeline for financial news analysis.

This script orchestrates the entire process of:
1. Data processing
2. Feature extraction
3. Model training
4. Model evaluation
5. API server startup
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import subprocess
import time
import socket
import psutil  # Add this import for better process management

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
EVALUATION_DIR = ROOT_DIR / "evaluation"
SAMPLE_DATA_PATH = RAW_DATA_DIR / "sample_news.csv"
RAW_DATA_PATH = RAW_DATA_DIR / "news.csv"

# Default training parameters
DEFAULT_EPOCHS = 30
DEFAULT_DATA_PERCENTAGE = 50.0

# Default ports
DEFAULT_API_PORT = 8002
DEFAULT_WEB_PORT = 8003

# Ensure directories exist
for dir_path in [PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR, EVALUATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return None

def kill_process_on_port(port):
    """Kill process using the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.connections()
                for conn in connections:
                    if conn.laddr.port == port:
                        logger.info(f"Terminating process {proc.pid} using port {port}")
                        proc.terminate()
                        proc.wait(timeout=5)
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {str(e)}")
    return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial News Analysis Pipeline")
    
    parser.add_argument("--use-sample", action="store_true", help="Use sample data instead of full dataset")
    parser.add_argument("--data-path", type=str, help="Path to custom data file")
    parser.add_argument("--data-percentage", type=float, default=DEFAULT_DATA_PERCENTAGE, help="Percentage of data to use (1-100)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs for model training")
    parser.add_argument("--skip-processing", action="store_true", help="Skip data processing step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature extraction step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--no-api", action="store_true", help="Don't run API server after pipeline")
    parser.add_argument("--no-web", action="store_true", help="Don't run web UI after pipeline")
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT, help="Port for API server")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT, help="Port for web UI")
    
    return parser.parse_args()

def process_data(args):
    """
    Process raw data.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        
    Returns:
        str: Path to processed data.
    """
    logger.info("Starting data processing")
    
    try:
        from src.data_processing.preprocess import process_data as preprocess_data
        
        # Determine data path
        data_path = None
        if args.data_path:
            data_path = args.data_path
        elif args.use_sample:
            if not os.path.exists(SAMPLE_DATA_PATH):
                logger.error(f"Sample data not found at {SAMPLE_DATA_PATH}")
                return
            data_path = SAMPLE_DATA_PATH
        else:
            if not os.path.exists(RAW_DATA_PATH):
                logger.error(f"Raw data not found at {RAW_DATA_PATH}")
                return
            data_path = RAW_DATA_PATH
        
        # Process data
        processed_path = preprocess_data(
            use_sample=args.use_sample, 
            data_path=data_path,
            data_percentage=args.data_percentage
        )
        logger.info(f"Data processing completed. Processed data saved to {processed_path}")
        
        return processed_path
    
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise

def extract_features(args):
    """
    Extract features from processed data.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        
    Returns:
        str: Path to features.
    """
    logger.info("Starting feature extraction")
    
    try:
        from src.features.feature_engineering import extract_all_features
        
        # Extract features
        features_path = extract_all_features()
        logger.info(f"Feature extraction completed. Features saved to {features_path}")
        
        return features_path
    
    except Exception as e:
        logger.error(f"Error in feature extraction: {str(e)}")
        raise

def train_models(args):
    """
    Train models.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        
    Returns:
        dict: Paths to trained models.
    """
    logger.info(f"Starting model training with {args.epochs} epochs and {args.data_percentage}% of data")
    
    try:
        # Import necessary functions
        from src.models.train import train_all_models
        
        # Train all models
        model_paths = train_all_models(
            use_sample=args.use_sample,
            sentiment_model_type=args.sentiment_model,
            impact_model_type=args.impact_model,
            data_percentage=args.data_percentage,
            num_epochs=args.epochs
        )
        
        logger.info(f"Model training completed. Models saved to {MODELS_DIR}")
        
        return model_paths
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def evaluate_models(args):
    """
    Evaluate models.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        
    Returns:
        dict: Evaluation metrics.
    """
    logger.info("Starting model evaluation")
    
    try:
        from src.models.evaluate import evaluate_all_models
        
        # Evaluate models
        metrics = evaluate_all_models(use_sample=args.use_sample)
        
        logger.info(f"Model evaluation completed. Evaluation results saved to {EVALUATION_DIR}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def run_api_server(args):
    """Run API server."""
    # Check if port is in use
    if is_port_in_use(args.api_port):
        logger.warning(f"Port {args.api_port} is already in use")
        # Try to kill the process using the port
        if kill_process_on_port(args.api_port):
            logger.info(f"Successfully freed port {args.api_port}")
            time.sleep(2)  # Wait for port to be released
        else:
            # Try alternative ports
            available_port = find_available_port(args.api_port + 1)
            if available_port:
                logger.info(f"Using alternative port {available_port} for API server")
                args.api_port = available_port
            else:
                raise RuntimeError(f"No available ports found near {args.api_port}")

    logger.info(f"Starting API server on port {args.api_port}")
    
    try:
        # Set environment variable for models
        os.environ["MODELS_AVAILABLE"] = "true"
        
        # Run API server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.api.api:app",
            "--host", "0.0.0.0",
            "--port", str(args.api_port)
        ]
        
        api_process = subprocess.Popen(cmd)
        
        # Wait for API to start
        time.sleep(3)
        
        # Check if process is still running
        if api_process.poll() is not None:
            raise RuntimeError(f"API server failed to start on port {args.api_port}")
        
        logger.info(f"API server running at http://localhost:{args.api_port}")
        return api_process
    
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        raise

def run_web_ui(args):
    """Run web UI."""
    # Check if port is in use
    if is_port_in_use(args.web_port):
        logger.warning(f"Port {args.web_port} is already in use")
        # Try to kill the process using the port
        if kill_process_on_port(args.web_port):
            logger.info(f"Successfully freed port {args.web_port}")
            time.sleep(2)  # Wait for port to be released
        else:
            # Try alternative ports
            available_port = find_available_port(args.web_port + 1)
            if available_port:
                logger.info(f"Using alternative port {available_port} for Web UI")
                args.web_port = available_port
            else:
                raise RuntimeError(f"No available ports found near {args.web_port}")

    logger.info(f"Starting web UI on port {args.web_port}")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env["PORT"] = str(args.web_port)
        env["API_URL"] = f"http://localhost:{args.api_port}"
        
        # Run web server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "app:app",
            "--host", "0.0.0.0",
            "--port", str(args.web_port)
        ]
        
        web_process = subprocess.Popen(cmd, env=env)
        
        # Wait for web UI to start
        time.sleep(3)
        
        # Check if process is still running
        if web_process.poll() is not None:
            raise RuntimeError(f"Web UI failed to start on port {args.web_port}")
        
        logger.info(f"Web UI running at http://localhost:{args.web_port}")
        return web_process
    
    except Exception as e:
        logger.error(f"Error starting web UI: {str(e)}")
        raise

def main():
    """Main function to run the pipeline."""
    start_time = time.time()
    logger.info("Starting financial news analysis pipeline")
    
    # Parse arguments
    args = parse_args()
    
    # Validate data percentage
    if args.data_percentage <= 0 or args.data_percentage > 100:
        logger.error("Data percentage must be between 1 and 100")
        args.data_percentage = DEFAULT_DATA_PERCENTAGE
        logger.info(f"Using default data percentage: {args.data_percentage}%")
    
    # Validate epochs
    if args.epochs <= 0:
        logger.error("Number of epochs must be positive")
        args.epochs = DEFAULT_EPOCHS
        logger.info(f"Using default number of epochs: {args.epochs}")
    
    api_process = None
    web_process = None
    
    try:
        # Run API server
        if not args.no_api:
            api_process = run_api_server(args)
        
        # Run web UI
        if not args.no_web:
            web_process = run_web_ui(args)
        
        # Keep the script running if API or web UI is running
        if api_process or web_process:
            try:
                logger.info("Press Ctrl+C to stop the server(s)")
                
                while True:
                    # Check API server
                    if api_process and api_process.poll() is not None:
                        logger.warning("API server stopped unexpectedly. Restarting...")
                        api_process.terminate()
                        api_process.wait(timeout=5)
                        time.sleep(2)
                        api_process = run_api_server(args)
                    
                    # Check web UI
                    if web_process and web_process.poll() is not None:
                        logger.warning("Web UI stopped unexpectedly. Restarting...")
                        web_process.terminate()
                        web_process.wait(timeout=5)
                        time.sleep(2)
                        web_process = run_web_ui(args)
                    
                    time.sleep(5)
            
            except KeyboardInterrupt:
                logger.info("Stopping server(s)")
            
            finally:
                # Terminate processes
                if api_process:
                    logger.info("Terminating API server...")
                    api_process.terminate()
                    try:
                        api_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        api_process.kill()
                
                if web_process:
                    logger.info("Terminating Web UI...")
                    web_process.terminate()
                    try:
                        web_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        web_process.kill()
        
        # Calculate total runtime
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        
        # Ensure processes are terminated
        if api_process:
            api_process.terminate()
        if web_process:
            web_process.terminate()
        
        sys.exit(1)

if __name__ == "__main__":
    main() 