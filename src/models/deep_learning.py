"""
Deep learning models for financial news analysis using PyTorch.
"""
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
FEATURES_PATH = ROOT_DIR / "data" / "processed" / "features" / "all_features.csv"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
MAX_SEQUENCE_LENGTH = 200
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Tokenizer:
    """Simple tokenizer class to replace Keras tokenizer."""
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}
        self.document_count = 0
        
    def fit_on_texts(self, texts):
        """Build vocabulary from texts."""
        self.document_count = len(texts)
        for text in texts:
            for word in text.split():
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create word_index
        if self.num_words:
            sorted_words = sorted_words[:self.num_words-1]  # -1 for OOV token
            
        # Add OOV token at index 1
        self.word_index = {'<OOV>': 1}
        self.index_word = {1: '<OOV>'}
        
        # Add rest of words
        for i, (word, _) in enumerate(sorted_words):
            idx = i + 2  # +2 because 0 is padding, 1 is OOV
            self.word_index[word] = idx
            self.index_word[idx] = word
            
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of integers."""
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(1)  # OOV token
            sequences.append(sequence)
        return sequences

def prepare_text_data(texts, tokenizer=None, max_sequence_length=MAX_SEQUENCE_LENGTH):
    """
    Prepare text data for deep learning models.
    
    Args:
        texts (list): List of text strings.
        tokenizer (Tokenizer, optional): Tokenizer object. If None, create a new one.
        max_sequence_length (int): Maximum sequence length for padding.
        
    Returns:
        tuple: (sequences, tokenizer)
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        logger.info("Creating new tokenizer")
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(texts)
        
        # Save tokenizer
        logger.info(f"Saving tokenizer to {TOKENIZER_PATH}")
        joblib.dump(tokenizer, TOKENIZER_PATH)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        # Truncate if longer than max_sequence_length
        if len(seq) > max_sequence_length:
            seq = seq[:max_sequence_length]
        # Pad if shorter than max_sequence_length
        else:
            seq = seq + [0] * (max_sequence_length - len(seq))
        padded_sequences.append(seq)
    
    return np.array(padded_sequences), tokenizer

def prepare_labels(labels, label_encoder=None):
    """
    Prepare labels for deep learning models.
    
    Args:
        labels (list): List of label strings or integers.
        label_encoder (LabelEncoder, optional): Sklearn label encoder. If None, create a new one.
        
    Returns:
        tuple: (encoded_labels, label_encoder)
    """
    # Create label encoder if not provided
    if label_encoder is None:
        logger.info("Creating new label encoder")
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        
        # Save label encoder
        logger.info(f"Saving label encoder to {LABEL_ENCODER_PATH}")
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    # Encode labels
    encoded_labels = label_encoder.transform(labels)
    
    # Convert to one-hot encoding for multi-class
    num_classes = len(label_encoder.classes_)
    if num_classes > 2:
        # Convert to PyTorch tensor
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)
    else:
        # For binary classification, use a single output
        encoded_labels = torch.tensor(encoded_labels, dtype=torch.float32).unsqueeze(1)
    
    return encoded_labels, label_encoder

class TextDataset(Dataset):
    """PyTorch Dataset for text data."""
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LSTMModel(nn.Module):
    """LSTM model for text classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class GRUModel(nn.Module):
    """GRU model for text classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class CNNRNNModel(nn.Module):
    """Hybrid CNN-RNN model for text classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(CNNRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN layers
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2)
        
        # RNN layer
        self.lstm = nn.LSTM(hidden_dim * 3, hidden_dim, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        
        # CNN forward (need to transpose for Conv1d which expects [batch, channels, seq_len])
        embedded_permuted = embedded.permute(0, 2, 1)
        
        conv1_out = torch.relu(self.conv1(embedded_permuted))
        conv2_out = torch.relu(self.conv2(embedded_permuted))
        conv3_out = torch.relu(self.conv3(embedded_permuted))
        
        # Global max pooling
        conv1_out = torch.max_pool1d(conv1_out, conv1_out.shape[2]).squeeze(2)
        conv2_out = torch.max_pool1d(conv2_out, conv2_out.shape[2]).squeeze(2)
        conv3_out = torch.max_pool1d(conv3_out, conv3_out.shape[2]).squeeze(2)
        
        # Concatenate CNN outputs
        cnn_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        
        # Reshape for LSTM
        cnn_out = cnn_out.unsqueeze(1).repeat(1, embedded.size(1), 1)
        
        # LSTM forward
        lstm_out, (hidden, _) = self.lstm(cnn_out)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        return self.fc(hidden)

def build_lstm_model(max_sequence_length=MAX_SEQUENCE_LENGTH, max_num_words=MAX_NUM_WORDS, 
                    embedding_dim=EMBEDDING_DIM, num_classes=3):
    """
    Build an LSTM model for text classification.
    
    Args:
        max_sequence_length (int): Maximum sequence length.
        max_num_words (int): Maximum number of words in the vocabulary.
        embedding_dim (int): Dimensionality of the embedding layer.
        num_classes (int): Number of output classes.
        
    Returns:
        nn.Module: LSTM model.
    """
    logger.info("Building LSTM model")
    
    # Save model configuration
    model_config = {
        'model_type': 'lstm',
        'max_sequence_length': max_sequence_length,
        'max_num_words': max_num_words,
        'embedding_dim': embedding_dim,
        'hidden_dim': 128,
        'num_classes': num_classes,
        'dropout': 0.5
    }
    
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Build model
    model = LSTMModel(
        vocab_size=max_num_words,
        embedding_dim=embedding_dim,
        hidden_dim=128,
        num_classes=num_classes
    )
    
    return model

def build_gru_model(max_sequence_length=MAX_SEQUENCE_LENGTH, max_num_words=MAX_NUM_WORDS, 
                   embedding_dim=EMBEDDING_DIM, num_classes=3):
    """
    Build a GRU model for text classification.
    
    Args:
        max_sequence_length (int): Maximum sequence length.
        max_num_words (int): Maximum number of words in the vocabulary.
        embedding_dim (int): Dimensionality of the embedding layer.
        num_classes (int): Number of output classes.
        
    Returns:
        nn.Module: GRU model.
    """
    logger.info("Building GRU model")
    
    # Save model configuration
    model_config = {
        'model_type': 'gru',
        'max_sequence_length': max_sequence_length,
        'max_num_words': max_num_words,
        'embedding_dim': embedding_dim,
        'hidden_dim': 128,
        'num_classes': num_classes,
        'dropout': 0.5
    }
    
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Build model
    model = GRUModel(
        vocab_size=max_num_words,
        embedding_dim=embedding_dim,
        hidden_dim=128,
        num_classes=num_classes
    )
    
    return model

def build_cnn_rnn_model(max_sequence_length=MAX_SEQUENCE_LENGTH, max_num_words=MAX_NUM_WORDS, 
                       embedding_dim=EMBEDDING_DIM, num_classes=3):
    """
    Build a hybrid CNN-RNN model for text classification.
    
    Args:
        max_sequence_length (int): Maximum sequence length.
        max_num_words (int): Maximum number of words in the vocabulary.
        embedding_dim (int): Dimensionality of the embedding layer.
        num_classes (int): Number of output classes.
        
    Returns:
        nn.Module: CNN-RNN model.
    """
    logger.info("Building CNN-RNN model")
    
    # Save model configuration
    model_config = {
        'model_type': 'cnn_rnn',
        'max_sequence_length': max_sequence_length,
        'max_num_words': max_num_words,
        'embedding_dim': embedding_dim,
        'hidden_dim': 128,
        'num_classes': num_classes,
        'dropout': 0.5
    }
    
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Build model
    model = CNNRNNModel(
        vocab_size=max_num_words,
        embedding_dim=embedding_dim,
        hidden_dim=128,
        num_classes=num_classes
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, model_name, 
               batch_size=BATCH_SIZE, epochs=EPOCHS):
    """
    Train a deep learning model.
    
    Args:
        model (nn.Module): PyTorch model to train.
        X_train (numpy.ndarray): Training data.
        y_train (torch.Tensor): Training labels.
        X_val (numpy.ndarray): Validation data.
        y_val (torch.Tensor): Validation labels.
        model_name (str): Name of the model for saving.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        
    Returns:
        nn.Module: Trained model.
    """
    logger.info(f"Training {model_name} model")
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Define loss function and optimizer
    num_classes = y_train.max().item() + 1 if len(y_train.shape) == 1 else y_train.shape[1]
    if num_classes > 2:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = MODELS_DIR / f"{model_name}.pt"
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            if num_classes <= 2:
                outputs = outputs.view(-1)  # Flatten for binary classification
            
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                if num_classes <= 2:
                    outputs = outputs.view(-1)  # Flatten for binary classification
                    loss = criterion(outputs, labels)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate accuracy
                if num_classes <= 2:
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                else:
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
        # Calculate average losses and accuracy
        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps
        accuracy = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"Val Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    return model

def predict(model, texts, tokenizer=None, label_encoder=None):
    """
    Make predictions with a trained model.
    
    Args:
        model (nn.Module): Trained PyTorch model.
        texts (list): List of text strings.
        tokenizer (Tokenizer, optional): Tokenizer object. If None, load from file.
        label_encoder (LabelEncoder, optional): Sklearn label encoder. If None, load from file.
        
    Returns:
        tuple: (predictions, predicted_labels)
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        if os.path.exists(TOKENIZER_PATH):
            logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
            tokenizer = joblib.load(TOKENIZER_PATH)
        else:
            logger.error(f"Tokenizer not found at {TOKENIZER_PATH}")
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    
    # Prepare text data
    sequences, _ = prepare_text_data(texts, tokenizer)
    
    # Create dataset and dataloader
    dataset = torch.tensor(sequences, dtype=torch.long)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # Move model to device and set to evaluation mode
    model = model.to(DEVICE)
    model.eval()
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # Convert outputs to probabilities
            if outputs.shape[1] > 1:  # Multi-class
                probs = torch.softmax(outputs, dim=1)
            else:  # Binary
                probs = torch.sigmoid(outputs)
                
            predictions.append(probs.cpu().numpy())
    
    # Concatenate predictions
    predictions = np.vstack(predictions)
    
    # Convert predictions to labels if label encoder is available
    if label_encoder is None:
        if os.path.exists(LABEL_ENCODER_PATH):
            logger.info(f"Loading label encoder from {LABEL_ENCODER_PATH}")
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
        else:
            logger.warning(f"Label encoder not found at {LABEL_ENCODER_PATH}")
            return predictions, None
    
    # Convert predictions to class indices
    if predictions.shape[1] > 1:  # Multi-class
        pred_indices = np.argmax(predictions, axis=1)
    else:  # Binary
        pred_indices = (predictions > 0.5).astype(int).flatten()
    
    # Convert indices to labels
    predicted_labels = label_encoder.inverse_transform(pred_indices)
    
    return predictions, predicted_labels

def load_trained_model(model_name):
    """
    Load a trained model.
    
    Args:
        model_name (str): Name of the model to load.
        
    Returns:
        tuple: (model, tokenizer, label_encoder)
    """
    model_path = MODELS_DIR / f"{model_name}.pt"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load model configuration
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, 'r') as f:
            model_config = json.load(f)
    else:
        logger.warning(f"Model configuration not found at {MODEL_CONFIG_PATH}")
        model_config = {
            'model_type': model_name.split('_')[0],
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'max_num_words': MAX_NUM_WORDS,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': 128,
            'num_classes': 3,
            'dropout': 0.5
        }
    
    # Load tokenizer
    if os.path.exists(TOKENIZER_PATH):
        logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
        tokenizer = joblib.load(TOKENIZER_PATH)
    else:
        logger.error(f"Tokenizer not found at {TOKENIZER_PATH}")
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    
    # Load label encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        logger.info(f"Loading label encoder from {LABEL_ENCODER_PATH}")
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    else:
        logger.error(f"Label encoder not found at {LABEL_ENCODER_PATH}")
        raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")
    
    # Create model
    num_classes = len(label_encoder.classes_)
    
    if model_config['model_type'] == 'lstm':
        model = LSTMModel(
            vocab_size=model_config.get('max_num_words', MAX_NUM_WORDS),
            embedding_dim=model_config.get('embedding_dim', EMBEDDING_DIM),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_classes=num_classes
        )
    elif model_config['model_type'] == 'gru':
        model = GRUModel(
            vocab_size=model_config.get('max_num_words', MAX_NUM_WORDS),
            embedding_dim=model_config.get('embedding_dim', EMBEDDING_DIM),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_classes=num_classes
        )
    elif model_config['model_type'] == 'cnn_rnn':
        model = CNNRNNModel(
            vocab_size=model_config.get('max_num_words', MAX_NUM_WORDS),
            embedding_dim=model_config.get('embedding_dim', EMBEDDING_DIM),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_classes=num_classes
        )
    else:
        logger.error(f"Unknown model type: {model_config['model_type']}")
        raise ValueError(f"Unknown model type: {model_config['model_type']}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    return model, tokenizer, label_encoder

if __name__ == "__main__":
    try:
        # Load features
        if not os.path.exists(FEATURES_PATH):
            logger.error(f"Features file not found: {FEATURES_PATH}")
            raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
        
        logger.info(f"Loading features from {FEATURES_PATH}")
        df = pd.read_csv(FEATURES_PATH)
        
        # Prepare data
        text_col = 'processed_content' if 'processed_content' in df.columns else 'news'
        
        if 'sentiment' in df.columns:
            target_col = 'sentiment'
        elif 'sentiment_category' in df.columns:
            target_col = 'sentiment_category'
        else:
            logger.error("No sentiment column found in the dataframe.")
            raise ValueError("No sentiment column found in the dataframe.")
        
        # Split data (assuming you've already split the data elsewhere)
        # For demonstration, we'll use a small subset
        sample_size = min(10000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        
        # Prepare text data
        X_sequences, tokenizer = prepare_text_data(df_sample[text_col])
        
        # Prepare labels
        y_encoded, label_encoder = prepare_labels(df_sample[target_col])
        
        # Build and train models
        # LSTM model
        lstm_model = build_lstm_model(num_classes=len(label_encoder.classes_))
        
        # GRU model
        gru_model = build_gru_model(num_classes=len(label_encoder.classes_))
        
        # CNN-RNN hybrid model
        cnn_rnn_model = build_cnn_rnn_model(num_classes=len(label_encoder.classes_))
        
        logger.info("Models built successfully. Ready for training.")
        
    except Exception as e:
        logger.error(f"Error in deep learning module: {str(e)}")
        raise 