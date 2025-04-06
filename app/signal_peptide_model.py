import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Concatenate, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

def encode_sequences(sequences, max_length=100):
    """
    One-hot encode protein sequences
    
    Args:
        sequences: List of protein sequences
        max_length: Maximum sequence length (default: 100)
        
    Returns:
        encoded_seqs: One-hot encoded sequences
    """
    # Define the amino acid vocabulary (20 standard amino acids)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    # Initialize encoded sequences
    n_seqs = len(sequences)
    n_features = len(amino_acids)
    encoded_seqs = np.zeros((n_seqs, max_length, n_features))
    
    # Encode each sequence
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_length]):
            if aa in aa_to_idx:
                encoded_seqs[i, j, aa_to_idx[aa]] = 1
    
    return encoded_seqs

def extract_features(sequences, max_length=100):
    """
    Extract features from protein sequences including hydrophobicity
    
    Args:
        sequences: List of protein sequences
        max_length: Maximum sequence length (default: 100)
        
    Returns:
        features: Features matrix for sequences
    """
    # Kyte-Doolittle hydrophobicity scale
    hydrophobicity_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 
        'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 
        'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    features = np.zeros((len(sequences), max_length, 2))
    
    for i, seq in enumerate(sequences):
        # Truncate or pad sequence to max_length
        seq = seq[:max_length].ljust(max_length, 'X')
        
        for j, aa in enumerate(seq):
            # Add hydrophobicity as a feature
            features[i, j, 0] = hydrophobicity_scale.get(aa, 0)
            # Add position as a feature
            features[i, j, 1] = j / max_length  # Normalized position
    
    return features

def process_fasta_data(file_path):
    """
    Process a FASTA file and extract sequences, labels, annotations, and cleavage sites
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        sequences: List of sequences
        labels: Binary labels (0: no SP, 1: SP)
        annotations: List of sequence annotations (S/O marking)
        cleavage_sites: List of cleavage site positions
    """
    sequences = []
    labels = []
    annotations = []
    cleavage_sites = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                header = lines[i].strip()
                parts = header.split('|')
                
                if len(parts) >= 3:
                    sp_status = parts[2]
                    label = 0 if sp_status == 'NO_SP' else 1
                    
                    # Get sequence
                    i += 1
                    if i < len(lines) and not lines[i].startswith('>'):
                        seq = lines[i].strip()
                        sequences.append(seq)
                        labels.append(label)
                        
                        # Check for annotation line
                        i += 1
                        if i < len(lines) and not lines[i].startswith('>') and all(c in 'SOI' for c in lines[i].strip()):
                            annotation = lines[i].strip()
                            annotations.append(annotation)
                            
                            # Find cleavage site
                            cleavage_site = -1
                            if 'S' in annotation and 'O' in annotation:
                                for j in range(len(annotation) - 1):
                                    if annotation[j] == 'S' and annotation[j+1] == 'O':
                                        cleavage_site = j + 1
                                        break
                            cleavage_sites.append(cleavage_site)
                            
                            i += 1
                        else:
                            annotations.append(None)
                            cleavage_sites.append(-1)
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
    
    return sequences, labels, annotations, cleavage_sites

def create_classification_model(max_length=100, n_features=20):
    """
    Create a CNN-LSTM model for signal peptide classification
    
    Args:
        max_length: Maximum sequence length
        n_features: Number of features (amino acids)
        
    Returns:
        model: Compiled classification model
    """
    # Input layer
    input_layer = Input(shape=(max_length, n_features))
    
    # Convolutional layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)
    
    # Fully connected layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    
    # Output layer
    output_layer = Dense(1, activation='sigmoid', name='classification_output')(x)
    
    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cleavage_site_model(max_length=100, n_features=20):
    """
    Create a model for cleavage site prediction
    
    Args:
        max_length: Maximum sequence length
        n_features: Number of features (amino acids)
        
    Returns:
        model: Compiled cleavage site prediction model
    """
    # Input layer
    input_layer = Input(shape=(max_length, n_features))
    
    # Convolutional layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Bidirectional LSTM layers for sequence position understanding
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer - predicting position (regression)
    output_layer = Dense(1, activation='linear', name='cleavage_site_output')(x)
    
    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),  # Use full reference instead of string
        metrics=[MeanAbsoluteError()]  # Use object instead of string
    )
    
    return model

def train_classification_model(X_train, y_train, max_length=100, batch_size=32, epochs=50, model_path='models'):
    """
    Train the signal peptide classification model
    
    Args:
        X_train: Training sequences (encoded)
        y_train: Training labels
        max_length: Maximum sequence length
        batch_size: Batch size for training
        epochs: Number of training epochs
        model_path: Path to save the trained model
        
    Returns:
        model: Trained classification model
        history: Training history
    """
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Create model
    model = create_classification_model(max_length, X_train.shape[2])
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_path, 'classification_model_best.keras'),  # Change from .h5 to .keras
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Split data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train the model
    history = model.fit(
        X_train_split,
        y_train_split,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save final model - use .keras format instead of .h5
    model.save(os.path.join(model_path, 'classification_model.keras'))
    
    # Save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_path, 'classification_history.csv'), index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'classification_training_history.png'))
    plt.close()
    
    return model, history

def train_cleavage_site_model(X_train, y_train, max_length=100, batch_size=32, epochs=50, model_path='models'):
    """
    Train the cleavage site prediction model
    
    Args:
        X_train: Training sequences (encoded)
        y_train: Training cleavage site positions
        max_length: Maximum sequence length
        batch_size: Batch size for training
        epochs: Number of training epochs
        model_path: Path to save the trained model
        
    Returns:
        model: Trained cleavage site model
        history: Training history
    """
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Create model
    model = create_cleavage_site_model(max_length, X_train.shape[2])
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(model_path, 'cleavage_site_model_best.keras'),
        monitor='val_mean_absolute_error',  # Use the full metric name
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_mean_absolute_error',  # Use the full metric name
        patience=5,  # Reduced patience to stop earlier when validation doesn't improve
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,  # Reduced patience for faster learning rate reduction
        min_lr=1e-6,
        verbose=1
    )
    
    # Split data into training and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train the model
    history = model.fit(
        X_train_split,
        y_train_split,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_path, 'cleavage_site_model.keras'))
    
    # Save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(model_path, 'cleavage_site_history.csv'), index=False)
    
    # Plot training history - use correct metric names
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Use the full metric name
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'cleavage_site_training_history.png'))
    plt.close()
    
    return model, history

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate the signal peptide classification model
    
    Args:
        model: Trained classification model
        X_test: Test sequences (encoded)
        y_test: Test labels
        
    Returns:
        results: Dictionary containing evaluation metrics
    """
    # Predict probabilities
    y_pred_prob = model.predict(X_test)
    
    # Convert to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    
    return results

def evaluate_cleavage_site_model(model, X_test, y_test, tolerance=5):
    """
    Evaluate the cleavage site prediction model
    
    Args:
        model: Trained cleavage site model
        X_test: Test sequences (encoded)
        y_test: Test cleavage site positions
        tolerance: Number of positions allowed for correct prediction
        
    Returns:
        results: Dictionary containing evaluation metrics
    """
    # Predict cleavage sites
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate accuracy within tolerance
    correct = np.sum(np.abs(y_pred - y_test) <= tolerance)
    accuracy_with_tolerance = correct / len(y_test)
    
    # Calculate absolute errors for each prediction
    abs_errors = np.abs(y_pred - y_test)
    
    # Create results dictionary
    results = {
        'mae': mae,
        f'accuracy_within_{tolerance}_positions': accuracy_with_tolerance,
        'absolute_errors': abs_errors
    }
    
    return results

def predict_signal_peptide(sequences, classification_model_path, cleavage_site_model_path, max_length=100):
    """
    Predict signal peptide presence and cleavage site for new sequences
    
    Args:
        sequences: List of protein sequences
        classification_model_path: Path to the trained classification model
        cleavage_site_model_path: Path to the trained cleavage site model
        max_length: Maximum sequence length
        
    Returns:
        predictions: DataFrame containing predictions
    """
    # Load models with custom objects to handle the mse/mae issue
    custom_objects = {
        'MeanSquaredError': MeanSquaredError,
        'MeanAbsoluteError': MeanAbsoluteError
    }
    
    # Load models
    classification_model = load_model(classification_model_path, custom_objects=custom_objects)
    cleavage_site_model = load_model(cleavage_site_model_path, custom_objects=custom_objects)
    
    # Encode sequences
    X = encode_sequences(sequences, max_length)
    
    # Make predictions
    sp_probs = classification_model.predict(X).flatten()
    sp_preds = (sp_probs > 0.5).astype(int)
    
    # Only predict cleavage sites for sequences predicted to have signal peptides
    cleavage_sites = np.zeros(len(sequences))
    for i, has_sp in enumerate(sp_preds):
        if has_sp:
            cleavage_sites[i] = cleavage_site_model.predict(X[i:i+1])[0][0]
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'sequence': sequences,
        'has_signal_peptide': sp_preds,
        'signal_peptide_probability': sp_probs,
        'cleavage_site': cleavage_sites
    })
    
    return predictions

def visualize_cleavage_sites(sequences, cleavage_sites, seq_ids=None, output_file=None):
    """
    Create a visualization of predicted cleavage sites
    
    Args:
        sequences: List of protein sequences
        cleavage_sites: List of cleavage site positions
        seq_ids: List of sequence IDs
        output_file: Path to save the visualization
    """
    if seq_ids is None:
        seq_ids = [f"Sequence {i+1}" for i in range(len(sequences))]
    
    # Get only sequences with positive cleavage sites
    valid_indices = [i for i, site in enumerate(cleavage_sites) if site > 0]
    if not valid_indices:
        print("No valid cleavage sites to visualize")
        return
    
    valid_sequences = [sequences[i] for i in valid_indices]
    valid_cleavage = [cleavage_sites[i] for i in valid_indices]
    valid_ids = [seq_ids[i] for i in valid_indices]
    
    # Create figure
    num_seqs = min(len(valid_sequences), 10)  # Show at most 10 sequences
    plt.figure(figsize=(14, num_seqs * 0.8))
    
    for i in range(num_seqs):
        seq = valid_sequences[i]
        site = valid_cleavage[i]
        seq_id = valid_ids[i]
        
        # Trim long sequences for visualization
        display_seq = seq[:min(50, len(seq))]
        
        # Create colormap: signal peptide in red, mature protein in green
        colors = ['#ffcccc'] * site + ['#ccffcc'] * (len(display_seq) - site)
        
        # Plot sequence as colored boxes
        plt.subplot(num_seqs, 1, i + 1)
        for j, (aa, color) in enumerate(zip(display_seq, colors)):
            plt.text(j + 0.5, 0.5, aa, ha='center', va='center', fontsize=10)
            plt.gca().add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=color, edgecolor='gray'))
        
        # Add vertical line at cleavage site
        if site < len(display_seq):
            plt.axvline(x=site, color='black', linestyle='-', linewidth=2)
        
        # Add sequence ID and cleavage position
        plt.title(f"{seq_id}: Cleavage site at position {site}", fontsize=10)
        plt.axis('off')
        plt.xlim(0, len(display_seq))
        plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Default paths
    train_fasta = 'data/signal_peptide_train.fasta'
    model_dir = 'model'
    
    # Train model
    print("Training signal peptide and cleavage site prediction model...")
    model, history = train_model(train_fasta, model_dir)
    
    print("\nModel training complete!")