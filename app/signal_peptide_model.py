import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

# Function to read and process FASTA data with cleavage site annotation
def process_fasta_data(file_path):
    sequences = []
    labels = []
    annotations = []
    cleavage_sites = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                # Parse header
                header = lines[i].strip()
                parts = header.split('|')
                
                # Extract signal peptide status
                sp_status = parts[2]
                if sp_status == 'NO_SP':
                    label = 0  # No signal peptide
                else:
                    label = 1  # Has signal peptide
                
                # Get sequence
                i += 1
                if i < len(lines):
                    seq = lines[i].strip()
                
                    # Get annotation (S for signal peptide, O for mature protein)
                    i += 1
                    if i < len(lines) and not lines[i].startswith('>'):
                        annotation = lines[i].strip()
                        
                        # Find cleavage site if it exists
                        cleavage_site = -1
                        if 'S' in annotation and 'O' in annotation:
                            # Find position where 'S' changes to 'O'
                            for j in range(len(annotation) - 1):
                                if annotation[j] == 'S' and annotation[j+1] == 'O':
                                    cleavage_site = j + 1  # +1 because we want the position after S
                                    break
                        
                        sequences.append(seq)
                        labels.append(label)
                        annotations.append(annotation)
                        cleavage_sites.append(cleavage_site)
                        
                        i += 1
                    else:
                        # Skip if there's no annotation line
                        sequences.append(seq)
                        labels.append(label)
                        annotations.append('N' * len(seq))  # Default annotation
                        cleavage_sites.append(-1)  # No cleavage site
                else:
                    i += 1
                    
                continue
            
            i += 1
    
    return sequences, labels, annotations, cleavage_sites

# Function to encode amino acid sequences
def encode_sequences(sequences, max_length=100):
    # Define amino acid encoding dictionary
    aa_dict = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
        'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        'X': 0, 'U': 0, 'O': 0, 'B': 0, 'Z': 0, 'J': 0
    }
    
    # Encode sequences
    encoded_seqs = []
    for seq in sequences:
        encoded = [aa_dict.get(aa, 0) for aa in seq]
        encoded_seqs.append(encoded)
    
    # Pad sequences to the same length
    padded_seqs = pad_sequences(encoded_seqs, maxlen=max_length, padding='post')
    
    return padded_seqs

# Extract sequence features
def extract_features(sequences):
    # Calculate basic features for each sequence
    features = []
    
    for seq in sequences:
        # N-terminal region (first 30 amino acids)
        n_term = seq[:30] if len(seq) >= 30 else seq
        
        # Feature 1: Hydrophobicity of N-terminal region
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        avg_hydro = sum(hydrophobicity.get(aa, 0) for aa in n_term) / len(n_term)
        
        # Feature 2: Charge of N-terminal region
        pos_charged = ['K', 'R', 'H']
        neg_charged = ['D', 'E']
        pos_count = sum(1 for aa in n_term if aa in pos_charged)
        neg_count = sum(1 for aa in n_term if aa in neg_charged)
        net_charge = pos_count - neg_count
        
        # Feature 3: Presence of potential cleavage site pattern
        # Common motif: small, uncharged residue at -1, -3 positions
        small_uncharged = ['A', 'G', 'S', 'T', 'C']
        cleavage_site = False
        
        for i in range(15, min(30, len(seq))):
            if i >= 3 and seq[i-1] in small_uncharged and seq[i-3] in small_uncharged:
                cleavage_site = True
                break
        
        features.append([avg_hydro, net_charge, 1 if cleavage_site else 0])
    
    return np.array(features)

# Prepare cleavage site data for training
def prepare_cleavage_site_data(sequences, cleavage_sites, max_length=100):
    """
    Prepare data for cleavage site prediction.
    
    Args:
        sequences: List of protein sequences
        cleavage_sites: List of cleavage site positions (or -1 if no cleavage site)
        max_length: Maximum sequence length for padding
        
    Returns:
        X: Encoded sequences for model input
        y: Position labels for cleavage sites
    """
    # Encode sequences
    X = encode_sequences(sequences, max_length)
    
    # Create labels for each position in the sequence
    y = np.zeros((len(sequences), max_length))
    
    for i, site in enumerate(cleavage_sites):
        if site > 0 and site < max_length:
            # Mark the cleavage site position
            y[i, site] = 1
    
    return X, y

# Build a model for signal peptide classification
def build_sp_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Binary classification (SP or No-SP)
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build a multitask model for both SP classification and cleavage site prediction
def build_multitask_model(input_shape, max_length=100):
    # Input layer
    sequence_input = Input(shape=input_shape)
    
    # Shared layers
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(sequence_input)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(filters=256, kernel_size=3, activation='relu')(pool2)
    
    # SP classification branch
    flat_sp = Flatten()(conv3)
    dense_sp = Dense(128, activation='relu')(flat_sp)
    drop_sp = Dropout(0.5)(dense_sp)
    output_sp = Dense(2, activation='softmax', name='sp_output')(drop_sp)
    
    # Cleavage site prediction branch (simplified approach)
    # This is a simplified approach - a full implementation would use a more
    # complex architecture specifically designed for sequence labeling
    conv_cs = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv3)
    output_cs = Dense(1, activation='sigmoid', name='cleavage_output')(conv_cs)
    
    # Create model with multiple outputs
    model = Model(inputs=sequence_input, outputs=[output_sp, output_cs])
    
    # Compile with appropriate loss functions and metrics
    model.compile(
        optimizer='adam',
        loss={
            'sp_output': 'categorical_crossentropy',
            'cleavage_output': 'binary_crossentropy'
        },
        metrics={
            'sp_output': 'accuracy',
            'cleavage_output': 'accuracy'
        },
        loss_weights={
            'sp_output': 1.0,
            'cleavage_output': 0.5  # Weight the losses
        }
    )
    
    return model

# Main function to run the signal peptide prediction
def main():
    # Create directories for outputs
    os.makedirs('images', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # File paths
    train_file = 'data/signal_peptide_train.fasta'
    benchmark_file = 'data/signal_peptide_benchmark.fasta'
    
    # Process training data with annotations
    print("Processing training data...")
    train_sequences, train_labels, train_annotations, train_cleavage_sites = process_fasta_data(train_file)
    
    # Process benchmark data with annotations
    print("Processing benchmark data...")
    benchmark_sequences, benchmark_labels, benchmark_annotations, benchmark_cleavage_sites = process_fasta_data(benchmark_file)
    
    # Analyze cleavage site distribution
    valid_cleavage_sites = [site for site in train_cleavage_sites if site >= 0]
    
    if valid_cleavage_sites:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_cleavage_sites, bins=30)
        plt.title('Distribution of Cleavage Site Positions')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Frequency')
        plt.savefig('images/cleavage_site_distribution.png')
        plt.close()
        
        print(f"Found {len(valid_cleavage_sites)} sequences with cleavage sites")
        print(f"Average cleavage site position: {np.mean(valid_cleavage_sites):.2f}")
    else:
        print("No valid cleavage sites found in the training data.")
        print("Continuing with signal peptide prediction only.")
    
    # Split training data for model validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_sequences, train_labels, test_size=0.2, random_state=42
    )
    
    # Encode sequences
    max_length = 100
    X_train_encoded = encode_sequences(X_train, max_length)
    X_val_encoded = encode_sequences(X_val, max_length)
    
    # Convert labels to categorical for classification
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    # Build and train the signal peptide classification model
    print("Training signal peptide classification model...")
    model = build_sp_model((max_length, 1))
    
    # Reshape input for CNN
    X_train_reshaped = X_train_encoded.reshape(X_train_encoded.shape[0], X_train_encoded.shape[1], 1)
    X_val_reshaped = X_val_encoded.reshape(X_val_encoded.shape[0], X_val_encoded.shape[1], 1)
    
    # Train model
    history = model.fit(
        X_train_reshaped,
        y_train_cat,
        epochs=20,
        batch_size=32,
        validation_data=(X_val_reshaped, y_val_cat),
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/training_history.png')
    plt.close()
    
    # Evaluate on validation set
    y_val_pred_prob = model.predict(X_val_reshaped)
    y_val_pred = np.argmax(y_val_pred_prob, axis=1)
    y_val_true = np.array(y_val)
    
    # Generate validation classification report
    print("Validation Set Classification Report:")
    print(classification_report(y_val_true, y_val_pred, target_names=['No SP', 'SP']))
    
    # Evaluate on benchmark dataset
    print("\nEvaluating on benchmark dataset...")
    X_benchmark_encoded = encode_sequences(benchmark_sequences, max_length)
    X_benchmark_reshaped = X_benchmark_encoded.reshape(X_benchmark_encoded.shape[0], X_benchmark_encoded.shape[1], 1)
    y_benchmark_cat = to_categorical(benchmark_labels, num_classes=2)
    
    # Evaluate model on benchmark data
    benchmark_loss, benchmark_acc = model.evaluate(X_benchmark_reshaped, y_benchmark_cat)
    print(f"Benchmark Accuracy: {benchmark_acc:.4f}")
    print(f"Benchmark Loss: {benchmark_loss:.4f}")
    
    # Get predictions on benchmark data
    y_benchmark_pred_prob = model.predict(X_benchmark_reshaped)
    y_benchmark_pred = np.argmax(y_benchmark_pred_prob, axis=1)
    y_benchmark_true = np.array(benchmark_labels)
    
    # Generate benchmark classification report
    print("\nBenchmark Set Classification Report:")
    print(classification_report(y_benchmark_true, y_benchmark_pred, target_names=['No SP', 'SP']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_benchmark_true, y_benchmark_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No SP', 'SP'], 
                yticklabels=['No SP', 'SP'])
    plt.title('Confusion Matrix (Benchmark Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.close()
    
    # Save model
    model.save('model/signal_peptide_model.h5')
    print("Model saved as 'model/signal_peptide_model.h5'")
    
    # If there are valid cleavage sites, train a simplified cleavage site model
    if len(valid_cleavage_sites) > 100:  # Only if we have enough data
        print("\nTraining a basic cleavage site model...")
        # This would be a more complex implementation in a full project
        # For now, we just demonstrate how to extract and use the cleavage site data
        print("Note: A complete cleavage site prediction model would require a more complex implementation")
        
        # Generate cleavage site prediction examples
        examples = []
        for i in range(min(5, len(valid_cleavage_sites))):
            idx = next((j for j, site in enumerate(train_cleavage_sites) if site > 0), 0)
            seq = train_sequences[idx]
            site = train_cleavage_sites[idx]
            examples.append({
                'Sequence': seq,
                'Cleavage Site Position': site,
                'Signal Sequence': seq[:site],
                'Mature Protein Start': seq[site:site+10] + '...'
            })
        
        # Print examples
        print("\nExample cleavage sites:")
        for i, example in enumerate(examples):
            print(f"Example {i+1}:")
            print(f"  Signal sequence: {example['Signal Sequence']}")
            print(f"  Cleavage site: between position {example['Cleavage Site Position']-1} and {example['Cleavage Site Position']}")
            print(f"  Mature protein: {example['Mature Protein Start']}")
            print()

if __name__ == '__main__':
    main()