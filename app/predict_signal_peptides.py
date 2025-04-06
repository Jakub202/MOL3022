#!/usr/bin/env python3
# Enhanced script for signal peptide prediction with cleavage site detection
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error

# Import the functions from the model file
from signal_peptide_model import (
    encode_sequences, extract_features, process_fasta_data,
    predict_with_cleavage, visualize_cleavage_sites
)

def load_fasta_for_prediction(seq_file):
    """
    Load sequences from a FASTA file for prediction
    
    Args:
        seq_file: Path to the FASTA file
        
    Returns:
        sequences: List of sequences
        seq_ids: List of sequence IDs
    """
    sequences = []
    seq_ids = []
    
    with open(seq_file, 'r') as file:
        current_id = None
        current_seq = ""
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append(current_seq)
                    seq_ids.append(current_id)
                    
                current_id = line[1:]  # Remove '>' character
                current_seq = ""
            else:
                current_seq += line
        
        if current_id is not None:  # Add the last sequence
            sequences.append(current_seq)
            seq_ids.append(current_id)
    
    return sequences, seq_ids

def predict_signal_peptide(seq_file, model_path='model/signal_peptide_model.keras'):
    """
    Predict signal peptides and cleavage sites for sequences in a FASTA file
    
    Args:
        seq_file: Path to FASTA file with sequences
        model_path: Path to saved model
    """
    # Load sequences
    print(f"Loading sequences from {seq_file}...")
    sequences, seq_ids = load_fasta_for_prediction(seq_file)
    
    if not sequences:
        print("No sequences found in the input file.")
        return
    
    # Make predictions using the dual-output model
    print("Making predictions...")
    try:
        sp_predictions, cleavage_predictions = predict_with_cleavage(sequences, model_path)
        pred_classes = sp_predictions['classes']
        pred_probs = sp_predictions['probabilities']
        cleavage_positions = cleavage_predictions['positions']
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Print results
    print("\nSignal Peptide and Cleavage Site Prediction Results:")
    print("=" * 90)
    print("{:<20} {:<20} {:<12} {:<15} {:<30}".format(
        "ID", "Prediction", "Probability", "Cleavage Site", "N-terminal Sequence"))
    print("-" * 90)
    
    # Collect results for sequences with signal peptides
    sp_results = []
    
    for i, seq_id in enumerate(seq_ids):
        short_id = seq_id.split('|')[0]
        prediction = "Signal Peptide" if pred_classes[i] == 1 else "No Signal Peptide"
        n_terminal = sequences[i][:30] if len(sequences[i]) >= 30 else sequences[i]
        
        cleavage_info = str(cleavage_positions[i]) if pred_classes[i] == 1 else "N/A"
        
        print("{:<20} {:<20} {:<12.3f} {:<15} {:<30}".format(
            short_id, prediction, pred_probs[i], cleavage_info, n_terminal
        ))
        
        # Collect sequences with signal peptides for visualization
        if pred_classes[i] == 1:
            sp_results.append({
                'id': short_id,
                'sequence': sequences[i],
                'cleavage_site': cleavage_positions[i]
            })
    
    # Visualize cleavage sites if any were predicted
    if sp_results:
        print(f"\nFound {len(sp_results)} sequences with signal peptides")
        
        # Create images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Extract data for visualization
        sp_sequences = [r['sequence'] for r in sp_results]
        sp_cleavage_sites = [r['cleavage_site'] for r in sp_results]
        sp_ids = [r['id'] for r in sp_results]
        
        # Visualize the cleavage sites
        output_file = 'images/predicted_cleavage_sites.png'
        visualize_cleavage_sites(sp_sequences, sp_cleavage_sites, sp_ids, output_file)
        print(f"Cleavage site visualization saved to {output_file}")
        
        # Also create a distribution plot if we have enough data
        if len(sp_results) > 3:
            plt.figure(figsize=(10, 6))
            sns.histplot(sp_cleavage_sites, bins=range(min(sp_cleavage_sites), max(sp_cleavage_sites) + 2))
            plt.title('Predicted Cleavage Site Position Distribution')
            plt.xlabel('Position in Sequence')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('images/cleavage_site_distribution.png')
            plt.close()
            print(f"Cleavage site distribution saved to images/cleavage_site_distribution.png")
    
    return pred_classes, pred_probs, cleavage_positions, sequences, seq_ids

def benchmark_model(benchmark_file='data/signal_peptide_benchmark.fasta', model_path='model/signal_peptide_model.keras'):
    """
    Benchmark the model using a test dataset and compute evaluation metrics
    
    Args:
        benchmark_file: Path to the benchmark FASTA file
        model_path: Path to the saved model
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from data_preparation import parse_fasta_file
    import os
    
    print("\nLoading benchmark dataset...")
    # Load benchmark data with true labels
    benchmark_df = parse_fasta_file(benchmark_file)
    sequences = list(benchmark_df['sequence'])
    true_labels = list(benchmark_df['has_sp'])
    seq_ids = list(benchmark_df['protein_id'])
    
    print(f"Loaded {len(sequences)} benchmark sequences")
    
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    
    # Make predictions with cleavage sites
    print("Making predictions...")
    sp_predictions, cleavage_predictions = predict_with_cleavage(sequences, model_path)
    pred_classes = sp_predictions['classes']
    pred_probs = sp_predictions['probabilities']
    cleavage_positions = cleavage_predictions['positions']
    
    # Calculate metrics for signal peptide prediction
    accuracy = accuracy_score(true_labels, pred_classes)
    precision = precision_score(true_labels, pred_classes)
    recall = recall_score(true_labels, pred_classes)
    f1 = f1_score(true_labels, pred_classes)
    cm = confusion_matrix(true_labels, pred_classes)
    
    # Print results
    print("\nModel Benchmark Results:")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"{cm[0][0]:6d} {cm[0][1]:6d}  | {cm[0][0] + cm[0][1]:6d} (True Negatives)")
    print(f"{cm[1][0]:6d} {cm[1][1]:6d}  | {cm[1][0] + cm[1][1]:6d} (True Positives)")
    print("-" * 19)
    print(f"{cm[0][0] + cm[1][0]:6d} {cm[0][1] + cm[1][1]:6d}  | {np.sum(cm):6d} (Total)")
    
    # Evaluate cleavage site prediction accuracy
    cleavage_mae = None
    cleavage_accuracy = None
    
    # Get actual cleavage sites from the benchmark data
    true_cleavage_sites = list(benchmark_df['cleavage_site'])
    
    # Calculate MAE only for sequences that truly have signal peptides
    sp_indices = [i for i, label in enumerate(true_labels) if label == 1]
    if sp_indices:
        true_cleavage = [true_cleavage_sites[i] for i in sp_indices]
        pred_cleavage = [cleavage_positions[i] for i in sp_indices]
        
        # Filter out invalid cleavage sites
        valid_indices = [i for i, site in enumerate(true_cleavage) if site > 0]
        if valid_indices:
            true_cleavage_valid = [true_cleavage[i] for i in valid_indices]
            pred_cleavage_valid = [pred_cleavage[i] for i in valid_indices]
            
            # Calculate metrics
            cleavage_mae = mean_absolute_error(true_cleavage_valid, pred_cleavage_valid)
            
            # Calculate accuracy within a tolerance window
            tolerance = 1  # Consider prediction correct if within ±1 position
            correct = 0
            for true_pos, pred_pos in zip(true_cleavage_valid, pred_cleavage_valid):
                if abs(true_pos - pred_pos) <= tolerance:
                    correct += 1
            
            cleavage_accuracy = correct / len(true_cleavage_valid)
            
            print("\nCleavage Site Prediction Results:")
            print("=" * 80)
            print(f"Mean Absolute Error: {cleavage_mae:.2f} positions")
            print(f"Accuracy (±{tolerance} position): {cleavage_accuracy:.4f}")
            
            # Plot comparison of true vs predicted cleavage sites
            plt.figure(figsize=(10, 6))
            plt.scatter(true_cleavage_valid, pred_cleavage_valid, alpha=0.5)
            
            # Add diagonal line for perfect predictions
            max_val = max(max(true_cleavage_valid), max(pred_cleavage_valid))
            plt.plot([0, max_val], [0, max_val], 'r--')
            
            plt.title('True vs Predicted Cleavage Site Positions')
            plt.xlabel('True Position')
            plt.ylabel('Predicted Position')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            os.makedirs('images', exist_ok=True)
            plt.savefig('images/cleavage_site_comparison.png')
            plt.close()
            print("Cleavage site comparison plot saved to images/cleavage_site_comparison.png")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['No SP', 'SP'])
    plt.yticks(tick_marks, ['No SP', 'SP'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    os.makedirs('images', exist_ok=True)
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    
    # Detailed results table
    print("\nDetailed Prediction Results:")
    print("=" * 90)
    print("{:<20} {:<13} {:<13} {:<15} {:<15}".format(
        "ID", "True Label", "Predicted", "True Site", "Predicted Site"))
    print("-" * 90)
    
    for i, seq_id in enumerate(seq_ids):
        true = "Signal Peptide" if true_labels[i] == 1 else "No Signal Peptide"
        pred = "Signal Peptide" if pred_classes[i] == 1 else "No Signal Peptide"
        
        true_site = true_cleavage_sites[i] if true_labels[i] == 1 else "N/A"
        pred_site = cleavage_positions[i] if pred_classes[i] == 1 else "N/A"
        
        match = "✓" if true_labels[i] == pred_classes[i] else "✗"
        
        print("{:<20} {:<13} {:<13} {:<15} {:<15} {}".format(
            seq_id, true, pred, true_site, pred_site, match
        ))
    
    # Return metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': pred_classes,
        'true_labels': true_labels,
        'cleavage_mae': cleavage_mae,
        'cleavage_accuracy': cleavage_accuracy
    }
    
    return metrics

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict signal peptides in protein sequences')
    parser.add_argument('--fasta', type=str, help='Path to the FASTA file with sequences to predict')
    parser.add_argument('--model', type=str, default='model/signal_peptide_model.keras', 
                        help='Path to the trained model')
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run benchmark on the test dataset')
    parser.add_argument('--benchmark_file', type=str, default='data/signal_peptide_benchmark.fasta',
                        help='Path to the benchmark FASTA file')
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("Running model benchmark...")
        benchmark_model(args.benchmark_file, args.model)
    elif args.fasta:
        print(f"Predicting signal peptides for sequences in {args.fasta}...")
        predict_signal_peptide(args.fasta, args.model)
    else:
        print("Please provide a FASTA file path with --fasta or use --benchmark to run the benchmark")
        sys.exit(1)