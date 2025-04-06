#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, accuracy_score

from signal_peptide_model import encode_sequences, evaluate_classification_model, evaluate_cleavage_site_model
from data_preparation import parse_fasta_file

def plot_confusion_matrix(cm, classes, output_path):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_cleavage_site_error(true_sites, pred_sites, output_path):
    """
    Plot cleavage site prediction errors
    
    Args:
        true_sites: True cleavage site positions
        pred_sites: Predicted cleavage site positions
        output_path: Path to save the plot
    """
    errors = pred_sites - true_sites
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.title('Cleavage Site Prediction Error Distribution')
    plt.xlabel('Error (Predicted - True)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Also create accuracy within different tolerances plot
    tolerances = list(range(1, 11))
    accuracies = []
    
    for tol in tolerances:
        correct = np.sum(np.abs(errors) <= tol)
        accuracies.append(correct / len(errors) * 100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tolerances, accuracies, marker='o', linewidth=2)
    plt.title('Cleavage Site Prediction Accuracy at Different Tolerance Levels')
    plt.xlabel('Tolerance (amino acid positions)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.xticks(tolerances)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_tolerance.png'))
    plt.close()

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark data
    print(f"Loading data from {args.benchmark_fasta}...")
    data_df = parse_fasta_file(args.benchmark_fasta)
    
    print(f"Found {len(data_df)} sequences.")
    print(f"Sequences with signal peptides: {data_df['has_sp'].sum()}")
    print(f"Sequences without signal peptides: {len(data_df) - data_df['has_sp'].sum()}")
    
    # Encode sequences
    sequences = data_df['sequence'].tolist()
    X_encoded = encode_sequences(sequences, max_length=args.max_length)
    
    # Classification labels
    y_class = data_df['has_sp'].values
    
    # Evaluate classification model
    print("\nEvaluating classification model...")
    try:
        # Define custom objects for loading the model
        custom_objects = {
            'MeanSquaredError': MeanSquaredError,
            'MeanAbsoluteError': MeanAbsoluteError
        }
        
        classification_model = load_model(args.classification_model, custom_objects=custom_objects)
        
        # Get predictions
        y_pred_prob = classification_model.predict(X_encoded)
        y_pred_class = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_class, y_pred_class)
        conf_matrix = confusion_matrix(y_class, y_pred_class)
        class_report = classification_report(y_class, y_pred_class)
        
        # Print results
        print(f"Classification Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            conf_matrix, 
            classes=['No SP', 'Has SP'],
            output_path=os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
    except Exception as e:
        print(f"Error evaluating classification model: {e}")
    
    # Evaluate cleavage site model (only for sequences with signal peptides)
    print("\nEvaluating cleavage site model...")
    try:
        # Define custom objects for loading the model
        custom_objects = {
            'MeanSquaredError': MeanSquaredError,
            'MeanAbsoluteError': MeanAbsoluteError
        }
        
        cleavage_site_model = load_model(args.cleavage_site_model, custom_objects=custom_objects)
        
        # Filter sequences with signal peptides
        sp_mask = data_df['has_sp'] == 1
        sp_data = data_df[sp_mask]
        
        if len(sp_data) > 0:
            # Get encoded sequences and true cleavage sites
            X_sp = X_encoded[sp_mask]
            true_sites = sp_data['cleavage_site'].values
            
            # Get predictions
            pred_sites = cleavage_site_model.predict(X_sp).flatten()
            
            # Round predictions to nearest integer (cleavage sites are integers)
            pred_sites_rounded = np.round(pred_sites).astype(int)
            
            # Calculate metrics
            mae = mean_absolute_error(true_sites, pred_sites)
            
            # Calculate accuracy within different tolerances
            tolerances = [1, 2, 3, 5, 10]
            for tol in tolerances:
                correct = np.sum(np.abs(pred_sites_rounded - true_sites) <= tol)
                acc_tol = correct / len(true_sites)
                print(f"Accuracy within {tol} positions: {acc_tol:.4f}")
            
            # Print results
            print(f"Mean Absolute Error: {mae:.4f}")
            
            # Plot error distribution
            plot_cleavage_site_error(
                true_sites,
                pred_sites,
                output_path=os.path.join(args.output_dir, 'cleavage_site_error.png')
            )
            
            # Save detailed results to file
            results_df = pd.DataFrame({
                'protein_id': sp_data['protein_id'].values,
                'sequence': sp_data['sequence'].values,
                'true_cleavage_site': true_sites,
                'predicted_cleavage_site': pred_sites,
                'rounded_prediction': pred_sites_rounded,
                'absolute_error': np.abs(pred_sites_rounded - true_sites)
            })
            
            results_df.to_csv(
                os.path.join(args.output_dir, 'cleavage_site_results.csv'),
                index=False
            )
            
        else:
            print("No sequences with signal peptides found. Skipping cleavage site model evaluation.")
            
    except Exception as e:
        print(f"Error evaluating cleavage site model: {e}")
    
    print("\nBenchmark complete! Results saved to:", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark signal peptide prediction models')
    parser.add_argument('--benchmark-fasta', default='data/signal_peptide_benchmark.fasta',
                        help='Path to benchmark FASTA file')
    parser.add_argument('--classification-model', default='models/classification_model.keras',  # Update default path
                        help='Path to trained classification model')
    parser.add_argument('--cleavage-site-model', default='models/cleavage_site_model.keras',  # Update default path
                        help='Path to trained cleavage site model')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('--output-dir', default='benchmark_results',
                        help='Directory to save benchmark results')
    
    args = parser.parse_args()
    main(args)
