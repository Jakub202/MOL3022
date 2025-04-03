#!/usr/bin/env python3
# Enhanced script for signal peptide prediction with optional cleavage site detection
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import the functions from the model file
from signal_peptide_model import encode_sequences, extract_features, process_fasta_data

def predict_signal_peptide(seq_file, model_path='model/signal_peptide_model.h5'):
    """
    Predict signal peptides for sequences in a FASTA file
    
    Args:
        seq_file: Path to FASTA file with sequences
        model_path: Path to saved model
    """
    # Load model
    model = load_model(model_path)
    
    # Read sequences
    sequences = []
    seq_ids = []
    annotations = []
    
    with open(seq_file, 'r') as file:
        lines = file.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                header = lines[i].strip()
                seq_ids.append(header[1:])  # Remove '>' character
                
                # Get sequence
                i += 1
                if i < len(lines):
                    seq = lines[i].strip()
                    sequences.append(seq)
                    
                    # Check for annotation line
                    i += 1
                    if i < len(lines) and not lines[i].startswith('>'):
                        annotations.append(lines[i].strip())
                        i += 1
                    else:
                        annotations.append(None)
                else:
                    break
            else:
                i += 1
    
    if not sequences:
        print("No sequences found in the input file.")
        return
    
    # Encode sequences
    max_length = 100
    encoded_seqs = encode_sequences(sequences, max_length)
    reshaped_seqs = encoded_seqs.reshape(encoded_seqs.shape[0], encoded_seqs.shape[1], 1)
    
    # Make predictions
    predictions = model.predict(reshaped_seqs)
    pred_classes = np.argmax(predictions, axis=1)
    pred_probs = np.max(predictions, axis=1)
    
    # Print results
    print("\nSignal Peptide Prediction Results:")
    print("=" * 80)
    print("{:<20} {:<20} {:<12} {:<30}".format("ID", "Prediction", "Probability", "N-terminal Sequence"))
    print("-" * 80)
    
    cleavage_results = []
    
    for i, seq_id in enumerate(seq_ids):
        prediction = "Signal Peptide" if pred_classes[i] == 1 else "No Signal Peptide"
        n_terminal = sequences[i][:30] if len(sequences[i]) >= 30 else sequences[i]
        print("{:<20} {:<20} {:<12.3f} {:<30}".format(
            seq_id.split('|')[0], prediction, pred_probs[i], n_terminal
        ))
        
        # Process annotation for cleavage site if available
        if pred_classes[i] == 1 and annotations[i] is not None:
            annotation = annotations[i]
            if 'S' in annotation and 'O' in annotation:
                # Find cleavage site (transition from S to O)
                for j in range(len(annotation) - 1):
                    if annotation[j] == 'S' and annotation[j+1] == 'O':
                        cleavage_site = j + 1
                        signal_seq = sequences[i][:cleavage_site]
                        mature_seq = sequences[i][cleavage_site:]
                        
                        cleavage_results.append({
                            'id': seq_id.split('|')[0],
                            'sequence': sequences[i],
                            'annotation': annotation,
                            'cleavage_site': cleavage_site,
                            'signal_seq': signal_seq,
                            'mature_seq': mature_seq
                        })
                        break
    
    # Display cleavage site information if available
    if cleavage_results:
        print("\nCleavage Site Analysis for Signal Peptides:")
        print("=" * 80)
        print("{:<15} {:<8} {:<20} {:<20}".format(
            "ID", "Position", "Signal Sequence", "Mature Protein Start"
        ))
        print("-" * 80)
        
        for result in cleavage_results:
            signal_display = result['signal_seq']
            mature_display = result['mature_seq'][:10] + "..." if len(result['mature_seq']) > 10 else result['mature_seq']
            
            print("{:<15} {:<8} {:<20} {:<20}".format(
                result['id'], 
                result['cleavage_site'],
                signal_display, 
                mature_display
            ))
        
        # Visualize cleavage site distribution
        if len(cleavage_results) > 1:
            plt.figure(figsize=(10, 6))
            cleavage_positions = [r['cleavage_site'] for r in cleavage_results]
            sns.histplot(cleavage_positions, bins=range(min(cleavage_positions), max(cleavage_positions) + 2))
            plt.title('Cleavage Site Position Distribution')
            plt.xlabel('Position in Sequence')
            plt.ylabel('Count')
            plt.tight_layout()
            
            # Create images directory if it doesn't exist
            os.makedirs('images', exist_ok=True)
            plt.savefig('images/prediction_cleavage_sites.png')
            plt.close()
            print(f"\nCleavage site distribution saved to images/prediction_cleavage_sites.png")
    
    return predictions, sequences, seq_ids, cleavage_results

def evaluate_benchmark(benchmark_file='data/signal_peptide_benchmark.fasta', model_path='model/signal_peptide_model.h5'):
    """
    Evaluate the model on the benchmark dataset
    
    Args:
        benchmark_file: Path to benchmark FASTA file
        model_path: Path to saved model
    
    Returns:
        accuracy: Overall accuracy on benchmark
        report: Classification report
        cm: Confusion matrix
    """
    print(f"\nEvaluating model on benchmark dataset: {benchmark_file}")
    print("=" * 80)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None, None, None
    
    model = load_model(model_path)
    
    # Process benchmark data
    try:
        # Use the process_fasta_data function from signal_peptide_model
        benchmark_sequences, benchmark_labels, benchmark_annotations, benchmark_cleavage_sites = process_fasta_data(benchmark_file)
    except Exception as e:
        print(f"Error processing benchmark file: {e}")
        return None, None, None
    
    # Encode sequences
    max_length = 100
    encoded_seqs = encode_sequences(benchmark_sequences, max_length)
    reshaped_seqs = encoded_seqs.reshape(encoded_seqs.shape[0], encoded_seqs.shape[1], 1)
    
    # Make predictions
    print("Running predictions on benchmark dataset...")
    predictions = model.predict(reshaped_seqs)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(benchmark_labels, pred_classes)
    report = classification_report(benchmark_labels, pred_classes, target_names=['No SP', 'SP'])
    cm = confusion_matrix(benchmark_labels, pred_classes)
    
    # Display results
    print(f"\nBenchmark Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No SP', 'SP'], 
                yticklabels=['No SP', 'SP'])
    plt.title('Confusion Matrix (Benchmark Evaluation)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/benchmark_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to images/benchmark_confusion_matrix.png")
    
    # Calculate cleavage site accuracy if annotations are available
    if benchmark_annotations:
        correct_cleavage = 0
        total_with_sp = 0
        
        for i, (pred, label, annotation) in enumerate(zip(pred_classes, benchmark_labels, benchmark_annotations)):
            if label == 1:  # Only for sequences with SP
                total_with_sp += 1
                
                # Find actual cleavage site from annotation
                actual_site = -1
                if 'S' in annotation and 'O' in annotation:
                    for j in range(len(annotation) - 1):
                        if annotation[j] == 'S' and annotation[j+1] == 'O':
                            actual_site = j + 1
                            break
                
                # TODO: Add cleavage site prediction code here if you implement a dedicated cleavage site predictor
        
        if total_with_sp > 0:
            print(f"\nFound {total_with_sp} sequences with signal peptides in benchmark set")
            # If you implement cleavage site prediction:
            # print(f"Cleavage Site Accuracy: {correct_cleavage / total_with_sp:.4f}")
    
    return accuracy, report, cm

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  For prediction: python predict_signal_peptides.py predict <fasta_file> [model_file]")
        print("  For benchmark evaluation: python predict_signal_peptides.py evaluate [benchmark_file] [model_file]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "predict":
        if len(sys.argv) < 3:
            print("Usage for prediction: python predict_signal_peptides.py predict <fasta_file> [model_file]")
            sys.exit(1)
        
        fasta_file = sys.argv[2]
        model_file = sys.argv[3] if len(sys.argv) > 3 else 'model/signal_peptide_model.h5'
        
        if not os.path.exists(fasta_file):
            print(f"Error: File {fasta_file} not found.")
            sys.exit(1)
        
        predict_signal_peptide(fasta_file, model_file)
    
    elif command == "evaluate":
        benchmark_file = sys.argv[2] if len(sys.argv) > 2 else 'data/signal_peptide_benchmark.fasta'
        model_file = sys.argv[3] if len(sys.argv) > 3 else 'model/signal_peptide_model.h5'
        
        if not os.path.exists(benchmark_file):
            print(f"Error: Benchmark file {benchmark_file} not found.")
            sys.exit(1)
        
        evaluate_benchmark(benchmark_file, model_file)
    
    else:
        print("Unknown command. Use 'predict' or 'evaluate'.")
        print("  For prediction: python predict_signal_peptides.py predict <fasta_file> [model_file]")
        print("  For benchmark evaluation: python predict_signal_peptides.py evaluate [benchmark_file] [model_file]")
        sys.exit(1)