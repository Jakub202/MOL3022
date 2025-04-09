#!/usr/bin/env python3
"""
Main script to run the entire signal peptide prediction pipeline: data preprocessing, model training, and benchmarking.
"""
import os
import argparse
from model import train_model_from_file
from benchmark import benchmark_model

def main():
    parser = argparse.ArgumentParser(description='Run signal peptide prediction pipeline')
    
    # General arguments
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing data files')
    parser.add_argument('--output-dir', default='output',
                      help='Directory to save output files')
    parser.add_argument('--max-length', type=int, default=100,
                      help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--train', action='store_true',
                      help='Run training phase')
    parser.add_argument('--train-fasta', default='train.fasta',
                      help='Filename of training FASTA file (relative to data-dir)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    
    # Benchmarking arguments
    parser.add_argument('--benchmark', action='store_true',
                      help='Run benchmarking phase')
    parser.add_argument('--benchmark-fasta', default='benchmark.fasta',
                      help='Filename of benchmark FASTA file (relative to data-dir)')
    parser.add_argument('--model', default=None,
                      help='Path to trained model (default: output-dir/models/sp_model.keras)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_dir = os.path.join(args.output_dir, 'models')
    results_dir = os.path.join(args.output_dir, 'results')
    
    # Set default model path if not provided
    if args.model is None:
        args.model = os.path.join(model_dir, 'sp_model.keras')
    
    # Run training if requested
    if args.train:
        print("=== Starting Model Training ===")
        train_fasta_path = os.path.join(args.data_dir, args.train_fasta)
        
        if not os.path.exists(train_fasta_path):
            print(f"ERROR: Training data file not found: {train_fasta_path}")
            print("Please check the file path or download the training data.")
            return
        
        model = train_model_from_file(
            fasta_file=train_fasta_path,
            model_path=model_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        print("=== Model Training Complete ===")
    
    # Run benchmarking if requested
    if args.benchmark:
        print("\n=== Starting Model Benchmarking ===")
        benchmark_fasta_path = os.path.join(args.data_dir, args.benchmark_fasta)
        
        if not os.path.exists(benchmark_fasta_path):
            print(f"ERROR: Benchmark data file not found: {benchmark_fasta_path}")
            print("Please check the file path or download the benchmark data.")
            return
        
        if not os.path.exists(args.model):
            print(f"ERROR: Model file not found: {args.model}")
            print("Please train the model first or provide a valid model path.")
            return
        
        metrics = benchmark_model(
            benchmark_file=benchmark_fasta_path,
            model_path=args.model,
            output_dir=results_dir,
            max_length=args.max_length
        )
        print("=== Model Benchmarking Complete ===")
        print(f"Results saved to: {results_dir}")
    
    # Print completion message
    if not args.train and not args.benchmark:
        print("No actions requested. Please use --train and/or --benchmark flags.")
    else:
        print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()