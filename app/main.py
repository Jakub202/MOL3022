#!/usr/bin/env python3
"""
Main entry point for the signal peptide prediction pipeline.
Handles training, benchmarking, and prediction commands.

Usage:
    python app/main.py train
    python app/main.py benchmark
    python app/main.py predict <path-to-fasta-file>
"""
import os
import sys
import argparse
from model import train_model_from_file
from benchmark import benchmark_model


# Default paths
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
)
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "sp_model.keras")
TRAIN_FASTA = "train.fasta"
BENCHMARK_FASTA = "benchmark.fasta"

# Constants
MAX_LENGTH = 100
BATCH_SIZE = 32
EPOCHS = 20


def train(args):
    """Train the model using default parameters"""
    print("=== Starting Model Training ===")

    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Set paths
    train_fasta_path = os.path.join(DATA_DIR, TRAIN_FASTA)

    # Check if training file exists
    if not os.path.exists(train_fasta_path):
        print(f"ERROR: Training data file not found: {train_fasta_path}")
        print("Please check the file path or download the training data.")
        return

    # Train model
    model = train_model_from_file(
        fasta_file=train_fasta_path,
        model_path=MODEL_DIR,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    print("=== Model Training Complete ===")
    print(f"Model saved to: {DEFAULT_MODEL_PATH}")


def benchmark(args):
    """Benchmark the model using default parameters"""
    print("=== Starting Model Benchmarking ===")

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Set paths
    benchmark_fasta_path = os.path.join(DATA_DIR, BENCHMARK_FASTA)

    # Check if benchmark file exists
    if not os.path.exists(benchmark_fasta_path):
        print(f"ERROR: Benchmark data file not found: {benchmark_fasta_path}")
        print("Please check the file path or download the benchmark data.")
        return

    # Check if model exists
    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"ERROR: Model file not found: {DEFAULT_MODEL_PATH}")
        print("Please train the model first (python app/main.py train).")
        return

    # Run benchmarking
    metrics = benchmark_model(
        benchmark_file=benchmark_fasta_path,
        model_path=DEFAULT_MODEL_PATH,
        output_dir=RESULTS_DIR,
        max_length=MAX_LENGTH,
    )

    print("=== Model Benchmarking Complete ===")
    print(f"Results saved to: {RESULTS_DIR}")


def predict(args):
    """Make predictions on a FASTA file"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Predict signal peptides")
    parser.add_argument("--fasta", required=True, help="Path to FASTA file")
    args = parser.parse_args(args)

    fasta_path = args.fasta

    if not os.path.exists(fasta_path):
        print(f"Error: File not found: {fasta_path}")
        return

    print(f"=== Making Predictions on {os.path.basename(fasta_path)} ===")
    print("Loading sequences...")

    # Use the new predict_from_file function from model.py
    from model import predict_from_file

    # Make predictions directly from the file
    results_df = predict_from_file(fasta_path, DEFAULT_MODEL_PATH, MAX_LENGTH)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(
        RESULTS_DIR, os.path.basename(fasta_path).replace(".fasta", "_predictions.csv")
    )
    results_df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")
    print("\nSample predictions:")
    print(results_df.head())


def main():
    """Main function to handle command line arguments"""

    # Check debug mode first
    if DEBUG and len(sys.argv) < 2:
        print("\n=== DEBUG MODE: Automatically running benchmark ===")
        train([])
        return

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app/main.py train")
        print("  python app/main.py benchmark")
        print("  python app/main.py predict <path-to-fasta-file>")
        sys.exit(1)

    # Get command
    command = sys.argv[1]

    # Execute command
    if command == "train":
        train(sys.argv[2:])
    elif command == "benchmark":
        benchmark(sys.argv[2:])
    elif command == "predict":
        predict(sys.argv[2:])
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, benchmark, predict")
        sys.exit(1)


DEBUG = True  # Set to False for production
if __name__ == "__main__":
    main()
