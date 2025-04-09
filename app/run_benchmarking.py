#!/usr/bin/env python3
"""
Script to run benchmarking using a CSV data string.
This is useful for quick testing and visualization.
"""
import os
import sys
from benchmark import benchmark_from_data_string

# Example usage: python run_benchmarking.py data.csv output/results

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_benchmarking.py [data_file.csv] [output_dir]")
        sys.exit(1)
    
    # Get input file and output directory
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results'
    
    # Read data from file
    try:
        with open(input_file, 'r') as f:
            data_str = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Run benchmarking
    print(f"Running benchmarking with data from {input_file}")
    print(f"Results will be saved to {output_dir}")
    
    try:
        metrics = benchmark_from_data_string(data_str, output_dir)
        print("Benchmarking completed successfully")
        print(f"Results saved to {output_dir}")
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        sys.exit(1)
    
    # Print summary of metrics
    print("\nMetrics Summary:")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"SP Class Accuracy: {metrics['sp_class_accuracy']:.4f}")
    print("\nCleavage site precision and recall metrics have been saved as plots.")
    print("See the figure files in the output directory for visualization.")

if __name__ == "__main__":
    main()
