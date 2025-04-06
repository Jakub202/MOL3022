#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from signal_peptide_model import encode_sequences, train_classification_model, train_cleavage_site_model
from data_preparation import parse_fasta_file

def prepare_data(data_df, max_length=100):
    """
    Prepare data for training both models
    
    Args:
        data_df: DataFrame with protein sequences and metadata
        max_length: Maximum sequence length
        
    Returns:
        X_encoded: Encoded sequences
        y_class: Classification labels
        y_cleavage: Cleavage site positions
    """
    # Prepare sequences and labels
    sequences = data_df['sequence'].tolist()
    X_encoded = encode_sequences(sequences, max_length=max_length)
    
    # Classification labels
    y_class = data_df['has_sp'].values
    
    # Cleavage site positions - only for sequences with signal peptides
    sp_mask = data_df['has_sp'] == 1
    sp_sequences = data_df[sp_mask]
    cleavage_sites = sp_sequences['cleavage_site'].values
    
    return X_encoded, y_class, sp_mask, cleavage_sites

def main(args):
    # Create output directory
    os.makedirs(args.model_path, exist_ok=True)
    
    print(f"Loading data from {args.train_fasta}...")
    train_df = parse_fasta_file(args.train_fasta)
    
    print(f"Found {len(train_df)} sequences.")
    print(f"Sequences with signal peptides: {train_df['has_sp'].sum()}")
    print(f"Sequences without signal peptides: {len(train_df) - train_df['has_sp'].sum()}")
    
    print("Preparing data for training...")
    X_encoded, y_class, sp_mask, cleavage_sites = prepare_data(train_df, args.max_length)
    
    # Train classification model
    if not args.skip_classification:
        print("\nTraining classification model...")
        classification_model, _ = train_classification_model(
            X_encoded, y_class,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_path=args.model_path
        )
        print("Classification model training complete!")
    
    # Train cleavage site model (only on sequences with signal peptides)
    if not args.skip_cleavage:
        print("\nTraining cleavage site prediction model...")
        if sum(sp_mask) > 0:
            # Take only sequences with signal peptides
            X_sp = X_encoded[sp_mask]
            cleavage_model, _ = train_cleavage_site_model(
                X_sp, cleavage_sites,
                max_length=args.max_length,
                batch_size=args.batch_size,
                epochs=args.epochs,
                model_path=args.model_path
            )
            print("Cleavage site model training complete!")
        else:
            print("No sequences with signal peptides found. Skipping cleavage site model training.")
    
    print("\nModel training complete! Models saved to:", args.model_path)
    print("Models saved in the modern .keras format for better compatibility.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train signal peptide prediction models')
    parser.add_argument('--train-fasta', default='data/signal_peptide_train.fasta',
                        help='Path to training FASTA file')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--model-path', default='models',
                        help='Path to save trained models')
    parser.add_argument('--skip-classification', action='store_true',
                        help='Skip training the classification model')
    parser.add_argument('--skip-cleavage', action='store_true',
                        help='Skip training the cleavage site model')
    
    args = parser.parse_args()
    main(args)
