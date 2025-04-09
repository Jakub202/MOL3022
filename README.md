# Signal Peptide Prediction Tool

A deep learning-based tool for predicting signal peptides, their classes, and cleavage sites in protein sequences.

## Overview

Signal peptides are short amino acid sequences that direct proteins to their proper cellular locations. This tool uses deep learning to:

1. Predict whether a protein sequence contains a signal peptide
2. Identify the signal peptide class (S: Sec/SPI, T: Tat/SPI or Tat/SPII, L: Sec/SPII, P: Sec/SPIII)
3. Predict the position of the signal peptide cleavage site

The model architecture combines convolutional neural networks (CNNs) and bidirectional LSTMs to learn the complex patterns in protein sequences that indicate the presence of signal peptides. It also takes into account the kingdom (Archaea, EUKARYA, bacteria-negative, bacteria-positive) to improve prediction accuracy.

## Installation

### Requirements
- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn

Install dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn seaborn
```

## Usage

### Training Model

To train the signal peptide prediction model:

```bash
python app/model.py --train-fasta data/signal_peptide_train.fasta --model-path models --epochs 100
```

Options:
- `--train-fasta`: Path to the training FASTA file
- `--max-length`: Maximum sequence length (default: 100)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--model-path`: Directory to save model (default: 'models')

### Benchmarking Model

To evaluate the model on a benchmark dataset:

```bash
python app/benchmark.py --benchmark-fasta data/signal_peptide_benchmark.fasta --model models/sp_model.keras --output-dir results
```

Options:
- `--benchmark-fasta`: Path to benchmark FASTA file
- `--model`: Path to trained model
- `--output-dir`: Directory to save benchmark results

### Making Predictions

#### From a FASTA File

```bash
python app/predict.py --fasta your_sequences.fasta --model models/sp_model.keras --output predictions.csv
```

Options:
- `--fasta`: Path to FASTA file with sequences
- `--model`: Path to trained model
- `--output`: Path to save prediction results (optional)

#### For a Single Sequence

```bash
python app/predict.py --sequence "MLLSVPLLLGLLGLAVASNPVFA" --kingdom "EUKARYA" --name "MyProtein" --model models/sp_model.keras
```

Options:
- `--sequence`: The amino acid sequence to analyze
- `--kingdom`: Kingdom information (Archaea, EUKARYA, bacteria-negative, bacteria-positive)
- `--name`: Name for the sequence (default: "Query")
- `--model`: Path to trained model
- `--output`: Path to save prediction results (optional)

## Data Format

The tool expects FASTA files with the following format:


