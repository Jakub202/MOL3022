# Signal Peptide Prediction Tool

A deep learning-based tool for predicting signal peptides, their classes, and cleavage sites in protein sequences.

## Overview

Signal peptides are short amino acid sequences that direct proteins to their proper cellular locations. This tool uses deep learning to:

1. Predict whether a protein sequence contains a signal peptide
2. Identify the signal peptide class (S: Sec/SPI, T: Tat/SPI or Tat/SPII, L: Sec/SPII, P: Sec/SPIII)
3. Predict the position of the signal peptide cleavage site

The model architecture combines convolutional neural networks (CNNs) and bidirectional LSTMs to learn the complex patterns in protein sequences that indicate the presence of signal peptides. It also takes into account the kingdom (Archaea, EUKARYA, bacteria-negative, bacteria-positive) to improve prediction accuracy.

## Installation

### Virtual Environment Setup
It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

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
python app/main.py train
```

The model uses training data from `data/signal_peptide_train.fasta` and saves the trained model to the `models` directory.

Training parameters can be modified in `app/model.py`:
- Maximum sequence length: 100
- Batch size: 32
- Epochs: 100
- Validation split: 0.2

### Benchmarking Model

To evaluate the model on a benchmark dataset:

```bash
python app/main.py benchmark
```

The benchmark uses test data from `data/signal_peptide_benchmark.fasta` and saves results to the `results` directory.

### Making Predictions

To predict signal peptides in new protein sequences:

```bash
python app/main.py predict <path-to-fasta-file>
```

Replace `<path-to-fasta-file>` with the path to your FASTA file containing protein sequences.

## Data Format

### Input Files
- Training data: `data/signal_peptide_train.fasta`
- Benchmark data: `data/signal_peptide_benchmark.fasta`
- Prediction input: Any properly formatted FASTA file

### Output Files
- Trained model: `models/sp_model_best.keras`
- Benchmark results: `results/benchmark_metrics.txt`, `results/confusion_matrix.png`, etc.
- Prediction results: Displayed in the console and saved to specified output file if provided

## Model Architecture

The model employs a multi-task learning approach with three outputs:
- Binary cross-entropy for signal peptide presence prediction
- Categorical cross-entropy for signal peptide class prediction
- Custom masked MSE loss for cleavage site prediction, which ignores examples without signal peptide


