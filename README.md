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

Training parameters are defined in `app/model.py`:
- Maximum sequence length: 100
- Batch size: 32
- Epochs: 100 (with early stopping)
- Validation split: 0.2

### Benchmarking Model

To evaluate the model on a benchmark dataset:

```bash
python app/main.py benchmark
```

The benchmark evaluates the model's performance on test data from `data/benchmark.fasta` and saves results to the `output/results` directory.

### Making Predictions

To predict signal peptides in new protein sequences:

```bash
python app/main.py predict <path-to-fasta-file>
```

Replace `<path-to-fasta-file>` with the path to your FASTA file containing protein sequences (example: `predict.fasta`).

## Data Format

### Input Files
- Training data: `data/train6.fasta`
- Benchmark data: `data/benchmark.fasta`
- Prediction input: Any properly formatted FASTA file

### Output Files
- Trained model: `models/sp_model_best.keras`
- Benchmark results: `output/results/benchmark_results.csv` and metrics visualizations
- Prediction results: Displayed in console and optionally saved to file

## Model Architecture

The model employs a multi-task learning approach with three outputs:
- Binary cross-entropy for signal peptide presence prediction
- Categorical cross-entropy for signal peptide class prediction
- Custom masked MSE loss for cleavage site prediction, which ignores examples without signal peptides

Signal peptide cleavage positions are constrained to biologically plausible ranges based on the signal peptide type:
- Sec/SPI (class 0): positions 15-35
- Tat/SPI (class 1): positions 20-40
- Sec/SPII (class 2): positions 15-30
- Sec/SPIII (class 3): positions 15-30


