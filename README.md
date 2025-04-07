# Signal Peptide Prediction Tool

A machine learning tool for predicting signal peptides in protein sequences and their cleavage sites.

## Overview

Signal peptides are short amino acid sequences that direct proteins to their proper cellular locations. This tool uses deep learning to:

1. Predict whether a protein sequence contains a signal peptide
2. Identify the position of the signal peptide cleavage site

The model architecture combines convolutional neural networks (CNNs) and bidirectional LSTMs to learn the complex patterns in protein sequences that indicate the presence of signal peptides.

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/MOL3022.git
cd MOL3022
```

### Setup Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The required dependencies include:
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Biopython

## Usage

### Training Models

To train the signal peptide prediction models:

```bash
python app/train_models.py --train-fasta data/signal_peptide_train.fasta --model-path models --epochs 100
```

Options:
- `--train-fasta`: Path to the training FASTA file
- `--max-length`: Maximum sequence length (default: 100)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--model-path`: Directory to save models (default: 'models')
- `--skip-classification`: Skip training the classification model
- `--skip-cleavage`: Skip training the cleavage site model

### Benchmarking Models

To evaluate the models on a benchmark dataset:

```bash
python app/benchmark.py --benchmark-fasta data/signal_peptide_benchmark.fasta --output-dir benchmark_results
```

Options:
- `--benchmark-fasta`: Path to benchmark FASTA file
- `--classification-model`: Path to trained classification model
- `--cleavage-site-model`: Path to trained cleavage site model
- `--output-dir`: Directory to save benchmark results

### Predicting Signal Peptides

#### From a FASTA File

```bash
python app/predict_signal_peptides.py --fasta your_sequences.fasta --model model/signal_peptide_model.keras
```

Options:
- `--fasta`: Path to FASTA file with sequences to predict
- `--model`: Path to trained model

#### For a Single Sequence

You can create a temporary FASTA file with your sequence:

```bash
echo ">Your_Sequence_ID
MLLSVPLLLGLLGLAVASNPVFA" > single_sequence.fasta

python app/predict_signal_peptides.py --fasta single_sequence.fasta
```

Alternatively, you can use the provided `predict_single.py` script:

```bash
python app/predict_single.py --sequence "MLLSVPLLLGLLGLAVASNPVFA" --name "MyProtein"
```

Options:
- `--sequence`: The amino acid sequence to analyze
- `--name`: Name/ID for the sequence (optional)
- `--model`: Path to trained model

### Running the Complete Pipeline

To train models, benchmark them, and make predictions on a new dataset:

```bash
# Train models
python app/train_models.py

# Benchmark models
python app/benchmark.py

# Make predictions
python app/predict_signal_peptides.py --fasta your_sequences.fasta
```

## Model Architecture

The tool uses two separate deep learning models:

1. **Classification Model**: Determines whether a sequence contains a signal peptide
   - Architecture: CNN + Bidirectional LSTM + Dense layers
   
2. **Cleavage Site Model**: Predicts the position of the cleavage site
   - Architecture: CNN + Bidirectional LSTM + Dense layers
   - Only used for sequences predicted to contain signal peptides

## File Structure

```
MOL3022/
├── app/
│   ├── signal_peptide_model.py  # Core model definitions
│   ├── data_preparation.py      # Data processing functions
│   ├── train_models.py          # Script for training models
│   ├── benchmark.py             # Script for model evaluation
│   ├── predict_signal_peptides.py # Script for making predictions
│   └── predict_single.py        # Script for single sequence prediction
├── data/
│   ├── signal_peptide_train.fasta    # Training data
│   └── signal_peptide_benchmark.fasta # Test/benchmark data
├── models/                      # Saved models directory
├── benchmark_results/           # Benchmark results
├── images/                      # Generated visualizations
└── README.md                    # This file
```

## References

- SignalP database: [https://services.healthtech.dtu.dk/service.php?SignalP-5.0](https://services.healthtech.dtu.dk/service.php?SignalP-5.0)
- Nielsen, H., Tsirigos, K. D., Brunak, S., & von Heijne, G. (2019). A Brief History of Protein Sorting Prediction. The Protein Journal, 38(3), 200–216.

## License

[MIT License](LICENSE)


