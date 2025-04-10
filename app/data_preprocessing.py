#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def parse_fasta_file(file_path):
    """
    Parse a FASTA file with signal peptide annotations

    Format:
    >Uniprot_AC|Kingdom|Type|Partition No
    amino-acid sequence
    annotation [S: Sec/SPI, T: Tat/SPI or Tat/SPII, L: Sec/SPII, P: Sec/SPIII, I: cytoplasm, M: transmembrane, O: extracellular]

    Returns:
        DataFrame with parsed data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    records = []
    with open(file_path, "r") as f:
        current_id = None
        current_seq = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Process previous record if exists
                if current_id is not None and current_seq is not None:
                    records.append({"header": current_id, "sequence": current_seq})

                # Start new record
                current_id = line[1:]  # Remove '>' prefix
                current_seq = None
            else:
                if current_seq is None:
                    # This is the amino acid sequence
                    current_seq = line
                else:
                    # This is the annotation line
                    annotation = line

                    # Process the complete record
                    header_parts = current_id.split("|")
                    protein_id = (
                        header_parts[0] if len(header_parts) > 0 else current_id
                    )

                    # Extract kingdom from header
                    kingdom = header_parts[1] if len(header_parts) > 1 else "unknown"

                    # Extract type from header
                    sp_type = header_parts[2] if len(header_parts) > 2 else "unknown"

                    # Determine if sequence has signal peptide and its class
                    has_sp = 0
                    sp_class = "I"  # Default to cytoplasm (no signal peptide)
                    cleavage_site = 0

                    # Extract signal peptide info from annotation
                    if len(annotation) > 0:
                        # Look for signal peptide characters
                        sp_chars = ['S', 'T', 'L', 'P']
                        has_sp = 0
                        sp_class = 'I'  # Default to cytoplasm (no signal peptide)
                        cleavage_site = 0
                        
                        # Check if any signal peptide characters exist
                        if any(char in sp_chars for char in annotation):
                            # Find the last occurrence of any signal peptide character
                            for i in range(len(annotation)):
                                if annotation[i] in sp_chars:
                                    has_sp = 1
                                    sp_class = annotation[i]
                                    cleavage_site = i + 1  # +1 because positions are 1-indexed
                                # Break after the last signal peptide character (when next char is different)
                                elif has_sp == 1:
                                    break

                    # Add record with all parsed information
                    records.append(
                        {
                            "protein_id": protein_id,
                            "kingdom": kingdom,
                            "type": sp_type,
                            "sequence": current_seq,
                            "annotation": annotation,
                            "has_sp": has_sp,
                            "sp_class": sp_class,
                            "cleavage_site": cleavage_site,
                        }
                    )

                    # Reset for next record
                    current_id = None
                    current_seq = None

        # Process the last record if exists
        if current_id is not None and current_seq is not None:
            records.append({"header": current_id, "sequence": current_seq})

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Validate cleavage sites
    sp_mask = df["has_sp"] == 1
    zero_cs = (df["cleavage_site"] == 0) & sp_mask
    if zero_cs.any():
        print(
            f"WARNING: {zero_cs.sum()} sequences with signal peptides have cleavage site = 0"
        )

    return df


def encode_amino_acid(aa):
    """One-hot encode a single amino acid"""
    aa_dict = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
    }

    # Return zeros for unknown amino acids
    if aa not in aa_dict:
        return np.zeros(20)

    # One-hot encode the amino acid
    encoding = np.zeros(20)
    encoding[aa_dict[aa]] = 1
    return encoding


def encode_sequences(sequences, max_length=100):
    """
    Encode amino acid sequences as one-hot encoded arrays

    Args:
        sequences: List of amino acid sequences
        max_length: Maximum sequence length to consider

    Returns:
        Encoded sequences as numpy array
    """
    n_samples = len(sequences)
    n_features = 20  # 20 amino acids

    # Initialize output array
    X = np.zeros((n_samples, max_length, n_features), dtype=np.float32)

    # Encode each sequence
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_length]):
            X[i, j] = encode_amino_acid(aa)

    return X


def encode_kingdom(kingdom_list):
    """
    Encode kingdom information as one-hot vectors

    Args:
        kingdom_list: List of kingdom strings

    Returns:
        One-hot encoded kingdom data
    """
    # Define mapping of kingdoms to indices
    kingdom_dict = {
        "ARCHAEA": 0,
        "EUKARYA": 1,
        "NEGATIVE": 2,
        "POSITIVE": 3,
        "unknown": 4,
    }

    # Initialize output array
    n_samples = len(kingdom_list)
    n_kingdoms = len(kingdom_dict)
    kingdom_encoded = np.zeros((n_samples, n_kingdoms), dtype=np.float32)

    # Encode each kingdom
    for i, kingdom in enumerate(kingdom_list):
        if kingdom.upper() in kingdom_dict:
            kingdom_encoded[i, kingdom_dict[kingdom.upper()]] = 1
        else:
            kingdom_encoded[i, kingdom_dict["unknown"]] = 1

    return kingdom_encoded


def encode_sp_class(sp_class_list):
    """
    Encode signal peptide class information as one-hot vectors

    Args:
        sp_class_list: List of signal peptide class strings

    Returns:
        One-hot encoded SP class data
    """
    # Define mapping of SP classes to indices
    sp_class_dict = {
        "S": 0,  # Sec/SPI signal peptide
        "T": 1,  # Tat/SPI or Tat/SPII signal peptide
        "L": 2,  # Sec/SPII signal peptide
        "P": 3,  # Sec/SPIII signal peptide
        "I": 4,  # cytoplasm
        "M": 5,  # transmembrane
        "O": 6,  # extracellular
        "unknown": 7,
    }

    # Initialize output array
    n_samples = len(sp_class_list)
    n_classes = len(sp_class_dict)
    sp_class_encoded = np.zeros((n_samples, n_classes), dtype=np.float32)

    # Encode each SP class
    for i, sp_class in enumerate(sp_class_list):
        if sp_class in sp_class_dict:
            sp_class_encoded[i, sp_class_dict[sp_class]] = 1
        else:
            sp_class_encoded[i, sp_class_dict["unknown"]] = 1

    return sp_class_encoded



def prepare_data_for_training(data_df, max_length=100):
    """
    Prepare data for training the model

    Args:
        data_df: DataFrame with protein sequences and metadata
        max_length: Maximum sequence length

    Returns:
        Dictionary with encoded input data and targets
    """
    # Encode sequences
    sequences = data_df["sequence"].tolist()
    X_seq = encode_sequences(sequences, max_length=max_length)

    # Encode kingdom information
    X_kingdom = encode_kingdom(data_df["kingdom"].tolist())

    # Classification target: has signal peptide
    y_has_sp = np.array(data_df["has_sp"].values, dtype=np.float32)

    # SP class prediction target
    y_sp_class = encode_sp_class(data_df["sp_class"].tolist())

    # Cleavage site position target
    y_cleavage = np.array(data_df["cleavage_site"].values, dtype=np.float32)

    # If all cleavage sites are 1, it might indicate a problem with the data
    # For sequences with signal peptides, generate realistic cleavage sites if needed
    sp_mask = data_df["has_sp"] == 1


    # Normalize cleavage site positions for training
    y_cleavage_normalized = normalize_cleavage_sites(y_cleavage, max_length)

    # Create masks for sequences with signal peptides
    sp_mask = data_df["has_sp"] == 1

    print(f"Data preparation summary:")
    print(f"  Total sequences: {len(data_df)}")
    print(f"  Sequences with SP: {sp_mask.sum()}")
    print(f"  Sequences without SP: {len(data_df) - sp_mask.sum()}")
    print(f"  Mean cleavage site (for SP sequences): {y_cleavage[sp_mask].mean():.2f}")
    print(f"  Min cleavage site (for SP sequences): {y_cleavage[sp_mask].min()}")
    print(f"  Max cleavage site (for SP sequences): {y_cleavage[sp_mask].max()}")

    return {
        "X_seq": X_seq,
        "X_kingdom": X_kingdom,
        "y_has_sp": y_has_sp,
        "y_sp_class": y_sp_class,
        "y_cleavage": y_cleavage,
        "y_cleavage_normalized": y_cleavage_normalized,
        "sp_mask": sp_mask,
    }


def normalize_cleavage_sites(cleavage_sites, max_length=100):
    """
    Normalize cleavage site positions to be between 0 and 1
    
    Args:
        cleavage_sites: Array of cleavage site positions
        max_length: Maximum sequence length
        
    Returns:
        Normalized cleavage site positions
    """
    # Clip values to be within valid range
    clipped = np.clip(cleavage_sites, 0, max_length)
    
    # Normalize to [0, 1]
    normalized = clipped / max_length
    
    return normalized
