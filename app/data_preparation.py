#!/usr/bin/env python3
# Enhanced data preparation script with cleavage site detection
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter

def parse_fasta_file(file_path):
    """
    Parse a FASTA file and extract sequences, labels and annotations
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        data_df: DataFrame with sequences and metadata
    """
    records = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('>'):
                # Parse header
                header = lines[i].strip()
                parts = header.split('|')
                
                protein_id = parts[0][1:]  # Remove '>' character
                kingdom = parts[1] if len(parts) > 1 else "UNKNOWN"
                sp_status = parts[2] if len(parts) > 2 else "UNKNOWN"
                org_class = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
                
                # Get sequence
                i += 1
                seq = ""
                annotation = None
                
                # Read sequence line
                if i < len(lines) and not lines[i].startswith('>'):
                    seq = lines[i].strip()
                    i += 1
                    
                    # Read annotation line if it exists
                    if i < len(lines) and not lines[i].startswith('>'):
                        annotation_line = lines[i].strip()
                        # Check if it's an annotation line (contains S and O) or another sequence line
                        if all(c in 'SOI' for c in annotation_line):
                            annotation = annotation_line
                            i += 1
                
                # Determine cleavage site
                cleavage_site = -1
                if annotation and 'S' in annotation and 'O' in annotation:
                    for j in range(len(annotation) - 1):
                        if annotation[j] == 'S' and annotation[j+1] == 'O':
                            cleavage_site = j + 1
                            break
                
                records.append({
                    'protein_id': protein_id,
                    'kingdom': kingdom,
                    'sp_status': sp_status,
                    'org_class': org_class,
                    'sequence': seq,
                    'annotation': annotation,
                    'cleavage_site': cleavage_site,
                    'length': len(seq)
                })
                
                continue
            
            i += 1
    
    # Convert to DataFrame
    data_df = pd.DataFrame(records)
    
    # Convert SP status to binary
    data_df['has_sp'] = data_df['sp_status'].apply(lambda x: 0 if x == 'NO_SP' else 1)
    
    return data_df

def analyze_data(data_df, output_dir='images'):
    """
    Analyze the dataset and generate statistics and visualizations
    
    Args:
        data_df: DataFrame with sequence data
        output_dir: Directory to save analysis results
        
    Returns:
        stats_df: DataFrame with dataset statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    total_seqs = len(data_df)
    sp_count = data_df['has_sp'].sum()
    no_sp_count = total_seqs - sp_count
    
    # Distribution by kingdom
    kingdom_dist = data_df.groupby(['kingdom', 'has_sp']).size().unstack(fill_value=0)
    
    # Length distribution
    avg_length = data_df['length'].mean()
    min_length = data_df['length'].min()
    max_length = data_df['length'].max()
    
    # Amino acid composition analysis
    aa_counts = Counter()
    n_term_aa_counts = Counter()  # First 30 amino acids
    
    for seq in data_df['sequence']:
        aa_counts.update(seq)
        n_term = seq[:30]
        n_term_aa_counts.update(n_term)
    
    # Generate plots
    # 1. Class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='has_sp', data=data_df)
    plt.title('Signal Peptide Distribution')
    plt.xlabel('Has Signal Peptide')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.savefig(os.path.join(output_dir, 'sp_distribution.png'))
    plt.close()
    
    # 2. Sequence length distribution by class
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data_df, x='length', hue='has_sp', bins=30, kde=True)
    plt.title('Sequence Length Distribution by Signal Peptide Status')
    plt.xlabel('Sequence Length')
    plt.savefig(os.path.join(output_dir, 'length_distribution.png'))
    plt.close()
    
    # 3. Kingdom distribution
    if 'kingdom' in data_df.columns:
        plt.figure(figsize=(12, 6))
        kingdom_counts = data_df.groupby(['kingdom', 'has_sp']).size().unstack()
        kingdom_counts.plot(kind='bar', stacked=True)
        plt.title('Signal Peptide Distribution by Kingdom')
        plt.xlabel('Kingdom')
        plt.ylabel('Count')
        plt.legend(['No SP', 'SP'])
        plt.savefig(os.path.join(output_dir, 'kingdom_distribution.png'))
        plt.close()
    
    # 4. Amino acid composition
    plt.figure(figsize=(14, 7))
    amino_acids = sorted(aa_counts.keys())
    frequencies = [aa_counts[aa]/sum(aa_counts.values()) for aa in amino_acids]
    
    plt.bar(range(len(amino_acids)), frequencies)
    plt.xticks(range(len(amino_acids)), amino_acids)
    plt.title('Amino Acid Composition in All Sequences')
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'aa_composition.png'))
    plt.close()
    
    # 5. Analyze cleavage sites if available
    cleavage_sites = data_df[data_df['cleavage_site'] > 0]['cleavage_site']
    if not cleavage_sites.empty:
        plt.figure(figsize=(12, 6))
        sns.histplot(cleavage_sites, bins=range(int(min(cleavage_sites)), int(max(cleavage_sites)) + 2))
        plt.title('Distribution of Cleavage Site Positions')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'cleavage_site_distribution.png'))
        plt.close()
        
        # Calculate statistics on cleavage sites
        avg_cleavage_pos = cleavage_sites.mean()
        min_cleavage_pos = cleavage_sites.min()
        max_cleavage_pos = cleavage_sites.max()
        
        # Analyze amino acid distribution around cleavage sites
        # Get sequences with cleavage sites
        cleavage_seqs = data_df[data_df['cleavage_site'] > 0]
        
        # Extract amino acids at positions -3 to +3 around cleavage site
        cleavage_window = []
        
        for _, row in cleavage_seqs.iterrows():
            site = row['cleavage_site']
            seq = row['sequence']
            
            # Ensure we have enough sequence before and after
            if site >= 3 and site + 3 <= len(seq):
                # Extract window around cleavage site (-3 to +3)
                window = seq[site-3:site+3]
                cleavage_window.append({
                    'position': list(range(-3, 3)),
                    'amino_acid': list(window),
                    'sequence_id': row['protein_id']
                })
        
        if cleavage_window:
            # Convert to DataFrame for easier manipulation
            window_df = pd.DataFrame(cleavage_window)
            window_df = window_df.explode(['position', 'amino_acid'])
            
            # Plot amino acid frequency at each position
            plt.figure(figsize=(14, 8))
            for pos in range(-3, 3):
                pos_aa = window_df[window_df['position'] == pos]['amino_acid']
                pos_counts = pos_aa.value_counts() / len(pos_aa)
                
                plt.subplot(2, 3, pos+4)  # Adjust subplot position
                plt.bar(pos_counts.index, pos_counts.values)
                plt.title(f'Position {pos} relative to cleavage site')
                plt.xlabel('Amino Acid')
                plt.ylabel('Frequency')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cleavage_site_aa_distribution.png'))
            plt.close()
    
    # Compute hydrophobicity for N-terminal regions
    hydrophobicity_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 
        'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 
        'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    def calc_hydrophobicity(seq):
        return sum(hydrophobicity_scale.get(aa, 0) for aa in seq) / len(seq) if seq else 0
    
    data_df['n_term'] = data_df['sequence'].apply(lambda x: x[:30] if len(x) >= 30 else x)
    data_df['hydrophobicity'] = data_df['n_term'].apply(calc_hydrophobicity)
    
    # 6. Hydrophobicity distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data_df, x='hydrophobicity', hue='has_sp', bins=30, kde=True)
    plt.title('N-terminal Hydrophobicity Distribution by Signal Peptide Status')
    plt.xlabel('Average Hydrophobicity')
    plt.savefig(os.path.join(output_dir, 'hydrophobicity_distribution.png'))
    plt.close()
    
    # Compile statistics
    stats = {
        'Total sequences': total_seqs,
        'With signal peptide': sp_count,
        'Without signal peptide': no_sp_count,
        'SP percentage': (sp_count / total_seqs) * 100,
        'Average sequence length': avg_length,
        'Min sequence length': min_length,
        'Max sequence length': max_length
    }
    
    # Add cleavage site statistics if available
    if not cleavage_sites.empty:
        stats.update({
            'Sequences with cleavage sites': len(cleavage_sites),
            'Average cleavage site position': avg_cleavage_pos,
            'Min cleavage site position': min_cleavage_pos,
            'Max cleavage site position': max_cleavage_pos
        })
    
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    
    return stats_df

def analyze_fasta_files(train_fasta, benchmark_fasta, output_dir='images'):
    """
    Analyze existing training and benchmark FASTA files
    
    Args:
        train_fasta: Path to training FASTA file
        benchmark_fasta: Path to benchmark FASTA file
        output_dir: Directory to save analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse training FASTA file
    print("Parsing training FASTA file...")
    train_df = parse_fasta_file(train_fasta)
    
    # Parse benchmark FASTA file
    print("Parsing benchmark FASTA file...")
    benchmark_df = parse_fasta_file(benchmark_fasta)
    
    # Combine for overall statistics
    print("Analyzing combined data...")
    combined_df = pd.concat([train_df, benchmark_df])
    combined_stats = analyze_data(combined_df, output_dir)
    
    # Separate analysis for training set
    print("\nAnalyzing training set...")
    train_stats = analyze_data(train_df, output_dir)
    
    # Separate analysis for benchmark set
    print("\nAnalyzing benchmark set...")
    benchmark_stats = analyze_data(benchmark_df, output_dir)
    
    # Print statistics
    print("\nOverall Dataset Statistics:")
    print(combined_stats)
    
    print("\nTraining Set Statistics:")
    print(train_stats)
    
    print("\nBenchmark Set Statistics:")
    print(benchmark_stats)
    
    # Compare class distributions
    train_class_dist = train_df['has_sp'].value_counts(normalize=True)
    benchmark_class_dist = benchmark_df['has_sp'].value_counts(normalize=True)
    
    print("\nClass Distribution Comparison:")
    print("Training set: SP = {:.2f}%, No SP = {:.2f}%".format(
        train_class_dist.get(1, 0) * 100, 
        train_class_dist.get(0, 0) * 100
    ))
    print("Benchmark set: SP = {:.2f}%, No SP = {:.2f}%".format(
        benchmark_class_dist.get(1, 0) * 100, 
        benchmark_class_dist.get(0, 0) * 100
    ))
    
    # Compare cleavage site distributions if available
    train_cleavage = train_df[train_df['cleavage_site'] > 0]['cleavage_site']
    benchmark_cleavage = benchmark_df[benchmark_df['cleavage_site'] > 0]['cleavage_site']
    
    if not train_cleavage.empty and not benchmark_cleavage.empty:
        plt.figure(figsize=(12, 6))
        plt.hist([train_cleavage, benchmark_cleavage], bins=20, 
                 label=['Training', 'Benchmark'], alpha=0.7)
        plt.title('Cleavage Site Position Comparison')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'cleavage_comparison.png'))
        plt.close()
        
        print("\nCleavage Site Comparison:")
        print("Training set: Avg position = {:.2f}, Count = {}".format(
            train_cleavage.mean(), len(train_cleavage)
        ))
        print("Benchmark set: Avg position = {:.2f}, Count = {}".format(
            benchmark_cleavage.mean(), len(benchmark_cleavage)
        ))
    
    print("\nAnalysis complete!")
    
    return train_df, benchmark_df, combined_stats

def main():
    """
    Main function to analyze existing FASTA files for signal peptide prediction
    """
    # Create directories
    os.makedirs('images', exist_ok=True)
    
    # Paths to your existing FASTA files
    train_fasta = 'data/signal_peptide_train.fasta'
    benchmark_fasta = 'data/signal_peptide_benchmark.fasta'
    output_dir = 'images'
    
    # Analyze the existing FASTA files
    train_df, benchmark_df, stats = analyze_fasta_files(train_fasta, benchmark_fasta, output_dir)
    
    print("\nData analysis complete!")

if __name__ == '__main__':
    main()