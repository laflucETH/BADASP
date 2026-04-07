import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import argparse

def plot_length_distribution(fasta_path, output_dir):
    """
    Reads a FASTA file and plots the distribution of sequence lengths.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    lengths = []
    records = []
    
    # Check if file exists and is not empty
    if not os.path.exists(fasta_path) or os.path.getsize(fasta_path) == 0:
        print(f"Error: FASTA file {fasta_path} is missing or empty.")
        return
        
    for record in SeqIO.parse(fasta_path, "fasta"):
        lengths.append(len(record.seq))
        records.append({"id": record.id, "length": len(record.seq)})
        
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        print("No sequences found in " + fasta_path)
        return
        
    print(f"Total sequences: {len(df)}")
    print(f"Mean length: {df['length'].mean():.2f}")
    print(f"Median length: {df['length'].median():.2f}")
    print(f"Min length: {df['length'].min()}")
    print(f"Max length: {df['length'].max()}")
    
    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["length"], bins=50, kde=True, color="skyblue")
    
    plt.title("Sequence Length Distribution for IPR019888")
    plt.xlabel("Sequence Length (Amino Acids)")
    plt.ylabel("Frequency")
    
    # Adding vertical line for median
    plt.axvline(df['length'].median(), color='red', linestyle='dashed', linewidth=2, label=f'Median: {df["length"].median():.0f}')
    plt.legend()
    
    output_path = os.path.join(output_dir, "sequence_length_distribution.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved length distribution plot to {output_path}")
    plt.close()
    
    # Filter sequences (e.g. removing extreme outliers before MSA if needed)
    # For now, just save a basic statistical summary
    summary_path = os.path.join(output_dir, "eda_summary.csv")
    df.describe().to_csv(summary_path)
    print(f"Saved statistical summary to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on sequence data.")
    parser.add_argument("--input", type=str, default="data/raw/IPR019888.fasta", help="Input FASTA file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save plots")
    
    args = parser.parse_args()
    plot_length_distribution(args.input, args.output_dir)
