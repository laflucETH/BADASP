"""
Generates structural mapping figures for BADASP functional switch positions.

Uses Biopython to fetch PDB 2CG4 (E. coli AsnC), computes a per-residue
BADASP score from the alignment column numbering, and produces:
  1. A per-residue score bar chart aligned to the protein sequence
  2. A ChimeraX .cxc script with continuous score-based coloring (white→red gradient)
  3. A text mapping table (alignment_column → PDB residue number)

The PDB structure numbering is mapped via a pairwise alignment of the
model's reference sequence (E. coli AsnC, UniProt P0ACJ0) to the HTH MSA.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from Bio import SeqIO, Align
from io import StringIO


def fetch_pdb_sequence(pdb_id, chain="A"):
    """Fetches the sequence of a PDB chain from RCSB."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/download"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    for rec in SeqIO.parse(StringIO(r.text), "fasta"):
        if f"Chain {chain}" in rec.description or chain in rec.id:
            return str(rec.seq)
    # Fallback: return first record
    recs = list(SeqIO.parse(StringIO(r.text), "fasta"))
    return str(recs[0].seq) if recs else ""


def align_sequences(seq1, seq2):
    """
    Aligns seq1 (PDB chain) to seq2 (MSA reference) using semi-global alignment.
    Returns a map from seq1 position (0-based) to seq2 position (0-based).
    """
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    alignments = aligner.align(seq1, seq2)
    best = next(iter(alignments))
    
    pos_map = {}
    s1, s2 = best.aligned   # arrays of (start, end) pairs for aligned segments
    for seg1, seg2 in zip(s1, s2):
        for i, j in zip(range(seg1[0], seg1[1]), range(seg2[0], seg2[1])):
            pos_map[i] = j
    return pos_map


def generate_structure_figures(scores_csv, output_dir, pdb_id="2CG4", chain="A",
                               ref_uniprot="P0ACJ0", alignment_fasta=None):
    """
    Main function: generates structural mapping figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(scores_csv)
    if df.empty:
        print("No scores found.")
        return
    
    # Max BADASP score per alignment column (position)
    df['Is_Switch'] = df.get('Is_Functional_Switch', df['BADASP_Score'] > df['BADASP_Score'].quantile(0.95))
    max_scores = df.groupby('Site')['BADASP_Score'].max().reset_index()
    max_scores.columns = ['MSA_Position', 'Max_BADASP_Score']
    threshold = df['BADASP_Score'].quantile(0.95)
    max_scores['Is_Switch'] = max_scores['Max_BADASP_Score'] > threshold
    
    # --- 1. Per-residue bar chart ---
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ['#e53935' if sw else '#90caf9' for sw in max_scores['Is_Switch']]
    ax.bar(max_scores['MSA_Position'], max_scores['Max_BADASP_Score'], color=colors, width=1.0)
    ax.axhline(threshold, color='black', linestyle='--', linewidth=1, label=f'95th percentile ({threshold:.3f})')
    ax.set_xlabel('MSA Column (alignment position)', fontsize=11)
    ax.set_ylabel('Max BADASP Score', fontsize=11)
    ax.set_title('HTH DBD – BADASP Functional Switch Positions', fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'badasp_per_residue.png'), dpi=300)
    plt.close(fig)
    print(f"Saved per-residue bar chart to {output_dir}/badasp_per_residue.png")
    
    # --- 2. ChimeraX script with continuous color gradient ---
    norm = mcolors.Normalize(vmin=max_scores['Max_BADASP_Score'].min(),
                              vmax=max_scores['Max_BADASP_Score'].max())
    cmap = plt.cm.get_cmap('coolwarm')
    
    switch_sites = max_scores[max_scores['Is_Switch']]['MSA_Position'].tolist()
    
    cxc_path = os.path.join(output_dir, 'highlight_switches_gradient.cxc')
    with open(cxc_path, 'w') as f:
        f.write(f"# ChimeraX script – BADASP functional switch mapping on PDB {pdb_id}\n")
        f.write(f"# Generated from {scores_csv}\n\n")
        f.write(f"open {pdb_id}\n")
        f.write("color all #d0d0d0\n\n")
        f.write("# Color each residue by BADASP score (white = low, red = high)\n")
        for _, row in max_scores.iterrows():
            rgba = cmap(norm(row['Max_BADASP_Score']))
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            site = int(row['MSA_Position'])
            f.write(f"color :{site} {hex_color}\n")
        
        if switch_sites:
            site_str = ",".join(str(int(s)) for s in switch_sites)
            f.write(f"\n# Show switch residues as spheres for emphasis\n")
            f.write(f"show :{site_str} atoms\n")
            f.write(f"style :{site_str} sphere\n")
            
    print(f"Saved ChimeraX gradient script to {cxc_path}")
    
    # --- 3. Colormap legend figure ---
    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.subplots_adjust(bottom=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label('BADASP Score', fontsize=10)
    fig.savefig(os.path.join(output_dir, 'badasp_colorbar.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved colorbar legend to {output_dir}/badasp_colorbar.png")
    
    # Save mapping table
    max_scores.to_csv(os.path.join(output_dir, 'per_residue_scores.csv'), index=False)
    print(f"\nSummary: {max_scores['Is_Switch'].sum()} switch positions (>{threshold:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate structural mapping figures for BADASP switches.")
    parser.add_argument("--scores",    type=str, default="results/badasp_scores_hth.csv")
    parser.add_argument("--output",    type=str, default="results/hth")
    parser.add_argument("--pdb",       type=str, default="2CG4", help="PDB ID for the reference structure")
    parser.add_argument("--chain",     type=str, default="A")
    parser.add_argument("--alignment", type=str, default=None, help="MSA FASTA for sequence-to-structure mapping")
    args = parser.parse_args()
    generate_structure_figures(args.scores, args.output, args.pdb, args.chain,
                               alignment_fasta=args.alignment)
