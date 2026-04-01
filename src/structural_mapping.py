"""
Generates structural mapping figures for BADASP functional switch positions.

Uses Biopython to fetch PDB 2CG4 (E. coli AsnC), maps the alignment column
numbering to the true PDB residue numbering using the E. coli AsnC sequence
(P0ACJ0) from the MSA, and computes the switch count per residue.

Produces:
  1. A per-residue switch count bar chart
  2. A ChimeraX .cxc script that sets a custom attribute 'switch_count' on the
     structure and applies a white->red color gradient, replicating the visual
     style of Beltrao/Bradley.
"""

import os
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
    recs = list(SeqIO.parse(StringIO(r.text), "fasta"))
    return str(recs[0].seq) if recs else ""


def align_sequences(seq1, seq2):
    """
    Aligns seq1 (PDB chain) to seq2 (MSA reference sequence).
    Returns a map from seq2 position (0-based) to seq1 position (0-based).
    """
    from Bio import pairwise2
    # Local alignment with match=2, mismatch=-1, open=-10, extend=-1
    alns = pairwise2.align.localms(seq1, seq2, 2, -1, -10, -1)
    best = alns[0]
    
    pos_map = {}
    s1, s2, _, _, _ = best
    idx1, idx2 = 0, 0
    for char1, char2 in zip(s1, s2):
        if char1 != '-' and char2 != '-':
            pos_map[idx2] = idx1
        if char1 != '-': idx1 += 1
        if char2 != '-': idx2 += 1
    return pos_map


def generate_structure_figures(scores_csv, output_dir, pdb_id="2CG4", chain="A",
                               ref_uniprot="P0ACJ0", alignment_fasta=None):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(scores_csv)
    if df.empty:
        print("No scores found.")
        return
    
    # Identify functional switches (>95th percentile)
    threshold = df['BADASP_Score'].quantile(0.95)
    df['Is_Switch'] = df['BADASP_Score'] > threshold
    
    # As established by Beltrao/Bradley, count the absolute number of events 
    # exceeding the global 95th percentile threshold across all branches
    switches_df = df[df['Is_Switch']]
    switch_counts = switches_df.groupby('Site').size().reset_index(name='Switch_Count')
    
    # Map MSA Columns -> PDB Residue Numbers
    # Default is 1:1 if mapping FASTA is not provided or ref sequence is missing
    site_to_pdb = {}
    if alignment_fasta and os.path.exists(alignment_fasta):
        msa_records = list(SeqIO.parse(alignment_fasta, "fasta"))
        pdb_seq = fetch_pdb_sequence(pdb_id, chain)
        
        # Find best matching sequence in MSA to PDB sequence
        print(f"Aligning PDB {pdb_id} chain {chain} to MSA to find best reference mapping...")
        best_score = -1
        ref_seq_record = None
        
        from Bio import pairwise2
        
        for rec in msa_records:
            raw = str(rec.seq).replace("-", "")
            # Return max score only
            score = pairwise2.align.localms(pdb_seq, raw, 2, -1, -5, -1, score_only=True)
            if score > best_score:
                best_score = score
                ref_seq_record = rec
        
        if ref_seq_record:
            print(f"Found best match: {ref_seq_record.id} (Score: {best_score}). Mapping to PDB.")
            
            # Extract the raw sequence (no gaps) from the MSA reference
            msa_raw_seq = str(ref_seq_record.seq).replace("-", "")
            
            # Map from ungapped MSA sequence to ungapped PDB sequence
            ungapped_msa_to_pdb = align_sequences(pdb_seq, msa_raw_seq)
            
            # Map from MSA column (col_idx) to ungapped MSA sequence index
            col_to_ungapped = {}
            ungapped_idx = 0
            for col_idx, char in enumerate(str(ref_seq_record.seq)):
                if char != "-":
                    col_to_ungapped[col_idx] = ungapped_idx
                    ungapped_idx += 1
            
            # Combine mappings: MSA Column -> PDB Residue Index
            # PDB indices are usually 1-based, so we add 1 to the 0-based index.
            for col_idx in range(len(ref_seq_record.seq)):
                if col_idx in col_to_ungapped:
                    u_idx = col_to_ungapped[col_idx]
                    if u_idx in ungapped_msa_to_pdb:
                        site_to_pdb[col_idx + 1] = ungapped_msa_to_pdb[u_idx] + 1
        else:
            print("Warning: Could not align to MSA. Using 1:1 mapping.")
            site_to_pdb = {site: site for site in df['Site'].unique()}
    else:
        site_to_pdb = {site: site for site in df['Site'].unique()}

    # Merge mapped PDB numbers into the switch_counts dataframe
    switch_counts['PDB_Residue'] = switch_counts['Site'].map(site_to_pdb)
    # Filter out sites that didn't map to the PDB
    mapped_counts = switch_counts.dropna(subset=['PDB_Residue']).copy()
    mapped_counts['PDB_Residue'] = mapped_counts['PDB_Residue'].astype(int)
    
    max_count = mapped_counts['Switch_Count'].max() if not mapped_counts.empty else 1
    
    # --- 1. Per-residue Switch Count Bar Chart ---
    fig, ax = plt.subplots(figsize=(14, 4))
    # We plot against PDB Residue for clearer biological context
    ax.bar(mapped_counts['PDB_Residue'], mapped_counts['Switch_Count'], color='#e53935', width=1.0)
    ax.set_xlabel(f'PDB Residue Number (based on {pdb_id})', fontsize=11)
    ax.set_ylabel('Number of Switch Events', fontsize=11)
    ax.set_title('HTH DBD – Frequency of Functional Switch Events per Residue', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'badasp_switch_counts.png'), dpi=300)
    plt.close(fig)
    print(f"Saved bar chart to {output_dir}/badasp_switch_counts.png")
    
    # --- 2. ChimeraX publication-style script using custom attribute ---
    cxc_path = os.path.join(output_dir, 'mapped_switches_publication.cxc')
    with open(cxc_path, 'w') as f:
        f.write(f"# ChimeraX script – BADASP functional switch counts mapped on PDB {pdb_id}\n")
        f.write(f"open {pdb_id}\n")
        f.write(f"hide all\n")
        f.write("# Set publication-ready white background\n")
        f.write("set bg_color white\n\n")
        
        f.write("# Initialize switch_count attribute to 0 for all residues\n")
        f.write(f"setattr :* residues switch_count 0 create true\n\n")
        
        f.write("# Set switch_count attribute for observed functional switches\n")
        for _, row in mapped_counts.iterrows():
            pdb_res = int(row['PDB_Residue'])
            count = int(row['Switch_Count'])
            f.write(f"setattr :{pdb_res} residues switch_count {count}\n")
            
        f.write("\n# Apply color gradient (White -> Light Orange -> Firebrick Red) representing switch continuous intensity\n")
        f.write(f"color byattribute switch_count palette white:orange:red range 0,{float(max_count):.2f}\n\n")
        
    mapped_counts.to_csv(os.path.join(output_dir, 'per_residue_scores.csv'), index=False)
    print(f"Saved ChimeraX publication script and mapped_counts CSV to {output_dir}")


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
