import os
import pandas as pd
import numpy as np
from Bio import Phylo
from Bio import AlignIO
from Bio.Align import substitution_matrices
import argparse
import sys

def parse_iqtree_state(state_file):
    """
    Parses the IQ-TREE .state file which contains ancestral probabilities.
    """
    if not os.path.exists(state_file):
        print(f"Error: .state file {state_file} not found.")
        sys.exit(1)
        
    ancestral_probs = {}
    with open(state_file, 'r') as f:
        lines = f.readlines()
        
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Node\tSite"):
            start_idx = i
            break
            
    header = lines[start_idx].strip().split('\t')
    aa_list = header[3:]
    aa_chars = [aa.replace('p_', '') for aa in aa_list]
    
    for line in lines[start_idx+1:]:
        if not line.strip(): continue
        parts = line.strip().split('\t')
        node = parts[0]
        site = int(parts[1]) - 1
        
        if node not in ancestral_probs:
            ancestral_probs[node] = {}
            
        probs = np.array([float(p) for p in parts[3:]])
        ancestral_probs[node][site] = probs
        
    return ancestral_probs, aa_chars

def calculate_conservation(alignment, leaves):
    """
    Calculates the Shannon entropy (conservation) at each site for a given set of leaves.
    """
    leaf_names = set(leaves)
    seqs = [rec.seq for rec in alignment if rec.id in leaf_names]
    
    if not seqs:
        return np.zeros(alignment.get_alignment_length())
        
    length = len(seqs[0])
    conservation = np.zeros(length)
    
    for i in range(length):
        col = [seq[i] for seq in seqs if seq[i] != '-']
        if not col:
            conservation[i] = 0
            continue
        counts = pd.Series(col).value_counts(normalize=True)
        entropy = -sum(counts * np.log2(counts + 1e-9))
        
        max_entropy = np.log2(20)
        conservation[i] = max(0.0, 1.0 - (entropy / max_entropy))
        
    return conservation

def run_badasp(tree_file, alignment_file, state_file, output_csv):
    """
    Executes the BADASP analytical algorithm over all internal nodes using BLOSUM62.
    Formula: s = RC * AC * p(AC)
    """
    tree = Phylo.read(tree_file, "newick")
    tree.root_at_midpoint()
    
    alignment = AlignIO.read(alignment_file, "fasta")
    ancestral_probs, aa_chars = parse_iqtree_state(state_file)
    
    # Load BLOSUM62
    blosum = substitution_matrices.load("BLOSUM62")
    blosum_matrix = np.zeros((len(aa_chars), len(aa_chars)))
    
    for i, a1 in enumerate(aa_chars):
        for j, a2 in enumerate(aa_chars):
            # Biopython's matrix keys might be a tuple, order doesn't matter for symmetric matrix
            pair = (a1, a2)
            if pair in blosum:
                blosum_matrix[i, j] = blosum[pair]
            elif (a2, a1) in blosum:
                blosum_matrix[i, j] = blosum[(a2, a1)]
            else:
                blosum_matrix[i, j] = -4 # Default penalty for unknown pairs
                
    results = []
    
    for clade in tree.get_nonterminals():
        if not clade.name:
            continue
            
        children = clade.clades
        if len(children) != 2:
            continue
            
        child1, child2 = children[0], children[1]
        if child1.name not in ancestral_probs or child2.name not in ancestral_probs:
            continue
            
        leaves1 = [leaf.name for leaf in child1.get_terminals()]
        leaves2 = [leaf.name for leaf in child2.get_terminals()]
        
        if len(leaves1) < 3 or len(leaves2) < 3:
            continue
            
        cons1 = calculate_conservation(alignment, leaves1)
        cons2 = calculate_conservation(alignment, leaves2)
        node_name = clade.name
        
        for site in range(alignment.get_alignment_length()):
            if site not in ancestral_probs[child1.name] or site not in ancestral_probs[child2.name]:
                continue
                
            p1 = ancestral_probs[child1.name][site]
            p2 = ancestral_probs[child2.name][site]
            
            # AC: Expected substitution score using BLOSUM62
            # expected_score = p1^T * BLOSUM * p2
            expected_score = np.dot(p1, np.dot(blosum_matrix, p2))
            
            # We want to identify radical substitutions (negative BLOSUM scores)
            # AC is defined such that higher is more radical
            AC = max(0.0, -expected_score)
            
            # RC: Recent Conservation within the descendent clades
            RC = cons1[site] * cons2[site]
            
            # p(AC): Ancestral prediction confidence (product of the max probability at the site)
            p_AC = np.max(p1) * np.max(p2)
            
            # Total score
            score = RC * AC * p_AC
            
            if score > 0:
                results.append({
                    "Node": node_name,
                    "Child1_Size": len(leaves1),
                    "Child2_Size": len(leaves2),
                    "Site": site + 1,
                    "RC": round(RC, 3),
                    "AC": round(AC, 3),
                    "p_AC": round(p_AC, 3),
                    "BADASP_Score": round(score, 4)
                })
                
    df = pd.DataFrame(results)
    if not df.empty:
        # Beltrao & Bradley threshold: >95th percentile
        threshold_95 = df['BADASP_Score'].quantile(0.95)
        df['Is_Functional_Switch'] = df['BADASP_Score'] > threshold_95
        
        df = df.sort_values("BADASP_Score", ascending=False)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"BADASP analysis complete. Top 5% threshold is {threshold_95:.4f}.")
        print(f"Identified {df['Is_Functional_Switch'].sum()} 'switch' positions across the tree.")
    else:
        print("No significant functional determining sites found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BADASP scores.")
    parser.add_argument("--tree", type=str, default="data/interim/asr_output/IPR019888_aligned.fasta.treefile", help="Input Tree")
    parser.add_argument("--alignment", type=str, default="data/interim/IPR019888_aligned.fasta", help="Input Alignment")
    parser.add_argument("--state", type=str, default="data/interim/asr_output/IPR019888_aligned.fasta.state", help="IQ-TREE .state file")
    parser.add_argument("--output", type=str, default="results/badasp_scores.csv", help="Output CSV file")
    
    args = parser.parse_args()
    run_badasp(args.tree, args.alignment, args.state, args.output)
