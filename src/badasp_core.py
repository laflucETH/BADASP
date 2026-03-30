import os
import pandas as pd
import numpy as np
from Bio import Phylo
from Bio import AlignIO
import argparse
import sys

def parse_iqtree_state(state_file):
    """
    Parses the IQ-TREE .state file which contains ancestral probabilities.
    Returns a dictionary: {node_name: {site_index: {amino_acid: probability}}}
    """
    if not os.path.exists(state_file):
        print(f"Error: .state file {state_file} not found.")
        sys.exit(1)
        
    ancestral_probs = {}
    with open(state_file, 'r') as f:
        lines = f.readlines()
        
    # Skip header lines until "Node"
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Node\tSite"):
            start_idx = i
            break
            
    header = lines[start_idx].strip().split('\t')
    aa_list = header[3:] # Usually p_A, p_R, p_N, etc.
    aa_chars = [aa.replace('p_', '') for aa in aa_list]
    
    for line in lines[start_idx+1:]:
        if not line.strip(): continue
        parts = line.strip().split('\t')
        node = parts[0]
        site = int(parts[1]) - 1 # 0-indexed
        
        if node not in ancestral_probs:
            ancestral_probs[node] = {}
            
        probs = np.array([float(p) for p in parts[3:]])
        ancestral_probs[node][site] = probs
        
    return ancestral_probs, aa_chars

def calculate_conservation(alignment, leaves):
    """
    Calculates the Shannon entropy (conservation) at each site for a given set of leaves.
    Returns an array of conservation scores.
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
        
        # Max entropy for 20 amino acids is ~4.32. Map to 0-1 where 1 is highly conserved.
        max_entropy = np.log2(20)
        conservation[i] = max(0, 1 - (entropy / max_entropy))
        
    return conservation

def run_badasp(tree_file, alignment_file, state_file, output_csv):
    """
    Executes the BADASP analytical algorithm over all internal nodes of the tree.
    """
    tree = Phylo.read(tree_file, "newick")
    tree.root_at_midpoint() # Root the tree to define directionality
    
    alignment = AlignIO.read(alignment_file, "fasta")
    ancestral_probs, aa_chars = parse_iqtree_state(state_file)
    
    results = []
    
    # Traverse internal nodes
    for clade in tree.get_nonterminals():
        if not clade.name:
            continue # Skip unnamed nodes if any
            
        children = clade.clades
        if len(children) != 2:
            continue # Focus on bifurcations
            
        child1, child2 = children[0], children[1]
        
        # We need ancestral states for the two child clades
        if child1.name not in ancestral_probs or child2.name not in ancestral_probs:
            continue
            
        leaves1 = [leaf.name for leaf in child1.get_terminals()]
        leaves2 = [leaf.name for leaf in child2.get_terminals()]
        
        if len(leaves1) < 3 or len(leaves2) < 3:
            continue # Require minimum clade size for conservation stats
            
        cons1 = calculate_conservation(alignment, leaves1)
        cons2 = calculate_conservation(alignment, leaves2)
        
        node_name = clade.name
        
        for site in range(alignment.get_alignment_length()):
            if site not in ancestral_probs[child1.name] or site not in ancestral_probs[child2.name]:
                continue
                
            p1 = ancestral_probs[child1.name][site]
            p2 = ancestral_probs[child2.name][site]
            
            # Euclidean distance between probability vectors
            dist = np.linalg.norm(p1 - p2) / np.sqrt(2) # Normalize to 0-1
            
            # BADASP Score: High if conserved within subfamilies but differ between them
            score = cons1[site] * cons2[site] * dist
            
            if score > 0.1: # Only record somewhat significant scores
                results.append({
                    "Node": node_name,
                    "Child1_Size": len(leaves1),
                    "Child2_Size": len(leaves2),
                    "Site": site + 1,
                    "Cons1": round(cons1[site], 3),
                    "Cons2": round(cons2[site], 3),
                    "Distance": round(dist, 3),
                    "BADASP_Score": round(score, 4)
                })
                
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("BADASP_Score", ascending=False)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"BADASP analysis complete. Found {len(df)} candidate sites. Saved to {output_csv}")
    else:
        print("No significant functional determining sites found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BADASP scores.")
    parser.add_argument("--tree", type=str, default="data/interim/asr_output/IPR019888_aligned.fasta.treefile", help="Input Tree") # IQ-TREE outputs .treefile
    parser.add_argument("--alignment", type=str, default="data/interim/IPR019888_aligned.fasta", help="Input Alignment")
    parser.add_argument("--state", type=str, default="data/interim/asr_output/IPR019888_aligned.fasta.state", help="IQ-TREE .state file")
    parser.add_argument("--output", type=str, default="results/badasp_scores.csv", help="Output CSV file")
    
    args = parser.parse_args()
    run_badasp(args.tree, args.alignment, args.state, args.output)
