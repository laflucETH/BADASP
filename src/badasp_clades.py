import os
import argparse
import pandas as pd
from Bio import Phylo

def assign_clades(tree_file, scores_file, output_csv, prefix="BADASP_Clade_"):
    # 1. Identify specific branches with >95th percentile switch events
    df = pd.read_csv(scores_file)
    thresh = df['BADASP_Score'].quantile(0.95)
    switches = df[df['BADASP_Score'] >= thresh]
    switch_nodes = set(switches['Node'].unique())
    
    print(f"Identified {len(switch_nodes)} functional switch branches.")
    
    # 2. Parse tree
    tree = Phylo.read(tree_file, "newick")
    
    # The IQ-TREE generated .treefile inherently contains the 'NodeX' labels 
    # matched identically to the ancestral probability output states.
    # 3. Recursive top-down slicing
    clade_counter = [1]
    leaf_assignments = []
    
    def traverse(node, current_clade_id):
        # If this branch experienced a >95th%ile BADASP functional burst, it branches into a NEW distinct clade.
        assigned_clade = current_clade_id
        if node.name in switch_nodes:
            clade_counter[0] += 1
            assigned_clade = clade_counter[0]
            
        if node.is_terminal():
            leaf_assignments.append({
                'Sequence_ID': node.name,
                'Functional_Clade_ID': f"{prefix}{assigned_clade}"
            })
        else:
            for child in node.clades:
                traverse(child, assigned_clade)
                
    # Root starts in Clade 1
    traverse(tree.root, 1)
    
    # 4. Save
    out_df = pd.DataFrame(leaf_assignments)
    out_df.to_csv(output_csv, index=False)
    
    print(f"Algorithm successfully partitioned the tree into {clade_counter[0]} defined functional specificities.")
    print(f"Sequence -> Clade mapping saved to {output_csv}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, help="Newick phylogenetic tree")
    parser.add_argument("--scores", required=True, help="BADASP scores CSV")
    parser.add_argument("--output", required=True, help="Output clade mapping CSV")
    parser.add_argument("--prefix", default="BADASP_Clade_", help="Prefix for final taxonomic name")
    
    args = parser.parse_args()
    assign_clades(args.tree, args.scores, args.output, args.prefix)
