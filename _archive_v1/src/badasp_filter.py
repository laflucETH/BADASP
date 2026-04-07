import os
import argparse
import pandas as pd
from Bio import Phylo

def filter_deep_nodes(tree_file, badasp_csv, clusters_csv, output_csv):
    tree = Phylo.read(tree_file, 'newick')
    if not tree.root:
        tree.root_at_midpoint()
        
    scores_df = pd.read_csv(badasp_csv)
    clusters_df = pd.read_csv(clusters_csv)
    
    # Map leaf name to cluster ID
    leaf_to_cluster = dict(zip(clusters_df['Sequence_ID'], clusters_df['Subfamily_Cluster_ID']))
    
    # Map internal node names to the set of unique clusters underneath them
    node_clusters = {}
    
    # IQ-TREE labels internal nodes. Bio.Phylo parses them if they exist as node.name or node.confidence
    # We need to traverse and build the descendant cluster sets
    for clade in tree.find_clades(order='postorder'):
        if clade.is_terminal():
            c_id = leaf_to_cluster.get(clade.name)
            node_clusters[clade] = {c_id} if c_id is not None else set()
        else:
            c_set = set()
            for child in clade.clades:
                c_set.update(node_clusters[child])
            node_clusters[clade] = c_set
            
    # Now build a mapping of the internal node name (as written by IQ-TREE/BADASP) to whether it's deep or shallow
    # A node is deep if it acts as an ancestor to > 1 different subfamily clusters
    is_node_deep = {}
    for clade in tree.get_nonterminals():
        name = clade.name
        # sometimes IQ-TREE stores internal labels in confidence if they are numbers
        if name is None and clade.confidence is not None:
             name = str(clade.confidence)
             
        if name:
            c_set = node_clusters.get(clade, set())
            is_node_deep[name] = len(c_set) > 1

    print(f"Analyzed {len(is_node_deep)} internal nodes from tree.")
    
    # Filter the BADASP scores
    initial_rows = len(scores_df)
    scores_df['Is_Deep_Node'] = scores_df['Node'].map(is_node_deep).fillna(False)
    
    deep_scores = scores_df[scores_df['Is_Deep_Node']].copy()
    filtered_rows = len(deep_scores)
    
    print(f"Filtered out shallow intraclade mutational drift.")
    print(f"Retained {filtered_rows} out of {initial_rows} BADASP scoring events ({filtered_rows/initial_rows*100:.1f}%) occurring ONLY at inter-subfamily divergence nodes.")
    
    if filtered_rows == 0:
        print("WARNING: No deep nodes remaining! Try increasing k_clusters.")
        return
        
    # Recalculate 95th percentile strictly on deep nodes
    thresh_95 = deep_scores['BADASP_Score'].quantile(0.95)
    deep_scores['Is_Functional_Switch'] = deep_scores['BADASP_Score'] > thresh_95
    
    print(f"New Deep-Node Functional Switch 95th Percentile Threshold: {thresh_95:.4f}")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    deep_scores.to_csv(output_csv, index=False)
    print(f"Saved cleaned BADASP scores to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--badasp", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    filter_deep_nodes(args.tree, args.badasp, args.clusters, args.output)
