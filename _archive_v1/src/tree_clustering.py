import os
import argparse
import numpy as np
import pandas as pd
from Bio import Phylo
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

def get_leaf_distances(tree):
    """
    Computes patristic distances from the root for all leaves recursively.
    Note: For true tip-to-tip we would need all-pairs, but since we just want to cluster,
    we can convert the tree into a linkage matrix directly if we construct the distances,
    or we can cheat using cophenetic distances using DendroPy.
    Since Bio.Phylo doesn't have an instant distance matrix, we'll implement a fast
    hierarchical clustering directly from the tree structure!
    
    Actually, the easiest way to cluster a phylogenetic tree into N clades is just
    to cut it at a certain depth/height from the root.
    """
    # Assign depths
    depths = tree.depths()
    max_depth = max(depths.values())
    
    # We want to traverse from root and cut when we have K clades
    return depths, max_depth

def cluster_tree_clades(tree_file, k_clusters=50):
    # Load tree
    tree = Phylo.read(tree_file, 'newick')
    if not tree.root:
        tree.root_at_midpoint()
        
    depths, max_depth = get_leaf_distances(tree)
    
    # We will find the horizontal cut threshold that yields exactly k_clusters
    # A node is cut if its parent's depth < threshold and its depth >= threshold
    
    def count_clades_at_threshold(threshold):
        count = 0
        def traverse(node, current_depth):
            nonlocal count
            node_depth = current_depth + (node.branch_length or 0.0)
            if node_depth >= threshold and current_depth < threshold:
                count += 1
                return
            if node.is_terminal():
                if current_depth < threshold:
                    count += 1
                return
            for child in node.clades:
                traverse(child, node_depth)
        traverse(tree.root, 0.0)
        return count
        
    def assign_clusters(threshold):
        clusters = {}
        cluster_id = 1
        def traverse(node, current_depth, active_cluster=None):
            nonlocal cluster_id
            node_depth = current_depth + (node.branch_length or 0.0)
            assigned_cluster = active_cluster
            
            if node_depth >= threshold and current_depth < threshold:
                assigned_cluster = cluster_id
                cluster_id += 1
                
            if node.is_terminal():
                if assigned_cluster is None:
                    assigned_cluster = cluster_id
                    cluster_id += 1
                clusters[node.name] = assigned_cluster
                return
                
            for child in node.clades:
                traverse(child, node_depth, assigned_cluster)
                
        traverse(tree.root, 0.0)
        return clusters

    # Binary search for threshold
    low, high = 0.0, max_depth
    best_thresh = max_depth / 2
    
    for _ in range(50):
        mid = (low + high) / 2
        c = count_clades_at_threshold(mid)
        best_thresh = mid
        if c == k_clusters:
            break
        elif c > k_clusters:
            high = mid
        else:
            low = mid
            
    print(f"Cutting tree at depth {best_thresh:.4f} resulting in {count_clades_at_threshold(best_thresh)} subfamilies.")
    
    assignments = assign_clusters(best_thresh)
    df = pd.DataFrame(list(assignments.items()), columns=['Sequence_ID', 'Subfamily_Cluster_ID'])
    return df

def run_clustering(tree_file, output_csv, k_clusters=50):
    df = cluster_tree_clades(tree_file, k_clusters=k_clusters)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Clustered {len(df)} sequences into {df['Subfamily_Cluster_ID'].nunique()} functional subfamilies.")
    print(f"Saved cluster assignments to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k_clusters", type=int, default=50)
    args = parser.parse_args()
    
    run_clustering(args.tree, args.output, args.k_clusters)
