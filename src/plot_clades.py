import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from Bio import Phylo
import sys

# Increase recursion depth for large tree plotting
sys.setrecursionlimit(10000)

def plot_badasp_tree(tree_file, scores_file, output_pdf, threshold=0.95, title="BADASP Functional Clades"):
    print(f"Loading tree: {tree_file}")
    tree = Phylo.read(tree_file, "newick")
    tree.root_at_midpoint()
    
    # 1. Identify specific branches with target percentile switch events
    scores_df = pd.read_csv(scores_file)
    thresh = scores_df['BADASP_Score'].quantile(threshold)
    switches = scores_df[scores_df['BADASP_Score'] >= thresh]
    switch_nodes = set(switches['Node'].unique())
    
    # 2. Reproduce the clade tracing strictly top-down to map internal branch colors perfectly
    clade_counter = [1]
    
    # Recursive top-down coloring
    def assign_clades_top_down(node, current_clade_id):
        assigned_clade = current_clade_id
        if node.name in switch_nodes:
            clade_counter[0] += 1
            assigned_clade = clade_counter[0]
            
        node.clade_id = assigned_clade # Store dynamically
        
        if not node.is_terminal():
            for child in node.clades:
                assign_clades_top_down(child, assigned_clade)
                
    assign_clades_top_down(tree.root, 1)
    
    unique_clades = list(range(1, clade_counter[0] + 1))
    print(f"Algorithm colored exactly {len(unique_clades)} overarching specificities across the tree.")
    
    import seaborn as sns
    # Generate distinct pastel colors
    pastel_palette = sns.color_palette("pastel", len(unique_clades))
    np.random.seed(42)
    indexed_palette = list(pastel_palette)
    np.random.shuffle(indexed_palette) 
    clade_color_map = {clade: mcolors.to_hex(c) for clade, c in zip(unique_clades, indexed_palette)}
    
    # Actually physically apply the mapped colors
    for clade in tree.find_clades():
        clade.color = clade_color_map[clade.clade_id]
        if clade.is_terminal():
            clade.name = '' # Clear leaf names for clean plot

    # Plotting
    fig = plt.figure(figsize=(12, 18), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    
    # Hide all native node labels (especially dense leaf labels) for readability.
    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        label_func=lambda _: None,
        branch_labels=lambda _: None,
    )
    
    # Reconstruct Y-coordinates for brackets!
    terminals = tree.get_terminals()
    y_coords = {node: i+1 for i, node in enumerate(terminals)}
    
    clade_y_bounds = {}
    for leaf in terminals:
        y = y_coords[leaf]
        cid = leaf.clade_id
        if cid not in clade_y_bounds:
            clade_y_bounds[cid] = [y, y]
        else:
            clade_y_bounds[cid][0] = min(clade_y_bounds[cid][0], y)
            clade_y_bounds[cid][1] = max(clade_y_bounds[cid][1], y)
            
    # Determine the maximum X coordinate computationally
    ax_xlim = ax.get_xlim()
    max_x = ax_xlim[1] * 1.05
    bracket_extend = ax_xlim[1] * 0.02
    
    # Draw brackets!
    for clade_id, (min_y, max_y) in clade_y_bounds.items():
        color = clade_color_map[clade_id]
        
        # Don't draw the bracket for the absolute ancestral root clade if it spans the whole entire graph (avoids visual clutter)
        if clade_id == 1 and (max_y - min_y) > len(terminals) * 0.5:
            ax.text(max_x + bracket_extend, (min_y + max_y)/2, f"Clade {clade_id} (Ancestral Core)", va='center', ha='left', fontsize=9, color=color, fontweight='bold', alpha=0.7)
            continue
            
        ax.plot([max_x, max_x], [min_y, max_y], color=color, lw=3)
        ax.plot([max_x - bracket_extend, max_x], [max_y, max_y], color=color, lw=2)
        ax.plot([max_x - bracket_extend, max_x], [min_y, min_y], color=color, lw=2)
        
        mid_y = (min_y + max_y) / 2
        ax.text(max_x + bracket_extend, mid_y, f"Clade {clade_id}", va='center', ha='left', fontsize=8, color=color, fontweight='bold')
    
    ax.set_xlim(ax_xlim[0], max_x + bracket_extend * 15)
    ax.set_title(title, fontsize=16)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved beautifully-annotated contiguous biological tree to {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Phylogenetic Landscape of BADASP Switch Events")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    
    plot_badasp_tree(args.tree, args.scores, args.output, args.threshold, args.title)
