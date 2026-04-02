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

def plot_badasp_tree(tree_file, clade_csv, output_pdf, title="BADASP Functional Clades"):
    print(f"Loading tree: {tree_file}")
    tree = Phylo.read(tree_file, "newick")
    tree.root_at_midpoint()
    
    # Load sequence to clade mapping
    df = pd.read_csv(clade_csv)
    leaf_to_clade = dict(zip(df['Sequence_ID'], df['Functional_Clade_ID']))
    
    unique_clades = df['Functional_Clade_ID'].unique()
    print(f"Found {len(unique_clades)} distinct clades.")
    
    import seaborn as sns
    # Generate distinct pastel colors
    pastel_palette = sns.color_palette("pastel", len(unique_clades))
    np.random.seed(42)
    indexed_palette = list(pastel_palette)
    np.random.shuffle(indexed_palette) 
    clade_color_map = {clade: mcolors.to_hex(c) for clade, c in zip(unique_clades, indexed_palette)}
    
    # We must explicitly color internal branches identically to their children IF the children share a clade.
    def assign_clade_colors(node):
        if node.is_terminal():
            clade_id = leaf_to_clade.get(node.name, None)
            if clade_id:
                node.color = clade_color_map[clade_id]
                return clade_id
            return None
            
        child_clades = set()
        for child in node.clades:
            c = assign_clade_colors(child)
            if c:
                child_clades.add(c)
                
        if len(child_clades) == 1:
            shared_clade = list(child_clades)[0]
            node.color = clade_color_map[shared_clade]
            return shared_clade
        else:
            node.color = "#bbbbbb" # Subtle gray for internal generic structural splits
            return None
            
    assign_clade_colors(tree.root)
    
    # Plotting
    fig = plt.figure(figsize=(12, 18), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    
    Phylo.draw(tree, axes=ax, label_func=lambda x: '', do_show=False)
    
    # Reconstruct Y-coordinates. Phylo.draw internally iterates get_terminals() 
    # and assigns them Y=1, Y=2... from bottom to top.
    terminals = tree.get_terminals()
    y_coords = {node.name: i+1 for i, node in enumerate(terminals)}
    
    # We find the min/max Y for each clade to draw a functional bracket!
    clade_y_bounds = {}
    for name, clade_id in leaf_to_clade.items():
        if name in y_coords:
            y = y_coords[name]
            if clade_id not in clade_y_bounds:
                clade_y_bounds[clade_id] = [y, y]
            else:
                clade_y_bounds[clade_id][0] = min(clade_y_bounds[clade_id][0], y)
                clade_y_bounds[clade_id][1] = max(clade_y_bounds[clade_id][1], y)
                
    # Determine the maximum X coordinate computationally
    ax_xlim = ax.get_xlim()
    max_x = ax_xlim[1] * 1.05
    bracket_extend = ax_xlim[1] * 0.02
    
    # Draw brackets!
    for clade_id, (min_y, max_y) in clade_y_bounds.items():
        color = clade_color_map[clade_id]
        # Vertical line for the bracket spanning the clades physical height
        ax.plot([max_x, max_x], [min_y, max_y], color=color, lw=3)
        # Top and bottom horizontal notches
        ax.plot([max_x - bracket_extend, max_x], [max_y, max_y], color=color, lw=2)
        ax.plot([max_x - bracket_extend, max_x], [min_y, min_y], color=color, lw=2)
        
        # Determine the most prominent representative in this clade (we grab the first valid sequence structurally)
        mid_y = (min_y + max_y) / 2
        ax.text(max_x + bracket_extend, mid_y, clade_id, va='center', ha='left', fontsize=8, color=color, fontweight='bold')
    
    # Adjust axes specifically so the brackets fit flawlessly
    ax.set_xlim(ax_xlim[0], max_x + bracket_extend * 15)
    
    ax.set_title(title, fontsize=16)
    plt.axis('off') # Cleaner look
    
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved aesthetically-refined colored tree map to {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Phylogenetic Landscape of BADASP Switch Events")
    args = parser.parse_args()
    
    plot_badasp_tree(args.tree, args.mapping, args.output, args.title)
