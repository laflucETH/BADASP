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
    
    # Generate distinct colors using the turbo colormap
    cmap = plt.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0, 1, len(unique_clades)))
    np.random.seed(42)
    np.random.shuffle(colors) # Shuffle for maximum contrast between adjacent clades
    clade_color_map = {clade: mcolors.to_hex(c) for clade, c in zip(unique_clades, colors)}
    
    # We must explicitly color internal branches identically to their children IF the children share a clade.
    # We basically do a bottom-up propagation: if all children of a node belong to Clade X, the node is Clade X.
    # Otherwise, it's a structural root/transition branch.
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
            node.color = "#000000" # Black for nodes transitioning between distinct functional specificities
            return None
            
    assign_clade_colors(tree.root)
    
    # Plotting
    fig = plt.figure(figsize=(15, 20), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    
    Phylo.draw(tree, axes=ax, label_func=lambda x: '', do_show=False)
    
    ax.set_title(title, fontsize=16)
    plt.axis('off') # Cleaner look
    
    # Create legend (only for first 20 to avoid completely flooding the image)
    legend_elements = [Patch(facecolor=clade_color_map[c], label=c) for c in list(unique_clades)[:20]]
    if len(unique_clades) > 20:
        legend_elements.append(Patch(facecolor='#000000', label=f"+ {len(unique_clades)-20} more clades"))
        
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), title="Functional Clades")
    
    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved colored tree map to {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Phylogenetic Landscape of BADASP Switch Events")
    args = parser.parse_args()
    
    plot_badasp_tree(args.tree, args.mapping, args.output, args.title)
