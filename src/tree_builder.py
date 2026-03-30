import os
import subprocess
import argparse
import sys

def build_tree(input_alignment, output_tree):
    """
    Builds a phylogenetic tree using FastTree.
    """
    print(f"Building phylogenetic tree for {input_alignment} using FastTree...")
    
    if not os.path.exists(input_alignment):
        print(f"Error: Input alignment file {input_alignment} does not exist.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(output_tree), exist_ok=True)
    
    # FastTree is optimized for protein alignments with the default settings (WAG+CAT)
    cmd = ["fasttree", input_alignment]
    
    try:
        with open(output_tree, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, check=True, text=True)
        print(f"Successfully generated phylogenetic tree and saved to {output_tree}")
    except subprocess.CalledProcessError as e:
        print(f"Error running FastTree:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: FastTree is not installed or not in PATH. Please install FastTree.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Phylogenetic Tree using FastTree.")
    parser.add_argument("--input", type=str, default="data/interim/IPR019888_aligned.fasta", help="Input aligned FASTA file")
    parser.add_argument("--output", type=str, default="data/interim/IPR019888.tree", help="Output Newick tree file")
    
    args = parser.parse_args()
    build_tree(args.input, args.output)
