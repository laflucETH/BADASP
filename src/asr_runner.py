import os
import subprocess
import argparse
import sys
import shutil

def run_asr(input_alignment, input_tree, output_dir):
    """
    Runs Ancestral Sequence Reconstruction using IQ-TREE.
    """
    print(f"Running ASR for {input_alignment} using IQ-TREE...")
    
    if not os.path.exists(input_alignment):
        print(f"Error: Alignment file {input_alignment} does not exist.")
        sys.exit(1)
    if not os.path.exists(input_tree):
        print(f"Error: Tree file {input_tree} does not exist.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    # We copy the alignment and tree to the output directory to keep IQ-TREE outputs contained
    base_name = os.path.basename(input_alignment)
    work_alignment = os.path.join(output_dir, base_name)
    shutil.copy(input_alignment, work_alignment)
    
    cmd = [
        "iqtree",
        "-s", work_alignment,
        "-te", input_tree,
        "-asr",       # Perform Ancestral Sequence Reconstruction
        "-m", "WAG",  # Default amino acid model
        "-T", "AUTO", # Auto threads
        "-redo"       # Overwrite existing
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        state_file = work_alignment + ".state"
        if os.path.exists(state_file):
            print(f"Successfully generated ancestral states at: {state_file}")
        else:
            print("IQ-TREE finished, but no .state file found.")
    except subprocess.CalledProcessError as e:
        print(f"Error running IQ-TREE ASR:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: IQ-TREE is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ancestral Sequence Reconstruction using IQ-TREE.")
    parser.add_argument("--alignment", type=str, default="data/interim/IPR019888_aligned.fasta", help="Input aligned FASTA file")
    parser.add_argument("--tree", type=str, default="data/interim/IPR019888.tree", help="Input Newick tree file")
    parser.add_argument("--output_dir", type=str, default="data/interim/asr_output", help="Directory to save ASR outputs")
    
    args = parser.parse_args()
    run_asr(args.alignment, args.tree, args.output_dir)
