import os
import subprocess
import argparse
import sys

def cluster_sequences(input_fasta, output_fasta, identity=0.6, memory=2000):
    """
    Cluster protein sequences using CD-HIT.
    """
    print(f"Clustering {input_fasta} at {identity*100}% identity...")
    
    if not os.path.exists(input_fasta):
        print(f"Error: {input_fasta} does not exist.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    
    word_size = 5
    if identity < 0.7:
        word_size = 4
    if identity < 0.6:
        word_size = 3
        
    cmd = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(identity),
        "-n", str(word_size),
        "-M", str(memory),
        "-T", "0" # use all available threads
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        print(f"Successfully clustered sequences to {output_fasta}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running CD-HIT:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: CD-HIT is not installed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster sequences using CD-HIT.")
    parser.add_argument("--input", type=str, default="data/raw/IPR019888.fasta", help="Input FASTA")
    parser.add_argument("--output", type=str, default="data/raw/IPR019888_clustered.fasta", help="Output clustered FASTA")
    parser.add_argument("--identity", type=float, default=0.6, help="Identity threshold (default: 0.6)")
    
    args = parser.parse_args()
    cluster_sequences(args.input, args.output, args.identity)
