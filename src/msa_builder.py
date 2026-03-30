import os
import subprocess
import argparse
import sys

def build_msa(input_fasta, output_fasta, threads=1):
    """
    Builds a Multiple Sequence Alignment (MSA) using MAFFT.
    """
    print(f"Building MSA for {input_fasta} using MAFFT...")
    
    if not os.path.exists(input_fasta):
        print(f"Error: Input file {input_fasta} does not exist.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    
    cmd = ["mafft", "--auto", "--thread", str(threads), input_fasta]
    
    try:
        with open(output_fasta, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, check=True, text=True)
        print(f"Successfully generated MSA and saved to {output_fasta}")
    except subprocess.CalledProcessError as e:
        print(f"Error running MAFFT:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: MAFFT is not installed or not in PATH. Please install MAFFT.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Multiple Sequence Alignment using MAFFT.")
    parser.add_argument("--input", type=str, default="data/raw/IPR019888.fasta", help="Input FASTA file")
    parser.add_argument("--output", type=str, default="data/interim/IPR019888_aligned.fasta", help="Output aligned FASTA file")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for MAFFT (default: 1)")
    
    args = parser.parse_args()
    build_msa(args.input, args.output, args.threads)
