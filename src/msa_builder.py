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
            
        # Use a gap-fraction threshold of 0.5: remove columns where >50% of residues are gaps.
        # -automated1 over-trims this dataset (it retained only 26 of 1427 columns); 
        # -gt 0.5 yields ~151 columns, matching the expected ~155 AA domain length.
        print("Trimming MSA with trimAl (-gt 0.5, keeping columns present in >=50% of seqs)...")
        trimal_cmd = [
            "trimal",
            "-in", output_fasta,
            "-out", output_fasta + ".trimmed",
            "-gt", "0.5"
        ]
        
        trimal_process = subprocess.run(trimal_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if trimal_process.returncode != 0:
            print(f"Error running trimAl:\n{trimal_process.stderr}", file=sys.stderr)
            sys.exit(1)
            
        # Replace the original with the trimmed version
        os.rename(output_fasta + ".trimmed", output_fasta)
        
        print(f"Successfully generated and trimmed MSA. Saved to {output_fasta}")
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
