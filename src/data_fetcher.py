import os
import requests
import argparse
import sys

def fetch_uniprot_fasta(interpro_id, output_path, reviewed_only=False):
    """
    Fetches protein sequences associated with an InterPro ID from UniProt.
    """
    print(f"Fetching sequences for {interpro_id}...")
    
    # Construct the query
    query = f"xref:interpro-{interpro_id}"
    if reviewed_only:
        query += " AND reviewed:true"
    
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": query,
    }
    
    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        temp_path = output_path + ".tmp"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        # Filter sequences by length using interquartile range (IQR) to remove extreme outliers
        from Bio import SeqIO
        import numpy as np
        
        records = list(SeqIO.parse(temp_path, "fasta"))
        if not records:
            print("No sequences found from UniProt.")
            sys.exit(1)
            
        lengths = [len(r.seq) for r in records]
        q1, q3 = np.percentile(lengths, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_records = [r for r in records if lower_bound <= len(r.seq) <= upper_bound]
        
        SeqIO.write(filtered_records, output_path, "fasta")
        os.remove(temp_path)
        
        print(f"Successfully downloaded {len(records)} sequences. Filtered to {len(filtered_records)} within expected length ({lower_bound:.0f}-{upper_bound:.0f} AAs). Saved to {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from UniProt: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch InterPro associated sequences from UniProt.")
    parser.add_argument("--id", type=str, default="IPR019888", help="InterPro ID to fetch (default: IPR019888)")
    parser.add_argument("--output", type=str, default="data/raw/IPR019888.fasta", help="Output FASTA file path")
    parser.add_argument("--reviewed", action="store_true", help="Fetch reviewed sequences only (default: fetch all)")
    
    args = parser.parse_args()
    
    fetch_uniprot_fasta(args.id, args.output, reviewed_only=args.reviewed)
