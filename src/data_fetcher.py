import os
import requests
import argparse
import sys

def fetch_uniprot_fasta(interpro_id, output_path, reviewed_only=True):
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
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Successfully downloaded sequences to {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from UniProt: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch InterPro associated sequences from UniProt.")
    parser.add_argument("--id", type=str, default="IPR019888", help="InterPro ID to fetch (default: IPR019888)")
    parser.add_argument("--output", type=str, default="data/raw/IPR019888.fasta", help="Output FASTA file path")
    parser.add_argument("--all", action="store_true", help="Fetch unreviewed sequences as well (default: reviewed only)")
    
    args = parser.parse_args()
    
    fetch_uniprot_fasta(args.id, args.output, reviewed_only=not args.all)
