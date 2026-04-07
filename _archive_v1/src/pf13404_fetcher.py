import os
import subprocess
from Bio import SeqIO
import pandas as pd

def run_hmmsearch(input_fasta, hmm_file, output_hmmout):
    cmd = f"hmmsearch --domtblout {output_hmmout} {hmm_file} {input_fasta} > /dev/null"
    subprocess.run(cmd, shell=True, check=True)

def parse_hmmsearch(hmmout_file):
    hits = []
    with open(hmmout_file) as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 20:
                target_name = parts[0]
                e_value = float(parts[12])
                env_from = int(parts[19])
                env_to = int(parts[20])
                hits.append({
                    'id': target_name,
                    'e_value': e_value,
                    'start': env_from,
                    'end': env_to
                })
    return pd.DataFrame(hits)

def extract_pf13404():
    raw_fasta = "data/raw/IPR019888_clustered.fasta"
    hmm_file = "data/raw/hmms/PF13404.hmm"
    hmmout = "data/interim/IPR019888_pf13404.domtblout"
    out_fasta = "data/raw/IPR019888_pf13404.fasta"
    
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("data/raw/hmms", exist_ok=True)
    
    print("Running hmmsearch with PF13404...")
    run_hmmsearch(raw_fasta, hmm_file, hmmout)
    
    print("Parsing hits...")
    hits_df = parse_hmmsearch(hmmout)
    
    # Keep best hit per sequence
    hits_df = hits_df.sort_values('e_value').drop_duplicates('id')
    print(f"Found PF13404 domain in {len(hits_df)} sequences")
    
    seqs = {record.id: record for record in SeqIO.parse(raw_fasta, "fasta")}
    
    extracted = []
    for _, row in hits_df.iterrows():
        seq_id = row['id']
        start, end = int(row['start']), int(row['end'])
        if seq_id in seqs:
            rec = seqs[seq_id]
            # HMMER coords are 1-based, python is 0-based
            sub_seq = str(rec.seq)[start-1:end]
            extracted.append(f">{seq_id}_{start}_{end}\n{sub_seq}")
            
    with open(out_fasta, 'w') as f:
        f.write('\n'.join(extracted) + '\n')
    print(f"Saved {len(extracted)} PF13404 domains to {out_fasta}")

if __name__ == '__main__':
    extract_pf13404()
