#!/usr/bin/env python3
"""
Expand environmental/unresolved taxonomy detection across three FASTA files.
Output aggregated counts to CSV with expanded keyword detection.
"""
import re
import csv
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO


def is_environmental(os_value: str) -> bool:
    """
    Expanded detection of environmental/unresolved sequences.
    Catches: metagenome, environmental, uncultured, unidentified, mixed, enrichment,
    bacterium, marine, soil, water, sludge, sp., gen., candidatus, and other unresolved patterns.
    """
    if not os_value:
        return False
    
    # Use same pattern as reconciliation.py for consistency
    pattern = re.compile(
        r"metagenome|environmental|uncultured|unidentified|mixed|enrichment|"
        r"\bbacterium\b|\bmarine\b|\bsoil\b|\bwater\b|\bsludge\b|"
        r"\bsp\b|\bsp\.|gen\.|candidatus",
        re.IGNORECASE
    )
    
    return bool(pattern.search(os_value))


def audit_fasta(fasta_path: str, level_name: str) -> dict:
    """
    Scan FASTA file and aggregate by (OS, Is_Environmental).
    Returns dict: { (os_value, is_env): count }
    """
    aggregates = defaultdict(int)
    
    for record in SeqIO.parse(fasta_path, 'fasta'):
        # Extract OS= or OX= from description
        os_value = None
        if 'OS=' in record.description:
            match = re.search(r'OS=([^;]*)', record.description)
            if match:
                os_value = match.group(1).strip()
        elif 'OX=' in record.description:
            match = re.search(r'OX=([^;]*)', record.description)
            if match:
                os_value = match.group(1).strip()
        
        if not os_value:
            os_value = '[No OS/OX]'
        
        is_env = is_environmental(os_value)
        aggregates[(os_value, is_env)] += 1
    
    return aggregates


def main():
    files = [
        ('data/raw/IPR019888.fasta', 'Raw'),
        ('data/interim/IPR019888_length_filtered.fasta', 'Length_Filtered'),
        ('data/interim/IPR019888_clustered.fasta', 'Clustered'),
    ]
    
    all_data = []
    
    for fasta_file, level_name in files:
        path = Path(fasta_file)
        if not path.exists():
            print(f'⚠ File not found: {fasta_file}')
            continue
        
        aggregates = audit_fasta(fasta_file, level_name)
        
        total = sum(aggregates.values())
        env_count = sum(count for (os_val, is_env), count in aggregates.items() if is_env)
        env_pct = 100.0 * env_count / total if total > 0 else 0.0
        
        print(f'\n{level_name}:')
        print(f'  Total sequences: {total}')
        print(f'  Environmental/Unresolved: {env_count} ({env_pct:.1f}%)')
        print(f'  Resolved: {total - env_count} ({100 - env_pct:.1f}%)')
        
        for (os_val, is_env), count in sorted(aggregates.items()):
            all_data.append({
                'Pipeline_Level': level_name,
                'Taxa_Descriptor': os_val,
                'Is_Environmental': 'Yes' if is_env else 'No',
                'Sequence_Count': count,
            })
    
    # Write CSV
    output_csv = Path('results/environmental_taxa_audit.csv')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Pipeline_Level', 'Taxa_Descriptor', 'Is_Environmental', 'Sequence_Count'
        ])
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f'\n✓ CSV written to {output_csv}')
    print(f'  Total rows: {len(all_data)}')


if __name__ == '__main__':
    main()
