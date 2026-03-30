"""
Fetches only the DNA-Binding Domain (DBD / HTH region) of each AsnC-like TF.

Strategy:
  - IPR019888 describes the HTH DBD itself, but the sequences we downloaded are
    full-length proteins that also contain the C-terminal RAM (regulatory ligand
    binding) domain.
  - We query the UniProt Features API to retrieve the start/end coordinates of
    the IPR019888 (or PF04278 / Pfam HTH_AsnC) domain annotation for every
    accession and then slice out only those residues.
  - The resulting FASTA contains only the DBD segment (~60-70 AA), which can
    then be aligned and analysed independently.
"""

import os
import re
import sys
import time
import argparse
import requests
from Bio import SeqIO
from io import StringIO


DOMAIN_DESCRIPTION = "HTH asnC-type"   # UniProt DOMAIN annotation for the DBD


def get_domain_coordinates(uniprot_acc):
    """
    Queries the UniProt Features API for Pfam domain coordinates.
    Returns (start, end) 1-based inclusive, or None if not found.
    """
    url = f"https://www.ebi.ac.uk/proteins/api/features/{uniprot_acc}"
    headers = {"Accept": "application/json"}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            for feature in data.get("features", []):
                # Match by DOMAIN feature description (UniProt does not attach Pfam IDs here)
                if feature.get("type") == "DOMAIN" and DOMAIN_DESCRIPTION in feature.get("description", ""):
                    begin = int(feature["begin"])
                    end   = int(feature["end"])
                    return (begin, end)
            return None
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
    return None


def extract_dbd_sequences(input_fasta, output_fasta, padding=5):
    """
    For every sequence in input_fasta, look up the HTH DBD coordinates from
    UniProt and write the sliced region to output_fasta.

    Parameters
    ----------
    padding : int
        Extra residues to add on each side of the annotated domain (default 5).
    """
    records = list(SeqIO.parse(input_fasta, "fasta"))
    print(f"Extracting DBD regions for {len(records)} sequences...")

    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    written = 0
    skipped = 0

    with open(output_fasta, "w") as out_f:
        for i, rec in enumerate(records):
            # Parse the UniProt accession from the FASTA header (sp|ACC|NAME or tr|ACC|NAME)
            match = re.match(r"[a-z]{2}\|([A-Z0-9]+)\|", rec.id)
            acc = match.group(1) if match else rec.id.split("|")[0]

            coords = get_domain_coordinates(acc)
            if coords is None:
                skipped += 1
                continue

            start, end = coords
            start = max(1, start - padding)
            end   = min(len(rec.seq), end + padding)

            dbd_seq = rec.seq[start - 1: end]   # convert to 0-based slice
            rec.seq = dbd_seq
            SeqIO.write(rec, out_f, "fasta")
            written += 1

            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(records)} ({written} written, {skipped} skipped)...")
            # Be polite to the API
            time.sleep(0.05)

    print(f"\nDone. Wrote {written} DBD sequences to {output_fasta} ({skipped} skipped — no annotation found).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DBD sub-sequences from full-length AsnC-like TFs.")
    parser.add_argument("--input",   type=str, default="data/raw/IPR019888_clustered.fasta",
                        help="Input full-length FASTA (default: clustered)")
    parser.add_argument("--output",  type=str, default="data/raw/IPR019888_dbd.fasta",
                        help="Output DBD FASTA")
    parser.add_argument("--padding", type=int, default=5,
                        help="Flanking residues around domain (default: 5)")
    args = parser.parse_args()
    extract_dbd_sequences(args.input, args.output, args.padding)
