"""
Phase 5: Restricted BADASP Scoring for Specificity Determining Position (SDP) Identification.

Implementation of the adapted BADASP algorithm:
  Score = RC * AC * p(AC)
  
Where:
  - RC: Recent conservation within clade modern sequences
  - AC: Ancestral conservation (BLOSUM62 substitution score between LCA pairs)
  - p(AC): Posterior probability of ancestral assignment from IQ-TREE

This implementation restricts scoring to deep ancestral divergence events between
the 34 defined topological clades, filtering out shallow intra-clade drift.

95th percentile threshold isolates true "bursts" of functional divergence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from Bio import SeqIO, AlignIO
import logging

logger = logging.getLogger(__name__)


def load_state_file(state_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load IQ-TREE .state file and organize by node into a dictionary of DataFrames.
    
    Args:
        state_path: Path to IQ-TREE .state file
        
    Returns:
        Dictionary mapping node names (e.g., "Node10") to DataFrames with posterior probabilities
    """
    state_path = Path(state_path)
    state_data = {}
    
    with open(state_path, "r") as f:
        lines = f.readlines()
    
    # Skip header lines (lines starting with # and the header "Node Site State p_A...")
    data_lines = []
    for l in lines:
        if l.strip().startswith("#"):
            continue
        if l.strip().startswith("Node") and "Site" in l and "State" in l:
            # This is the column header row, skip it
            continue
        data_lines.append(l)
    
    if not data_lines:
        raise ValueError(f"No data lines found in state file {state_path}")
    
    # Parse tab-separated data
    for line in data_lines:
        if not line.strip():
            continue
        
        parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        
        try:
            node = parts[0]
            site = int(parts[1])
            state = parts[2]
            # Posterior probabilities: columns 3+ are p_A, p_R, ..., p_V
            probs = [float(p) for p in parts[3:]]
        except (ValueError, IndexError):
            continue
        
        if node not in state_data:
            state_data[node] = []
        
        state_data[node].append({
            "Site": site,
            "State": state,
            "probs": probs,
            "raw": parts[3:]
        })
    
    # Convert lists to DataFrames for each node
    result = {}
    for node, rows in state_data.items():
        df = pd.DataFrame(rows)
        result[node] = df
    
    return result


def parse_clade_assignments(assignments_path: Path) -> pd.DataFrame:
    """
    Parse clade assignment CSV file (output from Phase 3 clustering).
    
    Args:
        assignments_path: Path to tree_cluster_assignments.csv
        
    Returns:
        DataFrame with columns: sequence_id, clade_id, lca_node
    """
    assignments = pd.read_csv(assignments_path)
    return assignments


def calculate_recent_conservation(
    sequences: List[str], 
    position: int
) -> float:
    """
    Calculate recent conservation (RC) for a position within a clade's modern sequences.
    
    RC = frequency of most common amino acid / total sequences
    
    Args:
        sequences: List of aligned sequences (all same length)
        position: 0-indexed position in alignment
        
    Returns:
        Float between 0 and 1 representing conservation
    """
    if position < 0 or position >= len(sequences[0]):
        return 0.0
    
    if len(sequences) == 0:
        return 0.0
    
    # Extract position from all sequences
    residues = [seq[position] for seq in sequences if len(seq) > position]
    
    if not residues:
        return 0.0
    
    # Count most common residue
    from collections import Counter
    counts = Counter(residues)
    most_common_count = counts.most_common(1)[0][1]
    
    return most_common_count / len(residues)


def calculate_ancestral_conservation(aa1: str, aa2: str) -> float:
    """
    Calculate ancestral conservation (AC) using BLOSUM62 substitution matrix.
    
    AC = BLOSUM62 score between two ancestral amino acids.
        Positive scores favor the pair; negative scores penalize it.
    
    Args:
        aa1: First amino acid
        aa2: Second amino acid
        
    Returns:
        BLOSUM62 score (typically -4 to 11 for biological proteins)
    """
    # BLOSUM62 matrix (simplified 20x20 for standard amino acids)
    blosum62 = {
        ('A', 'A'): 4,   ('A', 'R'): -1,  ('A', 'N'): -2,  ('A', 'D'): -2,  ('A', 'C'): 0,
        ('A', 'Q'): -1,  ('A', 'E'): -1,  ('A', 'G'): 0,   ('A', 'H'): -2,  ('A', 'I'): -1,
        ('A', 'L'): -1,  ('A', 'K'): -1,  ('A', 'M'): -1,  ('A', 'F'): -2,  ('A', 'P'): -1,
        ('A', 'S'): 1,   ('A', 'T'): 0,   ('A', 'W'): -3,  ('A', 'Y'): -2,  ('A', 'V'): 0,
        
        ('R', 'A'): -1,  ('R', 'R'): 5,   ('R', 'N'): 0,   ('R', 'D'): -2,  ('R', 'C'): -3,
        ('R', 'Q'): 1,   ('R', 'E'): 0,   ('R', 'G'): -2,  ('R', 'H'): 0,   ('R', 'I'): -3,
        ('R', 'L'): -2,  ('R', 'K'): 2,   ('R', 'M'): -1,  ('R', 'F'): -3,  ('R', 'P'): -2,
        ('R', 'S'): -1,  ('R', 'T'): -1,  ('R', 'W'): -3,  ('R', 'Y'): -2,  ('R', 'V'): -3,
        
        ('N', 'A'): -2,  ('N', 'R'): 0,   ('N', 'N'): 6,   ('N', 'D'): 1,   ('N', 'C'): -3,
        ('N', 'Q'): 0,   ('N', 'E'): 0,   ('N', 'G'): 0,   ('N', 'H'): 1,   ('N', 'I'): -3,
        ('N', 'L'): -4,  ('N', 'K'): 0,   ('N', 'M'): -2,  ('N', 'F'): -3,  ('N', 'P'): -2,
        ('N', 'S'): 1,   ('N', 'T'): 0,   ('N', 'W'): -4,  ('N', 'Y'): -2,  ('N', 'V'): -3,
        
        ('D', 'A'): -2,  ('D', 'R'): -2,  ('D', 'N'): 1,   ('D', 'D'): 6,   ('D', 'C'): -3,
        ('D', 'Q'): 0,   ('D', 'E'): 2,   ('D', 'G'): -1,  ('D', 'H'): -1,  ('D', 'I'): -3,
        ('D', 'L'): -4,  ('D', 'K'): -1,  ('D', 'M'): -3,  ('D', 'F'): -3,  ('D', 'P'): -1,
        ('D', 'S'): 0,   ('D', 'T'): -1,  ('D', 'W'): -4,  ('D', 'Y'): -3,  ('D', 'V'): -3,
        
        ('C', 'A'): 0,   ('C', 'R'): -3,  ('C', 'N'): -3,  ('C', 'D'): -3,  ('C', 'C'): 9,
        ('C', 'Q'): -3,  ('C', 'E'): -4,  ('C', 'G'): -3,  ('C', 'H'): -3,  ('C', 'I'): -1,
        ('C', 'L'): -1,  ('C', 'K'): -3,  ('C', 'M'): -1,  ('C', 'F'): -2,  ('C', 'P'): -3,
        ('C', 'S'): -1,  ('C', 'T'): -1,  ('C', 'W'): -2,  ('C', 'Y'): -2,  ('C', 'V'): -1,
        
        ('Q', 'A'): -1,  ('Q', 'R'): 1,   ('Q', 'N'): 0,   ('Q', 'D'): 0,   ('Q', 'C'): -3,
        ('Q', 'Q'): 5,   ('Q', 'E'): 2,   ('Q', 'G'): -2,  ('Q', 'H'): 0,   ('Q', 'I'): -3,
        ('Q', 'L'): -2,  ('Q', 'K'): 1,   ('Q', 'M'): 0,   ('Q', 'F'): -3,  ('Q', 'P'): -1,
        ('Q', 'S'): 0,   ('Q', 'T'): -1,  ('Q', 'W'): -2,  ('Q', 'Y'): -1,  ('Q', 'V'): -2,
        
        ('E', 'A'): -1,  ('E', 'R'): 0,   ('E', 'N'): 0,   ('E', 'D'): 2,   ('E', 'C'): -4,
        ('E', 'Q'): 2,   ('E', 'E'): 5,   ('E', 'G'): -2,  ('E', 'H'): 0,   ('E', 'I'): -3,
        ('E', 'L'): -3,  ('E', 'K'): 1,   ('E', 'M'): -2,  ('E', 'F'): -3,  ('E', 'P'): -1,
        ('E', 'S'): 0,   ('E', 'T'): -1,  ('E', 'W'): -3,  ('E', 'Y'): -2,  ('E', 'V'): -2,
        
        ('G', 'A'): 0,   ('G', 'R'): -2,  ('G', 'N'): 0,   ('G', 'D'): -1,  ('G', 'C'): -3,
        ('G', 'Q'): -2,  ('G', 'E'): -2,  ('G', 'G'): 6,   ('G', 'H'): -2,  ('G', 'I'): -4,
        ('G', 'L'): -4,  ('G', 'K'): -2,  ('G', 'M'): -3,  ('G', 'F'): -3,  ('G', 'P'): -2,
        ('G', 'S'): 0,   ('G', 'T'): -2,  ('G', 'W'): -2,  ('G', 'Y'): -3,  ('G', 'V'): -3,
        
        ('H', 'A'): -2,  ('H', 'R'): 0,   ('H', 'N'): 1,   ('H', 'D'): -1,  ('H', 'C'): -3,
        ('H', 'Q'): 0,   ('H', 'E'): 0,   ('H', 'G'): -2,  ('H', 'H'): 8,   ('H', 'I'): -3,
        ('H', 'L'): -3,  ('H', 'K'): -1,  ('H', 'M'): -2,  ('H', 'F'): -1,  ('H', 'P'): -2,
        ('H', 'S'): -1,  ('H', 'T'): -2,  ('H', 'W'): -2,  ('H', 'Y'): 2,   ('H', 'V'): -3,
        
        ('I', 'A'): -1,  ('I', 'R'): -3,  ('I', 'N'): -3,  ('I', 'D'): -3,  ('I', 'C'): -1,
        ('I', 'Q'): -3,  ('I', 'E'): -3,  ('I', 'G'): -4,  ('I', 'H'): -3,  ('I', 'I'): 4,
        ('I', 'L'): 2,   ('I', 'K'): -3,  ('I', 'M'): 1,   ('I', 'F'): 0,   ('I', 'P'): -3,
        ('I', 'S'): -2,  ('I', 'T'): -1,  ('I', 'W'): -3,  ('I', 'Y'): -1,  ('I', 'V'): 3,
        
        ('L', 'A'): -1,  ('L', 'R'): -2,  ('L', 'N'): -4,  ('L', 'D'): -4,  ('L', 'C'): -1,
        ('L', 'Q'): -2,  ('L', 'E'): -3,  ('L', 'G'): -4,  ('L', 'H'): -3,  ('L', 'I'): 2,
        ('L', 'L'): 4,   ('L', 'K'): -2,  ('L', 'M'): 2,   ('L', 'F'): 0,   ('L', 'P'): -3,
        ('L', 'S'): -2,  ('L', 'T'): -1,  ('L', 'W'): -2,  ('L', 'Y'): -1,  ('L', 'V'): 1,
        
        ('K', 'A'): -1,  ('K', 'R'): 2,   ('K', 'N'): 0,   ('K', 'D'): -1,  ('K', 'C'): -3,
        ('K', 'Q'): 1,   ('K', 'E'): 1,   ('K', 'G'): -2,  ('K', 'H'): -1,  ('K', 'I'): -3,
        ('K', 'L'): -2,  ('K', 'K'): 5,   ('K', 'M'): -1,  ('K', 'F'): -3,  ('K', 'P'): -1,
        ('K', 'S'): 0,   ('K', 'T'): -1,  ('K', 'W'): -3,  ('K', 'Y'): -2,  ('K', 'V'): -2,
        
        ('M', 'A'): -1,  ('M', 'R'): -1,  ('M', 'N'): -2,  ('M', 'D'): -3,  ('M', 'C'): -1,
        ('M', 'Q'): 0,   ('M', 'E'): -2,  ('M', 'G'): -3,  ('M', 'H'): -2,  ('M', 'I'): 1,
        ('M', 'L'): 2,   ('M', 'K'): -1,  ('M', 'M'): 5,   ('M', 'F'): 0,   ('M', 'P'): -2,
        ('M', 'S'): -1,  ('M', 'T'): -1,  ('M', 'W'): -1,  ('M', 'Y'): -1,  ('M', 'V'): 1,
        
        ('F', 'A'): -2,  ('F', 'R'): -3,  ('F', 'N'): -3,  ('F', 'D'): -3,  ('F', 'C'): -2,
        ('F', 'Q'): -3,  ('F', 'E'): -3,  ('F', 'G'): -3,  ('F', 'H'): -1,  ('F', 'I'): 0,
        ('F', 'L'): 0,   ('F', 'K'): -3,  ('F', 'M'): 0,   ('F', 'F'): 6,   ('F', 'P'): -4,
        ('F', 'S'): -2,  ('F', 'T'): -2,  ('F', 'W'): 1,   ('F', 'Y'): 3,   ('F', 'V'): -1,
        
        ('P', 'A'): -1,  ('P', 'R'): -2,  ('P', 'N'): -2,  ('P', 'D'): -1,  ('P', 'C'): -3,
        ('P', 'Q'): -1,  ('P', 'E'): -1,  ('P', 'G'): -2,  ('P', 'H'): -2,  ('P', 'I'): -3,
        ('P', 'L'): -3,  ('P', 'K'): -1,  ('P', 'M'): -2,  ('P', 'F'): -4,  ('P', 'P'): 7,
        ('P', 'S'): -1,  ('P', 'T'): -1,  ('P', 'W'): -4,  ('P', 'Y'): -3,  ('P', 'V'): -2,
        
        ('S', 'A'): 1,   ('S', 'R'): -1,  ('S', 'N'): 1,   ('S', 'D'): 0,   ('S', 'C'): -1,
        ('S', 'Q'): 0,   ('S', 'E'): 0,   ('S', 'G'): 0,   ('S', 'H'): -1,  ('S', 'I'): -2,
        ('S', 'L'): -2,  ('S', 'K'): 0,   ('S', 'M'): -1,  ('S', 'F'): -2,  ('S', 'P'): -1,
        ('S', 'S'): 4,   ('S', 'T'): 1,   ('S', 'W'): -3,  ('S', 'Y'): -2,  ('S', 'V'): -2,
        
        ('T', 'A'): 0,   ('T', 'R'): -1,  ('T', 'N'): 0,   ('T', 'D'): -1,  ('T', 'C'): -1,
        ('T', 'Q'): -1,  ('T', 'E'): -1,  ('T', 'G'): -2,  ('T', 'H'): -2,  ('T', 'I'): -1,
        ('T', 'L'): -1,  ('T', 'K'): -1,  ('T', 'M'): -1,  ('T', 'F'): -2,  ('T', 'P'): -1,
        ('T', 'S'): 1,   ('T', 'T'): 5,   ('T', 'W'): -2,  ('T', 'Y'): -2,  ('T', 'V'): 0,
        
        ('W', 'A'): -3,  ('W', 'R'): -3,  ('W', 'N'): -4,  ('W', 'D'): -4,  ('W', 'C'): -2,
        ('W', 'Q'): -2,  ('W', 'E'): -3,  ('W', 'G'): -2,  ('W', 'H'): -2,  ('W', 'I'): -3,
        ('W', 'L'): -2,  ('W', 'K'): -3,  ('W', 'M'): -1,  ('W', 'F'): 1,   ('W', 'P'): -4,
        ('W', 'S'): -3,  ('W', 'T'): -2,  ('W', 'W'): 11,  ('W', 'Y'): 2,   ('W', 'V'): -3,
        
        ('Y', 'A'): -2,  ('Y', 'R'): -2,  ('Y', 'N'): -2,  ('Y', 'D'): -3,  ('Y', 'C'): -2,
        ('Y', 'Q'): -1,  ('Y', 'E'): -2,  ('Y', 'G'): -3,  ('Y', 'H'): 2,   ('Y', 'I'): -1,
        ('Y', 'L'): -1,  ('Y', 'K'): -2,  ('Y', 'M'): -1,  ('Y', 'F'): 3,   ('Y', 'P'): -3,
        ('Y', 'S'): -2,  ('Y', 'T'): -2,  ('Y', 'W'): 2,   ('Y', 'Y'): 7,   ('Y', 'V'): -1,
        
        ('V', 'A'): 0,   ('V', 'R'): -3,  ('V', 'N'): -3,  ('V', 'D'): -3,  ('V', 'C'): -1,
        ('V', 'Q'): -2,  ('V', 'E'): -2,  ('V', 'G'): -3,  ('V', 'H'): -3,  ('V', 'I'): 3,
        ('V', 'L'): 1,   ('V', 'K'): -2,  ('V', 'M'): 1,   ('V', 'F'): -1,  ('V', 'P'): -2,
        ('V', 'S'): -2,  ('V', 'T'): 0,   ('V', 'W'): -3,  ('V', 'Y'): -1,  ('V', 'V'): 4,
    }
    
    aa1_up = aa1.upper()
    aa2_up = aa2.upper()
    
    # Check both orderings (matrix is symmetric but we'll be safe)
    if (aa1_up, aa2_up) in blosum62:
        return float(blosum62[(aa1_up, aa2_up)])
    elif (aa2_up, aa1_up) in blosum62:
        return float(blosum62[(aa2_up, aa1_up)])
    else:
        # Return neutral score if amino acids not in matrix
        return 0.0


def extract_posterior_probability(
    state_data: Dict[str, pd.DataFrame],
    node: str,
    site: int,
    aa: str
) -> float:
    """
    Extract posterior probability for a specific node, site, and amino acid from state data.
    
    Args:
        state_data: Dictionary from load_state_file
        node: Node name (e.g., "Node10")
        site: 1-indexed alignment site
        aa: Amino acid single letter code
        
    Returns:
        Posterior probability (0.0-1.0) or 0.0 if not found
    """
    if node not in state_data:
        return 0.0
    
    node_df = state_data[node]
    
    # Find row with matching site
    site_rows = node_df[node_df["Site"] == site]
    if site_rows.empty:
        return 0.0
    
    row = site_rows.iloc[0]
    
    # Map amino acid to index in probability array
    # Order from IQ-TREE state file: A R N D C Q E G H I L K M F P S T W Y V
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    if aa.upper() not in aa_order:
        return 0.0
    
    aa_idx = aa_order.index(aa.upper())
    
    # Extract probability from probs array
    if "probs" in row and aa_idx < len(row["probs"]):
        return float(row["probs"][aa_idx])
    
    return 0.0


def compute_badasp_scores(
    alignment_path: Path,
    assignments_path: Path,
    ancestral_path: Path,
    state_path: Path,
    clusters_path: Optional[Path] = None,
    min_clade_size: int = 5,
) -> pd.DataFrame:
    """
    Compute restricted BADASP scores for all alignment positions.
    
    Restriction: Only scores divergences between top-level clade LCAs,
    ignoring within-clade and shallow branch variations.
    
    Args:
        alignment_path: Path to trimmed alignment FASTA
        assignments_path: Path to clade assignments CSV
        ancestral_path: Path to ancestral sequences FASTA
        state_path: Path to IQ-TREE state file
        clusters_path: Path to tree_clusters.csv with LCA node info
        min_clade_size: Minimum sequences per clade to include
        
    Returns:
        DataFrame with columns: position, rc, ac, p_ac, badasp_score
    """
    # Load data
    alignment = AlignIO.read(alignment_path, "fasta")
    assignments = parse_clade_assignments(assignments_path)
    ancestral_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(ancestral_path, "fasta")}
    state_data = load_state_file(state_path)
    
    # Load clusters information if provided (contains LCA node mapping)
    if clusters_path:
        clusters_df = pd.read_csv(clusters_path)
        # Use lca_node_asr if available (corrected ASR node names), otherwise use lca_node
        node_col = "lca_node_asr" if "lca_node_asr" in clusters_df.columns else "lca_node"
        lca_mapping = dict(zip(clusters_df["clade_id"], clusters_df[node_col]))
    else:
        lca_mapping = {}
    
    # Filter clades by minimum size
    clade_sizes = assignments.groupby("clade_id").size()
    valid_clades = clade_sizes[clade_sizes >= min_clade_size].index.tolist()
    assignments = assignments[assignments["clade_id"].isin(valid_clades)]
    
    # Get alignment length
    aln_length = alignment.get_alignment_length()
    
    # Group sequences by clade
    clade_sequences = {}
    clade_lca_nodes = {}
    
    # Create mapping from terminal name to sequence
    seq_dict = {record.id: str(record.seq) for record in alignment}
    
    for clade_id in valid_clades:
        clade_seqs = assignments[assignments["clade_id"] == clade_id]
        seq_ids = clade_seqs["terminal_name"].tolist()
        
        # Get sequences for this clade
        seqs = [seq_dict[sid] for sid in seq_ids if sid in seq_dict]
        clade_sequences[clade_id] = seqs
        
        # Get LCA node for this clade
        if clade_id in lca_mapping:
            lca_node = lca_mapping[clade_id]
        else:
            # Fallback: use first entry's lca_node if available, or generate node name
            if "lca_node" in clade_seqs.columns:
                lca_node = clade_seqs.iloc[0]["lca_node"]
            else:
                lca_node = f"InternalNode_{clade_id}"
        lca_mapping[clade_id] = lca_node
        clade_lca_nodes[clade_id] = lca_node
    
    # Compute scores for each position
    scores = []
    for pos in range(aln_length):
        # RC: conservation within each clade
        rc_values = []
        for clade_id in valid_clades:
            rc = calculate_recent_conservation(clade_sequences[clade_id], pos)
            rc_values.append(rc)
        
        # Average RC across clades
        rc_avg = np.mean(rc_values) if rc_values else 0.0
        
        # AC: divergence between clade LCA pairs
        ac_values = []
        p_ac_values = []
        
        # Compare all pairs of clade LCAs
        clade_list = list(valid_clades)
        for i in range(len(clade_list)):
            for j in range(i + 1, len(clade_list)):
                clade_i = clade_list[i]
                clade_j = clade_list[j]
                
                lca_i = clade_lca_nodes[clade_i]
                lca_j = clade_lca_nodes[clade_j]
                
                # Get ancestral amino acids at this position
                if lca_i in ancestral_seqs and lca_j in ancestral_seqs:
                    if pos < len(ancestral_seqs[lca_i]) and pos < len(ancestral_seqs[lca_j]):
                        aa_i = ancestral_seqs[lca_i][pos]
                        aa_j = ancestral_seqs[lca_j][pos]
                        
                        # Skip gaps
                        if aa_i != '-' and aa_j != '-':
                            ac = calculate_ancestral_conservation(aa_i, aa_j)
                            ac_values.append(ac)
                            
                            # Extract posterior probability (use aa_i's probability)
                            p_ac = extract_posterior_probability(state_data, lca_i, pos + 1, aa_i)
                            p_ac_values.append(p_ac)
        
        # Average AC and p(AC) across all clade pairs
        ac_avg = np.mean(ac_values) if ac_values else 0.0
        p_ac_avg = np.mean(p_ac_values) if p_ac_values else 0.0
        
        # Normalize AC to 0-1 range (BLOSUM62 ranges from ~-4 to 11)
        # Shift and scale: (-4, 11) -> (0, 1)
        ac_normalized = (ac_avg + 4) / 15.0
        ac_normalized = np.clip(ac_normalized, 0, 1)
        
        # Compute BADASP score
        badasp_score = rc_avg * ac_normalized * p_ac_avg
        
        scores.append({
            "position": pos + 1,  # 1-indexed for user
            "rc": rc_avg,
            "ac": ac_normalized,
            "p_ac": p_ac_avg,
            "badasp_score": badasp_score,
        })
    
    return pd.DataFrame(scores)


def identify_sdps(scores_df: pd.DataFrame, percentile: float = 95.0) -> Tuple[pd.DataFrame, float]:
    """
    Identify Specificity Determining Positions (SDPs) at the given percentile.
    
    Args:
        scores_df: DataFrame from compute_badasp_scores
        percentile: Percentile threshold (default 95 for 95th percentile)
        
    Returns:
        Tuple of (SDP DataFrame, threshold value)
    """
    threshold = np.percentile(scores_df["badasp_score"], percentile)
    sdps = scores_df[scores_df["badasp_score"] >= threshold].copy()
    sdps = sdps.sort_values("badasp_score", ascending=False).reset_index(drop=True)
    
    return sdps, threshold


class BADASPCore:
    """
    Wrapper class for restricted BADASP scoring pipeline.
    
    Usage:
        core = BADASPCore(alignment_path, assignments_path, ancestral_path, state_path, 
                         clusters_path=clusters_path)
        scores = core.compute_scores()
        sdps, threshold = core.identify_sdps()
        core.save_results(output_dir)
    """
    
    def __init__(
        self,
        alignment_path: Path,
        assignments_path: Path,
        ancestral_path: Path,
        state_path: Path,
        clusters_path: Optional[Path] = None,
        min_clade_size: int = 5,
    ):
        self.alignment_path = Path(alignment_path)
        self.assignments_path = Path(assignments_path)
        self.ancestral_path = Path(ancestral_path)
        self.state_path = Path(state_path)
        self.clusters_path = Path(clusters_path) if clusters_path else None
        self.min_clade_size = min_clade_size
        
        self.scores = None
        self.sdps = None
        self.threshold = None
    
    def compute_scores(self) -> pd.DataFrame:
        """Compute BADASP scores for all positions."""
        self.scores = compute_badasp_scores(
            self.alignment_path,
            self.assignments_path,
            self.ancestral_path,
            self.state_path,
            clusters_path=self.clusters_path,
            min_clade_size=self.min_clade_size,
        )
        logger.info(f"BADASP scores computed for {len(self.scores)} positions")
        return self.scores
    
    def identify_sdps(self, percentile: float = 95.0) -> Tuple[pd.DataFrame, float]:
        """Identify SDPs at given percentile."""
        if self.scores is None:
            self.compute_scores()
        
        self.sdps, self.threshold = identify_sdps(self.scores, percentile=percentile)
        logger.info(f"SDPs identified: {len(self.sdps)} positions above {percentile}th percentile (threshold: {self.threshold:.4f})")
        return self.sdps, self.threshold
    
    def save_results(self, output_dir: Path):
        """Save full scores and SDP table to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.scores is None:
            raise ValueError("Run compute_scores() first")
        
        scores_path = output_dir / "badasp_scores.csv"
        self.scores.to_csv(scores_path, index=False)
        logger.info(f"Full BADASP scores saved to {scores_path}")
        
        if self.sdps is not None:
            sdp_path = output_dir / "badasp_sdps.csv"
            self.sdps.to_csv(sdp_path, index=False)
            logger.info(f"SDP table saved to {sdp_path}")
