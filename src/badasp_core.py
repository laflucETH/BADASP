"""Phase 5 multi-level BADASP scoring for Groups, Families, and Subfamilies."""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.BaseTree import Clade, Tree

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_blosum62_matrix() -> Dict[Tuple[str, str], int]:
    return {
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


def _blosum62_pair_score(aa1: str, aa2: str) -> float:
    matrix = _get_blosum62_matrix()
    aa1 = aa1.upper()
    aa2 = aa2.upper()
    if (aa1, aa2) in matrix:
        return float(matrix[(aa1, aa2)])
    if (aa2, aa1) in matrix:
        return float(matrix[(aa2, aa1)])
    return 0.0


def load_state_file(state_path: Path) -> Dict[str, pd.DataFrame]:
    state_path = Path(state_path)
    state_data: Dict[str, List[Dict[str, object]]] = {}

    with state_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    data_lines = []
    for line in lines:
        if line.strip().startswith("#"):
            continue
        if line.strip().startswith("Node") and "Site" in line and "State" in line:
            continue
        data_lines.append(line)

    if not data_lines:
        raise ValueError(f"No data lines found in state file {state_path}")

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
            probs = [float(p) for p in parts[3:]]
        except (ValueError, IndexError):
            continue
        state_data.setdefault(node, []).append({"Site": site, "State": state, "probs": probs, "raw": parts[3:]})

    return {node: pd.DataFrame(rows) for node, rows in state_data.items()}


def parse_clade_assignments(assignments_path: Path) -> pd.DataFrame:
    return pd.read_csv(assignments_path)


def calculate_recent_conservation(sequences: List[str], position: int) -> float:
    if not sequences:
        return 0.0
    residues = [seq[position] for seq in sequences if len(seq) > position and seq[position] not in {"-", "."}]
    if not residues:
        return 0.0
    if len(residues) == 1:
        return 1.0
    pair_scores = []
    for i in range(len(residues)):
        for j in range(i + 1, len(residues)):
            pair_scores.append(_blosum62_pair_score(residues[i], residues[j]))
    if not pair_scores:
        return 0.0
    mean_score = float(np.mean(pair_scores))
    normalized_score = (mean_score + 4.0) / 15.0
    return float(np.clip(normalized_score, 0.0, 1.0))


def calculate_ancestral_conservation(aa1: str, aa2: str) -> float:
    return 1.0 if aa1.upper() == aa2.upper() and aa1.upper() not in {"-", "."} else -1.0


def extract_posterior_probability(state_data: Dict[str, pd.DataFrame], node: str, site: int, aa: str) -> float:
    if node not in state_data:
        return 0.0
    node_df = state_data[node]
    site_rows = node_df[node_df["Site"] == site]
    if site_rows.empty:
        return 0.0
    row = site_rows.iloc[0]
    aa_order = "ARNDCQEGHILKMFPSTWYV"
    aa_upper = aa.upper()
    if aa_upper not in aa_order:
        return 0.0
    aa_idx = aa_order.index(aa_upper)
    if "probs" in row and aa_idx < len(row["probs"]):
        return float(row["probs"][aa_idx])
    return 0.0


def _level_columns(assignments: pd.DataFrame, level: str) -> Tuple[str, str]:
    id_col = f"{level}_id"
    lca_col = f"{level}_lca_node"
    if id_col not in assignments.columns or lca_col not in assignments.columns:
        raise KeyError(f"Missing {id_col} or {lca_col} in assignments")
    return id_col, lca_col


def _resolve_hierarchical_lca_nodes(assignments: pd.DataFrame, tree: Tree) -> Dict[str, str]:
    lca_columns = [col for col in assignments.columns if col.endswith("_lca_node")]
    if not lca_columns:
        return {}

    node_members: Dict[str, List[str]] = defaultdict(list)
    for _, row in assignments.iterrows():
        sequence_id = str(row["sequence_id"])
        for col in lca_columns:
            label = str(row[col]).strip()
            if label and sequence_id not in node_members[label]:
                node_members[label].append(sequence_id)

    resolved: Dict[str, str] = {}
    for label, members in node_members.items():
        lca_node = tree.common_ancestor(members)
        if not lca_node.name:
            raise ValueError(f"LCA node for {label} has no node name in ASR tree.")
        resolved[label] = lca_node.name
    return resolved


def _pairs_within_parent(assignments: pd.DataFrame, level: str, parent_col: Optional[str] = None) -> List[Tuple[int, int]]:
    id_col, _ = _level_columns(assignments, level)
    if parent_col is None:
        ids = sorted(assignments[id_col].dropna().unique().tolist())
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((int(ids[i]), int(ids[j])))
        return pairs

    pairs: List[Tuple[int, int]] = []
    for parent_value, group in assignments[[id_col, parent_col]].drop_duplicates().groupby(parent_col):
        ids = sorted(group[id_col].dropna().unique().tolist())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((int(ids[i]), int(ids[j])))
    return sorted(set(tuple(sorted(pair)) for pair in pairs))


def build_hierarchical_sister_pairs(assignments: pd.DataFrame, tree_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    # The current hierarchy already encodes parent/child nesting, so sister-pairing is performed within each level.
    _ = tree_path  # retained for API parity and future tree-aware refinement.
    return {
        "groups": _pairs_within_parent(assignments, "group"),
        "families": _pairs_within_parent(assignments, "family", parent_col="group_id"),
        "subfamilies": _pairs_within_parent(assignments, "subfamily", parent_col="family_id"),
    }


def _compute_level_scores(
    level: str,
    alignment,
    assignments: pd.DataFrame,
    ancestral_seqs: Dict[str, str],
    state_data: Dict[str, pd.DataFrame],
    pairs: List[Tuple[int, int]],
    resolved_lca_nodes: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    id_col, lca_col = _level_columns(assignments, level)
    seq_dict = {record.id: str(record.seq) for record in alignment}
    aln_length = alignment.get_alignment_length()

    level_grouped = assignments.groupby(id_col)
    level_sequences: Dict[int, List[str]] = {}
    level_lcas: Dict[int, str] = {}
    for cluster_id, frame in level_grouped:
        seq_ids = frame["sequence_id"].tolist()
        level_sequences[int(cluster_id)] = [seq_dict[sid] for sid in seq_ids if sid in seq_dict]
        raw_lca = str(frame.iloc[0][lca_col]).strip()
        if raw_lca not in resolved_lca_nodes:
            raise KeyError(f"No resolved ASR node found for {raw_lca}.")
        level_lcas[int(cluster_id)] = resolved_lca_nodes[raw_lca]

    pairwise_scores: List[Dict[str, object]] = []
    for cluster_a, cluster_b in pairs:
        if cluster_a not in level_sequences or cluster_b not in level_sequences:
            continue
        lca_a = level_lcas[cluster_a]
        lca_b = level_lcas[cluster_b]
        if lca_a not in ancestral_seqs or lca_b not in ancestral_seqs:
            continue
        for pos in range(aln_length):
            if pos >= len(ancestral_seqs[lca_a]) or pos >= len(ancestral_seqs[lca_b]):
                continue
            aa_a = ancestral_seqs[lca_a][pos]
            aa_b = ancestral_seqs[lca_b][pos]
            if aa_a in {"-", "."} or aa_b in {"-", "."}:
                continue
            rc_a = calculate_recent_conservation(level_sequences[cluster_a], pos)
            rc_b = calculate_recent_conservation(level_sequences[cluster_b], pos)
            rc = (rc_a + rc_b) / 2.0
            ac = calculate_ancestral_conservation(aa_a, aa_b)
            p_a = extract_posterior_probability(state_data, lca_a, pos + 1, aa_a)
            p_b = extract_posterior_probability(state_data, lca_b, pos + 1, aa_b)
            p_ac = (p_a + p_b) / 2.0
            score = rc - (ac * p_ac)
            pairwise_scores.append(
                {
                    "level": level,
                    "pair": f"{cluster_a}-{cluster_b}",
                    "position": pos + 1,
                    "rc": rc,
                    "ac": ac,
                    "p_ac": p_ac,
                    "score": score,
                }
            )

    flat_scores = [entry["score"] for entry in pairwise_scores]
    global_threshold = float(np.percentile(flat_scores, 95)) if flat_scores else 0.0

    position_rows = []
    for pos in range(1, aln_length + 1):
        pos_scores = [entry["score"] for entry in pairwise_scores if entry["position"] == pos]
        max_score = max(pos_scores) if pos_scores else 0.0
        switch_count = int(sum(1 for score in pos_scores if score > global_threshold))
        position_rows.append(
            {
                "position": pos,
                "max_score": float(max_score),
                "switch_count": switch_count,
                "global_threshold": global_threshold,
                "badasp_score": float(max_score),
            }
        )

    score_df = pd.DataFrame(position_rows)
    sdp_df, threshold = identify_sdps(score_df)

    if pairwise_scores:
        max_entry = max(pairwise_scores, key=lambda row: row["score"])
        print(
            f"[{level.upper()}] Max Pos: Pos={max_entry['position']}, RC={max_entry['rc']:.6f}, AC={max_entry['ac']:.6f}, p={max_entry['p_ac']:.6f}, Score={max_entry['score']:.6f}"
        )
        print(f"[{level.upper()}] Global pooled threshold (95th percentile): {global_threshold:.6f}")

    return {
        "pairwise": pd.DataFrame(pairwise_scores),
        "scores": score_df,
        "sdps": sdp_df,
        "threshold": threshold,
        "pairs": pairs,
    }


def compute_multilevel_badasp_scores(
    alignment_path: Path,
    assignments_path: Path,
    ancestral_path: Path,
    state_path: Path,
    tree_path: Path,
    min_clade_size: int = 5,
) -> Dict[str, Dict[str, object]]:
    alignment = AlignIO.read(alignment_path, "fasta")
    assignments = pd.read_csv(assignments_path)
    ancestral_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(ancestral_path, "fasta")}
    state_data = load_state_file(state_path)
    tree = Phylo.read(str(tree_path), "newick")

    filtered = assignments.copy()
    for level in ("group", "family", "subfamily"):
        id_col, _ = _level_columns(filtered, level)
        counts = filtered.groupby(id_col).size()
        valid = counts[counts >= min_clade_size].index.tolist()
        filtered = filtered[filtered[id_col].isin(valid)]

    resolved_lca_nodes = _resolve_hierarchical_lca_nodes(filtered, tree)

    results: Dict[str, Dict[str, object]] = {}
    hierarchical_pairs = build_hierarchical_sister_pairs(filtered, tree_path)
    level_name_map = {
        "groups": "group",
        "families": "family",
        "subfamilies": "subfamily",
    }
    for level in ("groups", "families", "subfamilies"):
        result = _compute_level_scores(
            level=level_name_map[level],
            alignment=alignment,
            assignments=filtered,
            ancestral_seqs=ancestral_seqs,
            state_data=state_data,
            pairs=hierarchical_pairs[level],
            resolved_lca_nodes=resolved_lca_nodes,
        )
        results[level] = result

    return results


def identify_sdps(scores_df: pd.DataFrame, percentile: float = 95.0) -> Tuple[pd.DataFrame, float]:
    if "switch_count" in scores_df.columns:
        max_switch_count = scores_df["switch_count"].max()
        sdps = scores_df[scores_df["switch_count"] == max_switch_count].copy()
        sdps = sdps.sort_values(["switch_count", "max_score"], ascending=[False, False]).reset_index(drop=True)
        threshold = float(scores_df["global_threshold"].iloc[0]) if "global_threshold" in scores_df.columns and not scores_df.empty else float(np.percentile(scores_df["badasp_score"], percentile))
        return sdps, threshold

    threshold = float(np.percentile(scores_df["badasp_score"], percentile))
    sdps = scores_df[scores_df["badasp_score"] >= threshold].copy()
    sdps = sdps.sort_values("badasp_score", ascending=False).reset_index(drop=True)
    return sdps, threshold


class BADASPCore:
    def __init__(
        self,
        alignment_path: Path,
        assignments_path: Path,
        ancestral_path: Path,
        state_path: Path,
        tree_path: Path,
        min_clade_size: int = 5,
    ):
        self.alignment_path = Path(alignment_path)
        self.assignments_path = Path(assignments_path)
        self.ancestral_path = Path(ancestral_path)
        self.state_path = Path(state_path)
        self.tree_path = Path(tree_path)
        self.min_clade_size = min_clade_size
        self.results = None
        self.sdps = None
        self.thresholds = None

    def compute_scores(self) -> Dict[str, Dict[str, object]]:
        self.results = compute_multilevel_badasp_scores(
            alignment_path=self.alignment_path,
            assignments_path=self.assignments_path,
            ancestral_path=self.ancestral_path,
            state_path=self.state_path,
            tree_path=self.tree_path,
            min_clade_size=self.min_clade_size,
        )
        logger.info("BADASP multilevel scores computed")
        return self.results

    def identify_sdps(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
        if self.results is None:
            self.compute_scores()
        assert self.results is not None
        self.sdps = {level: payload["sdps"] for level, payload in self.results.items()}
        self.thresholds = {level: payload["threshold"] for level, payload in self.results.items()}
        return self.sdps, self.thresholds

    def save_results(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.results is None:
            raise ValueError("Run compute_scores() first")
        for level, payload in self.results.items():
            score_path = output_dir / f"badasp_scores_{level}.csv"
            payload["scores"].to_csv(score_path, index=False)
            if payload["sdps"] is not None:
                sdp_path = output_dir / f"badasp_sdps_{level}.csv"
                payload["sdps"].to_csv(sdp_path, index=False)
