"""Phase 5 duplication-directed BADASP scoring for high-confidence duplication nodes."""

from __future__ import annotations

import argparse
import csv
import logging
import warnings
from collections import defaultdict
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.BaseTree import Clade, Tree
from tqdm import tqdm

logger = logging.getLogger(__name__)

SKIPPED_AUDIT_LOG = Path("results/badasp_scoring/skipped_audit.log")


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


def load_reconciliation_events(reconciliation_csv: Optional[Path]) -> Dict[str, str]:
    if reconciliation_csv is None:
        return {}

    reconciliation_csv = Path(reconciliation_csv)
    if not reconciliation_csv.exists():
        raise FileNotFoundError(f"Reconciliation CSV not found: {reconciliation_csv}")

    reconciliation_df = pd.read_csv(reconciliation_csv)
    required_columns = {"node_name", "event_type"}
    missing = required_columns - set(reconciliation_df.columns)
    if missing:
        raise ValueError(f"Reconciliation CSV is missing required columns: {sorted(missing)}")

    events: Dict[str, str] = {}
    for _, row in reconciliation_df.iterrows():
        node_name = row["node_name"]
        event_type = row["event_type"]
        if pd.isna(node_name) or pd.isna(event_type):
            continue
        node_label = str(node_name).strip()
        event_label = str(event_type).strip()
        if node_label:
            events[node_label] = event_label
    return events


def _leaf_signature(node) -> Tuple[str, ...]:
    return tuple(sorted(terminal.name for terminal in node.get_terminals() if terminal.name))


def _build_named_node_signatures(tree: Tree) -> Dict[str, Tuple[str, ...]]:
    signatures: Dict[str, Tuple[str, ...]] = {}
    for node in tree.get_nonterminals(order="level"):
        if not node.name:
            continue
        signature = _leaf_signature(node)
        if signature:
            signatures[str(node.name)] = signature
    return signatures


def _remap_reconciliation_events_to_asr_nodes(
    reconciliation_events: Dict[str, str],
    topology_tree: Tree,
    asr_tree: Tree,
) -> Dict[str, str]:
    topology_signatures = _build_named_node_signatures(topology_tree)
    asr_signature_to_name = {
        signature: node_name for node_name, signature in _build_named_node_signatures(asr_tree).items()
    }

    remapped: Dict[str, str] = {}
    unmapped_topology_nodes: List[str] = []

    for topology_node_name, event_type in reconciliation_events.items():
        topology_signature = topology_signatures.get(topology_node_name)
        if topology_signature is None:
            unmapped_topology_nodes.append(topology_node_name)
            continue

        asr_node_name = asr_signature_to_name.get(topology_signature)
        if asr_node_name is None:
            unmapped_topology_nodes.append(topology_node_name)
            continue

        previous = remapped.get(asr_node_name)
        if previous is None:
            remapped[asr_node_name] = event_type
            continue

        if previous != event_type:
            remapped[asr_node_name] = "Speciation" if "Speciation" in {previous, event_type} else event_type

    if unmapped_topology_nodes:
        warnings.warn(
            f"Unmapped reconciliation nodes: {len(unmapped_topology_nodes)}. "
            f"Examples: {unmapped_topology_nodes[:5]}",
            UserWarning,
            stacklevel=2,
        )

    return remapped


def _filter_pairs_by_reconciliation(
    pairs: List[Tuple[int, int]],
    level_lcas: Dict[int, str],
    reconciliation_events: Dict[str, str],
) -> Tuple[List[Tuple[int, int]], int, int]:
    if not reconciliation_events:
        return pairs, 0, 0

    lca_values = set(level_lcas.values())
    recon_keys = set(reconciliation_events.keys())
    matching_nodes = lca_values & recon_keys
    if not matching_nodes:
        raise ValueError(
            f"No reconciliation node names matched ASR LCA nodes. ASR sample: {sorted(lca_values)[:3]}, "
            f"Reconciliation sample: {sorted(recon_keys)[:3]}"
        )
        return pairs, 0, 0

    kept_pairs: List[Tuple[int, int]] = []
    skipped_pairs = 0
    skipped_speciation_pairs = 0
    for cluster_a, cluster_b in pairs:
        lca_a = level_lcas[cluster_a]
        lca_b = level_lcas[cluster_b]
        event_a = reconciliation_events.get(lca_a, "Duplication")
        event_b = reconciliation_events.get(lca_b, "Duplication")
        if event_a != "Duplication" or event_b != "Duplication":
            skipped_pairs += 1
            if event_a == "Speciation" or event_b == "Speciation":
                skipped_speciation_pairs += 1
            continue
        kept_pairs.append((cluster_a, cluster_b))
    return kept_pairs, skipped_pairs, skipped_speciation_pairs


def _validate_lca_coverage(level: str, level_lcas: Dict[int, str], ancestral_seqs: Dict[str, str], min_coverage: float = 0.95) -> None:
    if not level_lcas:
        raise ValueError(f"No LCA nodes were resolved for {level}.")

    missing = [node_id for node_id in level_lcas.values() if node_id not in ancestral_seqs]
    coverage = 1.0 - (len(missing) / float(len(level_lcas)))
    if missing:
        message = (
            f"{level.capitalize()} ASR coverage is {coverage:.2%} ({len(missing)}/{len(level_lcas)} LCA nodes missing from ancestral FASTA)."
        )
        if coverage < min_coverage:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=2)


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

    counts = Counter(residues)
    residue_types = list(counts)
    total_pairs = (len(residues) * (len(residues) - 1)) // 2
    if total_pairs == 0:
        return 0.0

    weighted_pair_sum = 0.0
    for i, aa_i in enumerate(residue_types):
        count_i = counts[aa_i]

        # Same-residue pairs contribute nC2 pairs at the diagonal substitution score.
        same_pairs = (count_i * (count_i - 1)) // 2
        if same_pairs:
            weighted_pair_sum += same_pairs * _blosum62_pair_score(aa_i, aa_i)

        for aa_j in residue_types[i + 1 :]:
            weighted_pair_sum += (count_i * counts[aa_j]) * _blosum62_pair_score(aa_i, aa_j)

    mean_score = weighted_pair_sum / float(total_pairs)
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


def _reconstruct_ancestral_sequence_from_state(state_data: Dict[str, pd.DataFrame], node: str) -> Optional[str]:
    if node not in state_data:
        return None

    node_df = state_data[node]
    if node_df.empty or "Site" not in node_df.columns or "State" not in node_df.columns:
        return None

    sequence_parts: List[str] = []
    for _, row in node_df.sort_values("Site").iterrows():
        state = str(row["State"]).strip()
        if not state:
            sequence_parts.append("X")
            continue
        sequence_parts.append(state[0])
    return "".join(sequence_parts)


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

    terminal_names = {terminal.name for terminal in tree.get_terminals() if terminal.name}
    node_members: Dict[str, set] = defaultdict(set)
    for _, row in assignments.iterrows():
        sequence_id = str(row["sequence_id"])
        for col in lca_columns:
            raw_label = row[col]
            if pd.isna(raw_label):
                continue
            label = str(raw_label).strip()
            if label:
                node_members[label].add(sequence_id)

    resolved: Dict[str, str] = {}
    for label, members in node_members.items():
        present_members = [member for member in members if member in terminal_names]
        if not present_members:
            raise ValueError(f"No members for {label} were found in the ASR tree.")
        lca_node = tree.common_ancestor(present_members)
        if not lca_node.name:
            raise ValueError(f"LCA node for {label} has no node name in ASR tree.")
        resolved[label] = lca_node.name
    return resolved


def _nearest_sister_pairs_for_level(
    assignments: pd.DataFrame,
    level: str,
    tree: Tree,
    parent_col: Optional[str] = None,
) -> List[Tuple[int, int]]:
    id_col, _ = _level_columns(assignments, level)
    members_by_cluster = assignments.groupby(id_col)["sequence_id"].apply(list).to_dict()

    cluster_to_parent: Dict[int, Optional[int]] = {}
    if parent_col is None:
        for cluster_id in members_by_cluster:
            cluster_to_parent[int(cluster_id)] = None
    else:
        parent_lookup = assignments[[id_col, parent_col]].drop_duplicates().set_index(id_col)[parent_col].to_dict()
        for cluster_id in members_by_cluster:
            cluster_to_parent[int(cluster_id)] = int(parent_lookup[cluster_id])

    pairs = set()
    cluster_ids = sorted(int(cid) for cid in members_by_cluster.keys())

    cluster_nodes: Dict[int, Clade] = {}
    for cluster_id in tqdm(cluster_ids, desc=f"Resolving {level} LCAs", unit="cluster"):
        cluster_nodes[cluster_id] = tree.common_ancestor(members_by_cluster[cluster_id])

    for cluster_id in tqdm(cluster_ids, desc=f"Pairing {level}s", unit="cluster"):
        parent_value = cluster_to_parent[cluster_id]
        candidates = [
            other_id
            for other_id in cluster_ids
            if other_id != cluster_id and cluster_to_parent[other_id] == parent_value
        ]
        if not candidates:
            continue

        cluster_node = cluster_nodes[cluster_id]
        nearest_id = None
        nearest_distance = float("inf")
        for other_id in candidates:
            other_node = cluster_nodes[other_id]
            distance = float(tree.distance(cluster_node, other_node))
            if distance < nearest_distance or (
                distance == nearest_distance and (nearest_id is None or other_id < nearest_id)
            ):
                nearest_distance = distance
                nearest_id = other_id

        if nearest_id is not None:
            pairs.add(tuple(sorted((cluster_id, nearest_id))))

    return sorted(pairs)


def build_duplication_sister_pairs(
    tree: Tree,
    reconciliation_events: Dict[str, str],
    min_clade_size: int = 5,
) -> List[Tuple[str, str, str]]:
    node_lookup = {node.name: node for node in tree.get_nonterminals() if node.name}
    duplication_pairs: List[Tuple[str, str, str]] = []

    for duplication_node in tqdm(
        sorted(node_name for node_name, event_type in reconciliation_events.items() if event_type == "Duplication"),
        desc="Resolving duplication nodes",
        unit="node",
    ):
        node = node_lookup.get(duplication_node)
        if node is None or len(node.clades) != 2:
            continue

        left, right = node.clades
        left_name = getattr(left, "name", None)
        right_name = getattr(right, "name", None)
        if not left_name or not right_name:
            continue

        if len(left.get_terminals()) < min_clade_size or len(right.get_terminals()) < min_clade_size:
            continue

        duplication_pairs.append((str(duplication_node), str(left_name), str(right_name)))

    return duplication_pairs


def compute_multilevel_badasp_scores(
    alignment_path: Path,
    assignments_path: Path,
    ancestral_path: Path,
    state_path: Path,
    tree_path: Path,
    min_clade_size: int = 5,
    reconciliation_csv: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    alignment = AlignIO.read(alignment_path, "fasta")
    ancestral_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(ancestral_path, "fasta")}
    state_data = load_state_file(state_path)
    topology_tree = Phylo.read(str(tree_path), "newick")
    reconciliation_events = load_reconciliation_events(reconciliation_csv)

    asr_tree = topology_tree
    asr_tree_path = Path(state_path).with_suffix(".treefile")
    if asr_tree_path.exists():
        asr_tree = Phylo.read(str(asr_tree_path), "newick")

    if reconciliation_events:
        asr_node_names = {node.name for node in asr_tree.get_nonterminals() if node.name}
        if not (set(reconciliation_events) & asr_node_names):
            remapped_reconciliation_events = _remap_reconciliation_events_to_asr_nodes(
                reconciliation_events,
                topology_tree,
                asr_tree,
            )
            reconciliation_events = {**reconciliation_events, **remapped_reconciliation_events}

    duplication_pairs = build_duplication_sister_pairs(asr_tree, reconciliation_events, min_clade_size=min_clade_size)
    node_lookup = {node.name: node for node in asr_tree.get_nonterminals() if node.name}
    seq_dict = {record.id: str(record.seq) for record in alignment}
    aln_length = alignment.get_alignment_length()

    pairwise_columns = ["duplication_node", "left_child", "right_child", "pair", "position", "rc", "ac", "p_ac", "score"]
    pair_col: List[str] = []
    dup_node_col: List[str] = []
    left_col: List[str] = []
    right_col: List[str] = []
    pos_col: List[int] = []
    rc_col: List[float] = []
    ac_col: List[float] = []
    p_ac_col: List[float] = []
    score_col: List[float] = []
    position_max_scores = np.full(aln_length, -np.inf, dtype=float)
    rc_cache: Dict[Tuple[str, int], float] = {}
    posterior_cache: Dict[Tuple[str, int, str], float] = {}
    skipped_missing_nodes = 0
    reconstructed_sequences: Dict[str, str] = {}

    def _ancestral_sequence(node_name: str) -> Optional[str]:
        if node_name in ancestral_seqs:
            return ancestral_seqs[node_name]
        if node_name not in reconstructed_sequences:
            reconstructed_sequences[node_name] = _reconstruct_ancestral_sequence_from_state(state_data, node_name) or ""
        return reconstructed_sequences[node_name] or None

    def _rc(node_name: str, pos: int, sequences: List[str]) -> float:
        key = (node_name, pos)
        if key not in rc_cache:
            rc_cache[key] = calculate_recent_conservation(sequences, pos)
        return rc_cache[key]

    def _posterior(node: str, site: int, aa: str) -> float:
        key = (node, site, aa)
        if key not in posterior_cache:
            posterior_cache[key] = extract_posterior_probability(state_data, node, site, aa)
        return posterior_cache[key]

    for duplication_node, left_name, right_name in tqdm(duplication_pairs, desc="Scoring duplication pairs", unit="pair"):
        left_node = node_lookup.get(left_name)
        right_node = node_lookup.get(right_name)
        if left_node is None or right_node is None:
            skipped_missing_nodes += 1
            continue

        left_sequence = _ancestral_sequence(left_name)
        right_sequence = _ancestral_sequence(right_name)
        missing_nodes = [node for node, sequence in ((left_name, left_sequence), (right_name, right_sequence)) if not sequence]
        if missing_nodes:
            skipped_missing_nodes += 1
            continue

        left_sequences = [seq_dict[leaf.name] for leaf in left_node.get_terminals() if leaf.name in seq_dict]
        right_sequences = [seq_dict[leaf.name] for leaf in right_node.get_terminals() if leaf.name in seq_dict]
        if not left_sequences or not right_sequences:
            skipped_missing_nodes += 1
            continue

        for pos in range(aln_length):
            if pos >= len(left_sequence) or pos >= len(right_sequence):
                continue

            aa_left = left_sequence[pos]
            aa_right = right_sequence[pos]
            if aa_left in {"-", "."} or aa_right in {"-", "."}:
                continue

            rc_left = _rc(left_name, pos, left_sequences)
            rc_right = _rc(right_name, pos, right_sequences)
            rc = (rc_left + rc_right) / 2.0
            ac = calculate_ancestral_conservation(aa_left, aa_right)
            p_left = _posterior(left_name, pos + 1, aa_left)
            p_right = _posterior(right_name, pos + 1, aa_right)
            p_ac = (p_left + p_right) / 2.0
            score = rc - (ac * p_ac)

            site = pos + 1
            dup_node_col.append(duplication_node)
            left_col.append(left_name)
            right_col.append(right_name)
            pair_col.append(f"{left_name}-{right_name}")
            pos_col.append(site)
            rc_col.append(rc)
            ac_col.append(ac)
            p_ac_col.append(p_ac)
            score_col.append(score)
            if score > position_max_scores[pos]:
                position_max_scores[pos] = score

    flat_scores = [score for score in score_col if score > 0]
    global_threshold = float(np.percentile(flat_scores, 95)) if flat_scores else 0.0

    switch_counts = np.zeros(aln_length, dtype=int)
    for idx, score in enumerate(score_col):
        if score > global_threshold:
            switch_counts[pos_col[idx] - 1] += 1

    position_rows = []
    for pos in range(1, aln_length + 1):
        max_score = float(position_max_scores[pos - 1]) if np.isfinite(position_max_scores[pos - 1]) else 0.0
        switch_count = int(switch_counts[pos - 1])
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

    if score_col:
        max_idx = int(np.argmax(score_col))
        print(
            f"[DUPLICATIONS] Max Pos: Pos={pos_col[max_idx]}, RC={rc_col[max_idx]:.6f}, AC={ac_col[max_idx]:.6f}, p={p_ac_col[max_idx]:.6f}, Score={score_col[max_idx]:.6f}"
        )
        print(f"[DUPLICATIONS] Global pooled threshold (95th percentile over score>0): {global_threshold:.6f}")
    if skipped_missing_nodes:
        print(f"[DUPLICATIONS] Skipped {skipped_missing_nodes} duplication pairs due to missing child nodes or ancestral sequences.")

    pairwise_df = pd.DataFrame(
        {
            "duplication_node": dup_node_col,
            "left_child": left_col,
            "right_child": right_col,
            "pair": pair_col,
            "position": pos_col,
            "rc": rc_col,
            "ac": ac_col,
            "p_ac": p_ac_col,
            "score": score_col,
        },
        columns=pairwise_columns,
    )

    return {
        "duplications": {
            "pairwise": pairwise_df,
            "scores": score_df,
            "sdps": sdp_df,
            "threshold": threshold,
            "pairs": duplication_pairs,
            "filtered_pairs": skipped_missing_nodes,
            "filtered_speciation_pairs": 0,
            "candidate_pairs": len(duplication_pairs),
        }
    }


def identify_sdps(scores_df: pd.DataFrame, percentile: float = 95.0) -> Tuple[pd.DataFrame, float]:
    if "switch_count" in scores_df.columns:
        max_switch_count = scores_df["switch_count"].max()
        if scores_df.empty or max_switch_count <= 0:
            threshold = float(scores_df["global_threshold"].iloc[0]) if "global_threshold" in scores_df.columns and not scores_df.empty else 0.0
            empty = scores_df.iloc[0:0].copy()
            return empty, threshold
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
        reconciliation_csv: Optional[Path] = None,
    ):
        self.alignment_path = Path(alignment_path)
        self.assignments_path = Path(assignments_path)
        self.ancestral_path = Path(ancestral_path)
        self.state_path = Path(state_path)
        self.tree_path = Path(tree_path)
        self.min_clade_size = min_clade_size
        self.reconciliation_csv = Path(reconciliation_csv) if reconciliation_csv is not None else None
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
            reconciliation_csv=self.reconciliation_csv,
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
        payload = self.results.get("duplications")
        if payload is None:
            raise KeyError("Missing duplications payload in BADASP results")

        payload["scores"].to_csv(output_dir / "badasp_scores_duplications.csv", index=False)
        if payload["sdps"] is not None:
            payload["sdps"].to_csv(output_dir / "badasp_sdps_duplications.csv", index=False)
        payload["pairwise"].to_csv(output_dir / "raw_pairwise_duplications.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 multi-level BADASP scoring")
    parser.add_argument("--alignment", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--assignments", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--ancestral", default="data/interim/ancestral_sequences.fasta")
    parser.add_argument("--state", default="data/interim/asr_run.state")
    parser.add_argument("--tree", default="results/topological_clustering/mad_rooted.tree")
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--output-dir", default="results/badasp_scoring")
    parser.add_argument("--reconciliation-csv", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    core = BADASPCore(
        alignment_path=Path(args.alignment),
        assignments_path=Path(args.assignments),
        ancestral_path=Path(args.ancestral),
        state_path=Path(args.state),
        tree_path=Path(args.tree),
        min_clade_size=args.min_clade_size,
        reconciliation_csv=Path(args.reconciliation_csv) if args.reconciliation_csv else None,
    )
    core.compute_scores()
    core.identify_sdps()
    core.save_results(Path(args.output_dir))

    if core.results is not None:
        payload = core.results.get("duplications", {})
        kept_pairs = len(payload.get("pairs", []))
        filtered_pairs = int(payload.get("filtered_pairs", 0))
        threshold = float(payload.get("threshold", 0.0))
        print(f"Scored {kept_pairs} duplication pairs and filtered {filtered_pairs} candidates.")
        print(f"Global SDP threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
