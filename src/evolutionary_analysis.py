from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Phylo, SeqIO
from Bio.PDB import PDBParser
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from src.pdb_mapper import PDBMapper


LEVELS = ["groups", "families", "subfamilies"]
LEVEL_MAP = {
    "groups": "group",
    "families": "family",
    "subfamilies": "subfamily",
}


def _ensure_node_names(tree) -> None:
    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"InternalNode_{idx}"


def calculate_lca_depth(tree_path: Path, member_names: Sequence[str]) -> float:
    tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(tree)
    lca = tree.common_ancestor(list(member_names))
    return float(tree.distance(tree.root, lca))


def calculate_ca_distance_matrix(pdb_path: Path, residue_numbers: Sequence[int]) -> pd.DataFrame:
    structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))

    ca_coords: Dict[int, np.ndarray] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                resseq = int(residue.id[1])
                if resseq not in residue_numbers:
                    continue
                if "CA" not in residue:
                    continue
                ca_coords[resseq] = residue["CA"].coord
            if ca_coords:
                break
        if ca_coords:
            break

    ordered = [int(r) for r in residue_numbers if int(r) in ca_coords]
    matrix = pd.DataFrame(index=ordered, columns=ordered, dtype=float)
    for left in ordered:
        for right in ordered:
            matrix.loc[left, right] = float(np.linalg.norm(ca_coords[left] - ca_coords[right]))
    return matrix


def compute_coevolution_matrix(events_df: pd.DataFrame) -> pd.DataFrame:
    required = {"branch_id", "position"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    branch_sets: Dict[int, set] = {}
    for position, group in events_df.groupby("position"):
        branch_sets[int(position)] = set(group["branch_id"].astype(str))

    positions = sorted(branch_sets)
    matrix = pd.DataFrame(index=positions, columns=positions, dtype=float)
    for left in positions:
        for right in positions:
            if left == right:
                matrix.loc[left, right] = 1.0
                continue
            union = branch_sets[left] | branch_sets[right]
            intersection = branch_sets[left] & branch_sets[right]
            matrix.loc[left, right] = float(len(intersection) / len(union)) if union else 0.0
    return matrix


def classify_physicochemical_shift(
    charge_change: str,
    hydrophobicity_change: str,
    volume_delta: float,
    volume_threshold: float = 45.0,
) -> str:
    charge_parts = str(charge_change).split("->")
    hydro_parts = str(hydrophobicity_change).split("->")
    charge_shift = len(charge_parts) == 2 and charge_parts[0] != charge_parts[1]
    hydro_shift = len(hydro_parts) == 2 and hydro_parts[0] != hydro_parts[1]
    size_shift = bool(pd.notna(volume_delta)) and abs(float(volume_delta)) >= float(volume_threshold)

    shift_count = int(charge_shift) + int(hydro_shift) + int(size_shift)
    if shift_count >= 2:
        return "multiple_complex"
    if charge_shift:
        return "charge_shift"
    if hydro_shift:
        return "hydrophobicity_shift"
    if size_shift:
        return "size_shift"
    return "none"


def rank_top_functional_sdps(
    subfamily_scores_df: pd.DataFrame,
    coevolution_matrix_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    if subfamily_scores_df.empty:
        return pd.DataFrame(
            columns=[
                "position",
                "switch_count",
                "max_score",
                "mean_coevolution",
                "major_transition_count",
                "shift_type",
                "functional_sdp_score",
            ]
        )

    scores = subfamily_scores_df[["position", "switch_count", "max_score"]].copy()
    scores["position"] = scores["position"].astype(int)

    coevo_strength: Dict[int, float] = {}
    if not coevolution_matrix_df.empty:
        for pos in coevolution_matrix_df.index:
            row = coevolution_matrix_df.loc[pos].drop(labels=[pos], errors="ignore")
            coevo_strength[int(pos)] = float(row.mean()) if not row.empty else 0.0
    scores["mean_coevolution"] = scores["position"].map(coevo_strength).fillna(0.0)

    shifts = shifts_df.copy()
    if shifts.empty:
        shifts = pd.DataFrame(columns=["position", "major_transition_count", "charge_change", "hydrophobicity_change", "volume_change"])
    shifts["position"] = shifts["position"].astype(int)
    shifts["shift_type"] = shifts.apply(
        lambda r: classify_physicochemical_shift(r["charge_change"], r["hydrophobicity_change"], r["volume_change"]),
        axis=1,
    )
    shifts["shift_strength"] = (
        shifts["major_transition_count"].fillna(0).astype(float)
        + shifts["volume_change"].abs().fillna(0).astype(float) / 25.0
        + shifts["shift_type"].isin(["multiple_complex", "charge_shift", "hydrophobicity_shift", "size_shift"]).astype(float)
    )

    merged = scores.merge(
        shifts[["position", "major_transition_count", "shift_type", "shift_strength"]],
        on="position",
        how="left",
    )
    merged["major_transition_count"] = merged["major_transition_count"].fillna(0.0)
    merged["shift_type"] = merged["shift_type"].fillna("none")
    merged["shift_strength"] = merged["shift_strength"].fillna(0.0)

    def _minmax(series: pd.Series) -> pd.Series:
        min_val = float(series.min())
        max_val = float(series.max())
        if max_val - min_val < 1e-12:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - min_val) / (max_val - min_val)

    merged["switch_norm"] = _minmax(merged["switch_count"].astype(float))
    merged["coevo_norm"] = _minmax(merged["mean_coevolution"].astype(float))
    merged["shift_norm"] = _minmax(merged["shift_strength"].astype(float))
    merged["functional_sdp_score"] = 0.45 * merged["switch_norm"] + 0.35 * merged["coevo_norm"] + 0.20 * merged["shift_norm"]

    cols = [
        "position",
        "switch_count",
        "max_score",
        "mean_coevolution",
        "major_transition_count",
        "shift_type",
        "functional_sdp_score",
    ]
    ranked = merged.sort_values(["functional_sdp_score", "switch_count", "max_score"], ascending=[False, False, False])[cols]
    return ranked.head(top_n).reset_index(drop=True)


def count_switches_per_domain(events_df: pd.DataFrame, domain_arch: Dict[str, Sequence[int]]) -> Dict[str, int]:
    counts: Dict[str, int] = {domain: 0 for domain in domain_arch}
    if events_df.empty:
        return counts

    positions = events_df["position"].astype(int).tolist()
    for pos in positions:
        for domain, span in domain_arch.items():
            start, end = int(span[0]), int(span[1])
            if start <= pos <= end:
                counts[domain] += 1
    return counts


def assign_coevolution_communities(matrix: pd.DataFrame, distance_cut: float = 0.6) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame(columns=["position", "community_id"])

    positions = [int(pos) for pos in matrix.index.tolist()]
    if len(positions) == 1:
        return pd.DataFrame({"position": positions, "community_id": [1]})

    dist_matrix = 1.0 - matrix.to_numpy(dtype=float)
    np.fill_diagonal(dist_matrix, 0.0)
    condensed = squareform(dist_matrix, checks=False)
    linkage = hierarchy.linkage(condensed, method="average")
    labels = hierarchy.fcluster(linkage, t=float(distance_cut), criterion="distance")
    return pd.DataFrame({"position": positions, "community_id": labels.astype(int)})


def extract_taxon_label(header: str) -> str:
    os_match = re.search(r"OS=([^=]+?)(?:\sOX=|\sGN=|\sPE=|\sSV=|$)", header)
    if os_match:
        return os_match.group(1).strip()

    bracket_match = re.search(r"\[([^\]]+)\]", header)
    if bracket_match:
        return bracket_match.group(1).strip()

    tokens = str(header).split("|")
    if len(tokens) >= 3 and tokens[2].strip():
        return tokens[2].strip().split()[0]

    words = str(header).split()
    if words:
        return words[0].strip()
    return "Unknown"


def _plot_clustered_heatmap(
    matrix: pd.DataFrame,
    output_svg: Path,
    title: str,
    cmap: str,
    cbar_label: Optional[str] = None,
    is_distance_matrix: bool = False,
) -> None:
    if matrix.empty:
        return

    values = matrix.to_numpy(dtype=float)
    if len(matrix.index) > 1:
        if is_distance_matrix:
            condensed = squareform(values, checks=False)
        else:
            condensed = pdist(values, metric="euclidean")
        linkage = hierarchy.linkage(condensed, method="average")
    else:
        linkage = None

    cluster = sns.clustermap(
        matrix,
        cmap=cmap,
        linewidths=0.0,
        figsize=(10, 9),
        row_cluster=True,
        col_cluster=True,
        row_linkage=linkage,
        col_linkage=linkage,
        cbar_kws={"label": cbar_label} if cbar_label else None,
    )
    cluster.fig.suptitle(title, y=1.02)
    cluster.fig.savefig(output_svg, format="svg")
    plt.close(cluster.fig)


def _plot_switch_timeline(events_df: pd.DataFrame, output_svg: Path) -> None:
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    palette = {"groups": "#1F77B4", "families": "#D95F02", "subfamilies": "#2CA02C"}

    plt.figure(figsize=(11, 6))
    sns.histplot(
        data=events_df,
        x="root_distance",
        hue="level",
        bins=40,
        stat="count",
        element="step",
        fill=False,
        common_bins=True,
        palette=palette,
    )
    plt.xlabel("Distance from Root")
    plt.ylabel("Switch Frequency")
    plt.title("Evolutionary Timeline of BADASP Switch Events")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def _plot_architecture_distribution(counts: Dict[str, int], output_svg: Path, level: str) -> None:
    rows: List[dict] = []
    for domain, count in counts.items():
        width = 1
        density = count / width
        if isinstance(count, (int, float)):
            density = float(count)
        rows.append({"domain": domain, "switch_count": int(count), "switch_density": density})

    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="domain", y="switch_count", color="#4C78A8")
    plt.ylabel("Switch Count")
    plt.xlabel("Domain")
    plt.title(f"Architectural Switch Distribution ({level.capitalize()})")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def _top_correlated_pairs(matrix: pd.DataFrame, top_n: int = 5) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []
    for i, left in enumerate(matrix.index.tolist()):
        for right in matrix.columns.tolist()[i + 1 :]:
            pairs.append((int(left), int(right), float(matrix.loc[left, right])))
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]


def _mean_upper_triangle(matrix: pd.DataFrame) -> float:
    if matrix.empty:
        return float("nan")
    arr = matrix.to_numpy(dtype=float)
    tri = arr[np.triu_indices(arr.shape[0], k=1)]
    if tri.size == 0:
        return float("nan")
    return float(np.nanmean(tri))


def _load_switch_events_for_level(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_path: Path,
    level: str,
) -> pd.DataFrame:
    singular = LEVEL_MAP[level]
    id_col = f"{singular}_id"

    tree = Phylo.read(str(tree_path), "newick")
    _ensure_node_names(tree)
    assignments = pd.read_csv(assignments_path)
    cluster_members = assignments.groupby(id_col)["sequence_id"].apply(list).to_dict()

    raw = pd.read_csv(raw_pairwise_path)
    threshold = float(np.percentile(raw["score"].astype(float), 95)) if not raw.empty else 0.0
    switched = raw[raw["score"] > threshold].copy()

    rows: List[dict] = []
    for _, row in switched.iterrows():
        try:
            left_str, right_str = str(row["pair"]).split("-")
            left_id = int(left_str)
            right_id = int(right_str)
        except ValueError:
            continue

        if left_id not in cluster_members or right_id not in cluster_members:
            continue

        members = list(cluster_members[left_id]) + list(cluster_members[right_id])
        lca = tree.common_ancestor(members)
        depth = float(tree.distance(tree.root, lca))
        rows.append(
            {
                "level": level,
                "pair": str(row["pair"]),
                "position": int(row["position"]),
                "score": float(row["score"]),
                "branch_id": str(lca.name),
                "root_distance": depth,
            }
        )

    return pd.DataFrame(rows)


def _consensus_sequence(msa_path: Path) -> str:
    records = list(SeqIO.parse(str(msa_path), "fasta"))
    aln_len = len(str(records[0].seq))
    consensus_chars: List[str] = []
    for i in range(aln_len):
        residues = [str(rec.seq)[i] for rec in records if str(rec.seq)[i] != "-"]
        consensus_chars.append(Counter(residues).most_common(1)[0][0] if residues else "X")
    return "".join(consensus_chars)


def _safe_window(sequence: str, center_1based: int, width: int = 15) -> str:
    idx = max(0, center_1based - 1)
    half = width // 2
    start = max(0, idx - half)
    end = min(len(sequence), idx + half + 1)
    window = sequence[start:end]
    if len(window) < 7:
        return sequence[max(0, idx - 3): min(len(sequence), idx + 4)]
    return window


def _run_tu_json(args: Sequence[str]) -> dict:
    proc = subprocess.run(list(args), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ToolUniverse command failed: {' '.join(args)}\\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def _tu_literature_hits(query: str, max_hits: int = 5) -> List[dict]:
    tools = _run_tu_json(["tu", "find", "pubmed pmc europe pmc literature search", "--json"]).get("tools", [])
    literature_tools = [tool.get("name") for tool in tools if tool.get("name")]

    payload_templates = [
        {"query": query, "max_results": max_hits},
        {"term": query, "max_results": max_hits},
        {"search_query": query, "max_results": max_hits},
    ]
    for tool_name in literature_tools:
        for payload in payload_templates:
            try:
                result = _run_tu_json(["tu", "run", tool_name, json.dumps(payload), "--json"])
            except RuntimeError:
                continue
            data = result.get("data", result)
            if isinstance(data, list) and data:
                return data[:max_hits]
            if isinstance(data, dict):
                for key in ("results", "articles", "items"):
                    value = data.get(key)
                    if isinstance(value, list) and value:
                        return value[:max_hits]
    return []


def _protparam(sequence: str) -> dict:
    payload = json.dumps({"sequence": sequence})
    result = _run_tu_json(["tu", "run", "ProtParam_calculate", payload, "--json"])
    if result.get("status") == "success":
        return result.get("data", {})
    return result


def _charge_class(residue: str) -> str:
    positive = {"K", "R", "H"}
    negative = {"D", "E"}
    if residue in positive:
        return "positive"
    if residue in negative:
        return "negative"
    return "neutral"


def _hydrophobic_class(residue: str) -> str:
    hydrophobic = {"A", "V", "I", "L", "M", "F", "W", "Y", "P"}
    return "hydrophobic" if residue in hydrophobic else "polar"


def _volume(residue: str) -> float:
    volumes = {
        "A": 88.6,
        "R": 173.4,
        "N": 114.1,
        "D": 111.1,
        "C": 108.5,
        "Q": 143.8,
        "E": 138.4,
        "G": 60.1,
        "H": 153.2,
        "I": 166.7,
        "L": 166.7,
        "K": 168.6,
        "M": 162.9,
        "F": 189.9,
        "P": 112.7,
        "S": 89.0,
        "T": 116.1,
        "W": 227.8,
        "Y": 193.6,
        "V": 140.0,
    }
    return float(volumes.get(residue, np.nan))


def _compute_level_physicochemical_shifts(
    level: str,
    level_scores: pd.DataFrame,
    assignments: pd.DataFrame,
    lca_to_asr: Dict[str, str],
    ancestral_records: Dict[str, str],
    alignment_records: Dict[str, str],
    consensus: str,
    tu_tools: List[str],
) -> pd.DataFrame:
    singular = LEVEL_MAP[level]
    id_col = f"{singular}_id"
    lca_col = f"{singular}_lca_node"

    grouped = assignments.groupby(id_col)
    top_positions = (
        level_scores[level_scores["switch_count"] > 0]
        .sort_values(["switch_count", "max_score"], ascending=[False, False])["position"]
        .astype(int)
        .head(10)
        .tolist()
    )

    rows: List[dict] = []
    for pos in top_positions:
        transitions = Counter()
        for _, cluster_df in grouped:
            lca_node = str(cluster_df[lca_col].iloc[0])
            lca_node = lca_to_asr.get(lca_node, lca_node)
            if lca_node not in ancestral_records:
                continue
            anc_seq = ancestral_records[lca_node]
            if pos - 1 >= len(anc_seq):
                continue
            anc_aa = anc_seq[pos - 1]
            recent_residues: List[str] = []
            for seq_id in cluster_df["sequence_id"].tolist():
                seq = alignment_records.get(seq_id)
                if seq is None or pos - 1 >= len(seq):
                    continue
                aa = seq[pos - 1]
                if aa != "-":
                    recent_residues.append(aa)
            if not recent_residues:
                continue
            recent_aa = Counter(recent_residues).most_common(1)[0][0]
            if anc_aa == "-" or anc_aa == recent_aa:
                continue
            transitions[(anc_aa, recent_aa)] += 1

        if not transitions:
            continue

        (anc_aa, recent_aa), n_switches = transitions.most_common(1)[0]
        window = _safe_window(consensus, pos, width=15)
        center = min(len(window) - 1, len(window) // 2)
        anc_window = window[:center] + anc_aa + window[center + 1 :]
        recent_window = window[:center] + recent_aa + window[center + 1 :]
        anc_props = _protparam(anc_window)
        recent_props = _protparam(recent_window)

        rows.append(
            {
                "level": level,
                "position": pos,
                "ancestral_aa": anc_aa,
                "recent_aa": recent_aa,
                "major_transition_count": int(n_switches),
                "charge_change": f"{_charge_class(anc_aa)}->{_charge_class(recent_aa)}",
                "hydrophobicity_change": f"{_hydrophobic_class(anc_aa)}->{_hydrophobic_class(recent_aa)}",
                "volume_change": float(_volume(recent_aa) - _volume(anc_aa)),
                "tu_tool_used": "ProtParam_calculate",
                "tu_related_tools": ";".join(tu_tools),
                "tu_gravy_ancestral": anc_props.get("gravy", np.nan),
                "tu_gravy_recent": recent_props.get("gravy", np.nan),
                "tu_delta_gravy": float(recent_props.get("gravy", 0.0) - anc_props.get("gravy", 0.0)),
                "tu_pI_ancestral": anc_props.get("isoelectric_point", np.nan),
                "tu_pI_recent": recent_props.get("isoelectric_point", np.nan),
                "tu_delta_pI": float(recent_props.get("isoelectric_point", 0.0) - anc_props.get("isoelectric_point", 0.0)),
            }
        )

    return pd.DataFrame(rows)


def _collect_taxonomic_distribution(
    level: str,
    assignments: pd.DataFrame,
    descriptions: Dict[str, str],
    top_positions: Sequence[int],
    alignment_records: Dict[str, str],
) -> pd.DataFrame:
    rows: List[dict] = []
    singular = LEVEL_MAP[level]
    id_col = f"{singular}_id"

    seq_taxon_rows: List[Tuple[str, str]] = []
    for seq_id in assignments["sequence_id"].tolist():
        desc = descriptions.get(seq_id, seq_id)
        seq_taxon_rows.append((seq_id, extract_taxon_label(desc)))
    seq_taxon = pd.DataFrame(seq_taxon_rows, columns=["sequence_id", "taxon"]) if seq_taxon_rows else pd.DataFrame(columns=["sequence_id", "taxon"])
    merged = assignments[["sequence_id", id_col]].merge(seq_taxon, on="sequence_id", how="left")

    taxon_totals = merged.groupby("taxon").size().sort_values(ascending=False)
    major_taxa = set(taxon_totals.head(25).index.tolist())

    for pos in [int(p) for p in top_positions]:
        for taxon in major_taxa:
            seq_ids = merged[merged["taxon"] == taxon]["sequence_id"].tolist()
            if not seq_ids:
                continue
            residues: List[str] = []
            for seq_id in seq_ids:
                seq = alignment_records.get(seq_id)
                if seq is None or pos - 1 >= len(seq):
                    continue
                aa = seq[pos - 1]
                if aa != "-":
                    residues.append(aa)
            total = len(seq_ids)
            present = len(residues)
            dominant = Counter(residues).most_common(1)[0][0] if residues else "-"
            rows.append(
                {
                    "level": level,
                    "position": pos,
                    "taxon": taxon,
                    "present_count": present,
                    "total_sequences": total,
                    "presence_fraction": float(present / total) if total else 0.0,
                    "dominant_residue": dominant,
                }
            )

    return pd.DataFrame(rows)


def run_phase7_analyses(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_groups: Path,
    raw_pairwise_families: Path,
    raw_pairwise_subfamilies: Path,
    groups_scores_path: Path,
    family_scores_path: Path,
    subfamily_scores_path: Path,
    msa_path: Path,
    ancestral_fasta_path: Path,
    asr_mapping_path: Path,
    domain_architecture_path: Path,
    pdb_path: Path,
    output_dir: Path,
    pdb_id: str = "2cg4",
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_pairwise_paths = {
        "groups": raw_pairwise_groups,
        "families": raw_pairwise_families,
        "subfamilies": raw_pairwise_subfamilies,
    }
    score_paths = {
        "groups": groups_scores_path,
        "families": family_scores_path,
        "subfamilies": subfamily_scores_path,
    }

    assignments = pd.read_csv(assignments_path)
    alignment_parsed = list(SeqIO.parse(str(msa_path), "fasta"))
    alignment_records = {rec.id: str(rec.seq) for rec in alignment_parsed}
    descriptions = {rec.id: rec.description for rec in alignment_parsed}
    ancestral_records = {rec.id: str(rec.seq) for rec in SeqIO.parse(str(ancestral_fasta_path), "fasta")}
    asr_map_df = pd.read_csv(asr_mapping_path)
    lca_to_asr = {
        str(row["lca_node"]): str(row["lca_node_asr"])
        for _, row in asr_map_df.iterrows()
        if pd.notna(row.get("lca_node")) and pd.notna(row.get("lca_node_asr"))
    }
    with domain_architecture_path.open("r") as handle:
        domain_architecture = json.load(handle)

    consensus = _consensus_sequence(msa_path)
    mapper = PDBMapper(pdb_id=pdb_id, pdb_file=str(pdb_path))
    mapping = mapper.map_alignment_to_structure(msa_path)

    tu_find = _run_tu_json(["tu", "find", "amino acid properties hydrophobicity charge", "--json"])
    tu_tools = [tool.get("name") for tool in tu_find.get("tools", [])[:5]]

    events_by_level: Dict[str, pd.DataFrame] = {}
    score_by_level: Dict[str, pd.DataFrame] = {}
    for level in LEVELS:
        events_by_level[level] = _load_switch_events_for_level(tree_path, assignments_path, raw_pairwise_paths[level], level)
        score_by_level[level] = pd.read_csv(score_paths[level])

    timeline_svg = output_dir / "switch_timeline.svg"
    all_events = pd.concat([events_by_level[level] for level in LEVELS], ignore_index=True)
    _plot_switch_timeline(all_events, timeline_svg)

    community_rows: List[pd.DataFrame] = []
    tax_rows: List[pd.DataFrame] = []
    all_shifts: List[pd.DataFrame] = []
    top_positions_for_lit: Dict[str, List[int]] = {}
    outputs: Dict[str, Path] = {"switch_timeline_svg": timeline_svg}

    for level in LEVELS:
        level_scores = score_by_level[level]
        level_events = events_by_level[level]

        top_positions = (
            level_scores[level_scores["switch_count"] > 0]
            .sort_values(["switch_count", "max_score"], ascending=[False, False])["position"]
            .astype(int)
            .head(15)
            .tolist()
        )
        top_positions_for_lit[level] = top_positions

        residue_numbers = [mapping[pos] for pos in top_positions if pos in mapping]
        distance_matrix = calculate_ca_distance_matrix(pdb_path, residue_numbers)
        distance_csv = output_dir / f"distance_matrix_{level}.csv"
        distance_matrix.to_csv(distance_csv, index=True)
        outputs[f"distance_matrix_csv_{level}"] = distance_csv

        heatmap_svg = output_dir / f"sdp_distance_heatmap_{level}.svg"
        _plot_clustered_heatmap(
            matrix=distance_matrix,
            output_svg=heatmap_svg,
            title=f"Top SDP C-alpha Distance Heatmap ({level.capitalize()})",
            cmap="mako_r",
            cbar_label="Distance (Å)",
            is_distance_matrix=True,
        )
        outputs[f"sdp_distance_heatmap_svg_{level}"] = heatmap_svg

        top_events = level_events[level_events["position"].isin(top_positions[:40])]
        if top_events.empty:
            coevo_matrix = pd.DataFrame(index=top_positions[:1], columns=top_positions[:1], data=1.0) if top_positions else pd.DataFrame()
        else:
            coevo_matrix = compute_coevolution_matrix(top_events[["branch_id", "position"]])

        coevo_csv = output_dir / f"coevolution_matrix_{level}.csv"
        coevo_matrix.to_csv(coevo_csv, index=True)
        outputs[f"coevolution_matrix_csv_{level}"] = coevo_csv

        coevo_svg = output_dir / f"coevolution_matrix_{level}.svg"
        _plot_clustered_heatmap(
            matrix=coevo_matrix,
            output_svg=coevo_svg,
            title=f"Co-evolution Matrix ({level.capitalize()})",
            cmap="viridis",
        )
        outputs[f"coevolution_matrix_svg_{level}"] = coevo_svg

        top_pairs = _top_correlated_pairs(coevo_matrix, top_n=5)
        mean_top_distance = _mean_upper_triangle(distance_matrix)
        print(f"Phase 7b terminal statistics ({level})")
        print(f"- Mean distance between top SDPs: {mean_top_distance:.3f} A")
        print("- Top 5 most highly correlated residue pairs:")
        for left, right, value in top_pairs:
            print(f"  ({left}, {right}) -> {value:.4f}")

        shifts_df = _compute_level_physicochemical_shifts(
            level=level,
            level_scores=level_scores,
            assignments=assignments,
            lca_to_asr=lca_to_asr,
            ancestral_records=ancestral_records,
            alignment_records=alignment_records,
            consensus=consensus,
            tu_tools=tu_tools,
        )
        shifts_csv = output_dir / f"physicochemical_shifts_{level}.csv"
        shifts_df.to_csv(shifts_csv, index=False)
        outputs[f"physicochemical_shifts_csv_{level}"] = shifts_csv
        all_shifts.append(shifts_df)

        functional_df = rank_top_functional_sdps(
            subfamily_scores_df=level_scores,
            coevolution_matrix_df=coevo_matrix,
            shifts_df=shifts_df,
            top_n=25,
        )
        functional_csv = output_dir / f"top_functional_sdps_{level}.csv"
        functional_df.to_csv(functional_csv, index=False)
        outputs[f"top_functional_sdps_csv_{level}"] = functional_csv

        domain_counts = count_switches_per_domain(level_events, domain_architecture)
        arch_svg = output_dir / f"architectural_distribution_{level}.svg"
        _plot_architecture_distribution(domain_counts, arch_svg, level)
        outputs[f"architectural_distribution_svg_{level}"] = arch_svg

        communities = assign_coevolution_communities(coevo_matrix, distance_cut=0.6)
        communities["level"] = level
        communities["community_size"] = communities.groupby("community_id")["position"].transform("count")
        community_rows.append(communities)

        tax_level = _collect_taxonomic_distribution(
            level=level,
            assignments=assignments,
            descriptions=descriptions,
            top_positions=functional_df["position"].astype(int).head(15).tolist() if not functional_df.empty else [],
            alignment_records=alignment_records,
        )
        tax_rows.append(tax_level)

    communities_df = pd.concat(community_rows, ignore_index=True) if community_rows else pd.DataFrame(columns=["position", "community_id", "level", "community_size"])
    communities_csv = output_dir / "coevolution_communities.csv"
    communities_df.to_csv(communities_csv, index=False)
    outputs["coevolution_communities_csv"] = communities_csv

    tax_df = pd.concat(tax_rows, ignore_index=True) if tax_rows else pd.DataFrame()
    tax_csv = output_dir / "taxonomic_sdp_distribution.csv"
    tax_df.to_csv(tax_csv, index=False)
    outputs["taxonomic_sdp_distribution_csv"] = tax_csv

    shifts_all_df = pd.concat(all_shifts, ignore_index=True) if all_shifts else pd.DataFrame()
    shifts_all_csv = output_dir / "physicochemical_shifts.csv"
    shifts_all_df.to_csv(shifts_all_csv, index=False)
    outputs["physicochemical_shifts_csv"] = shifts_all_csv

    families_subfamilies = shifts_all_df[shifts_all_df["level"].isin(["families", "subfamilies"])].copy() if not shifts_all_df.empty else pd.DataFrame()
    map_input_csv = output_dir / "physicochemical_shifts_for_mapping.csv"
    families_subfamilies.to_csv(map_input_csv, index=False)

    physio_cxc = Path("results/structural_mapping/highlight_physicochemistry.cxc")
    mapper.generate_physicochemical_chimerax_script(
        alignment_path=msa_path,
        physicochemical_csv=map_input_csv,
        output_cxc=physio_cxc,
    )
    outputs["physicochemical_mapping_cxc"] = physio_cxc

    lit_queries = [
        "IPR019888 active site",
        "IPR019888 specificity residues",
        "AraC family DNA-binding specificity residues",
    ]
    print("ToolUniverse literature cross-reference")
    print(f"- Queries executed: {', '.join(lit_queries)}")
    all_hits: List[dict] = []
    for query in lit_queries:
        all_hits.extend(_tu_literature_hits(query, max_hits=3))
    print(f"- Candidate literature hits retrieved: {len(all_hits)}")

    top_predicted = set()
    for level in LEVELS:
        top_predicted.update(top_positions_for_lit.get(level, [])[:10])
    matched_mentions: List[Tuple[str, int]] = []
    for hit in all_hits:
        blob = " ".join(str(hit.get(key, "")) for key in ["title", "abstract", "snippet", "summary"]).lower()
        for residue in sorted(top_predicted):
            if str(residue) in blob:
                matched_mentions.append((str(hit.get("title", "untitled")), residue))

    if matched_mentions:
        print("- Overlap between literature residue mentions and top predicted SDPs:")
        for title, residue in matched_mentions[:10]:
            print(f"  residue {residue}: {title}")
    else:
        print("- No direct residue-number overlap found in retrieved snippets (manual curation recommended).")

    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7 evolutionary and physicochemical analysis")
    parser.add_argument("--tree", default="results/topological_clustering/midpoint_rooted.tree")
    parser.add_argument("--assignments", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--raw-pairwise-groups", default="results/badasp_scoring/raw_pairwise_groups.csv")
    parser.add_argument("--raw-pairwise-families", default="results/badasp_scoring/raw_pairwise_families.csv")
    parser.add_argument("--raw-pairwise-subfamilies", default="results/badasp_scoring/raw_pairwise_subfamilies.csv")
    parser.add_argument("--groups-scores", default="results/badasp_scoring/badasp_scores_groups.csv")
    parser.add_argument("--family-scores", default="results/badasp_scoring/badasp_scores_families.csv")
    parser.add_argument("--subfamily-scores", default="results/badasp_scoring/badasp_scores_subfamilies.csv")
    parser.add_argument("--msa", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--ancestral", default="data/interim/ancestral_sequences.fasta")
    parser.add_argument("--asr-map", default="results/topological_clustering/tree_clusters_asr_mapped.csv")
    parser.add_argument("--domain-architecture", default="data/domain_architecture.json")
    parser.add_argument("--pdb", default="data/raw/2cg4.pdb")
    parser.add_argument("--pdb-id", default="2cg4")
    parser.add_argument("--output-dir", default="results/evolutionary_analysis")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = run_phase7_analyses(
        tree_path=Path(args.tree),
        assignments_path=Path(args.assignments),
        raw_pairwise_groups=Path(args.raw_pairwise_groups),
        raw_pairwise_families=Path(args.raw_pairwise_families),
        raw_pairwise_subfamilies=Path(args.raw_pairwise_subfamilies),
        groups_scores_path=Path(args.groups_scores),
        family_scores_path=Path(args.family_scores),
        subfamily_scores_path=Path(args.subfamily_scores),
        msa_path=Path(args.msa),
        ancestral_fasta_path=Path(args.ancestral),
        asr_mapping_path=Path(args.asr_map),
        domain_architecture_path=Path(args.domain_architecture),
        pdb_path=Path(args.pdb),
        output_dir=Path(args.output_dir),
        pdb_id=str(args.pdb_id),
    )
    for label, path in outputs.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
