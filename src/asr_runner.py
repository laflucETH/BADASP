import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from Bio import Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def run_iqtree_asr(
    alignment_path: Path,
    tree_path: Path,
    output_prefix: Path,
    iqtree_binary: str = "iqtree2",
    model: str = "LG+G",
    threads: str = "AUTO",
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        iqtree_binary,
        "-s",
        str(alignment_path),
        "-te",
        str(tree_path),
        "-m",
        model,
        "-asr",
        "-nt",
        str(threads),
        "--prefix",
        str(output_prefix),
    ]
    subprocess.run(cmd, check=True)


def parse_iqtree_state_sequences(state_file: Path) -> Dict[str, str]:
    if not state_file.exists():
        raise FileNotFoundError(f"IQ-TREE state file not found: {state_file}")

    per_node_sites: Dict[str, Dict[int, str]] = defaultdict(dict)
    with state_file.open("r", encoding="utf-8") as handle:
        header_found = False
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not header_found:
                if len(parts) < 3 or parts[0] != "Node" or parts[1] != "Site":
                    raise ValueError("Unexpected IQ-TREE .state format header.")
                header_found = True
                continue

            if len(parts) < 3:
                continue
            node, site_str, state = parts[0], parts[1], parts[2]
            if state == "-":
                state = "X"
            per_node_sites[node][int(site_str)] = state

    if not header_found:
        raise ValueError("Unexpected IQ-TREE .state format header.")

    node_sequences: Dict[str, str] = {}
    for node, site_map in per_node_sites.items():
        max_site = max(site_map) if site_map else 0
        seq_chars = [site_map.get(i, "X") for i in range(1, max_site + 1)]
        node_sequences[node] = "".join(seq_chars)

    return node_sequences


def _read_clade_members(assignments_csv: Path) -> Dict[int, List[str]]:
    clades: Dict[int, List[str]] = defaultdict(list)
    with assignments_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clade_id = int(row["clade_id"])
            clades[clade_id].append(row["terminal_name"])
    return clades


def _read_hierarchical_lca_members(assignments_csv: Path) -> Dict[str, List[str]]:
    members_by_node: Dict[str, List[str]] = defaultdict(list)
    with assignments_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return members_by_node

        lca_columns = [col for col in reader.fieldnames if col.endswith("_lca_node")]
        if not lca_columns:
            return members_by_node

        for row in reader:
            sequence_id = (row.get("sequence_id") or "").strip()
            if not sequence_id:
                continue
            for col in lca_columns:
                value = (row.get(col) or "").strip()
                if value and sequence_id not in members_by_node[value]:
                    members_by_node[value].append(sequence_id)
    return members_by_node


def extract_lca_ancestral_sequences(
    tree_path: Path,
    assignments_csv: Path,
    node_sequences: Dict[str, str],
    output_fasta: Path,
    min_clade_size: int = 5,
) -> int:
    hierarchical_nodes = _read_hierarchical_lca_members(assignments_csv)

    records: List[SeqRecord] = []

    # New hierarchical format: extract all unique LCA nodes across
    # group_lca_node/family_lca_node/subfamily_lca_node (and any *_lca_node columns).
    if hierarchical_nodes:
        tree = Phylo.read(str(tree_path), "newick")
        seen_nodes = set()
        for node_id in sorted(hierarchical_nodes):
            members = hierarchical_nodes[node_id]
            lca_node = tree.common_ancestor(members)
            resolved_node_id = lca_node.name
            if not resolved_node_id:
                raise ValueError(f"LCA node for {node_id} has no node name in ASR tree.")
            if resolved_node_id in seen_nodes:
                continue
            if resolved_node_id not in node_sequences:
                raise KeyError(f"No ancestral sequence found for LCA node {resolved_node_id}.")
            seen_nodes.add(resolved_node_id)
            records.append(
                SeqRecord(
                    Seq(node_sequences[resolved_node_id]),
                    id=resolved_node_id,
                    description="source=hierarchy_mapping",
                )
            )
    else:
        # Backward-compatible fallback for legacy single-level assignments.
        tree = Phylo.read(str(tree_path), "newick")
        clades = _read_clade_members(assignments_csv)
        seen_nodes = set()

        for clade_id, members in sorted(clades.items()):
            if len(members) < min_clade_size:
                continue

            lca_node = tree.common_ancestor(members)
            if not lca_node.name:
                raise ValueError(f"LCA node for clade {clade_id} has no node name in ASR tree.")

            node_id = lca_node.name
            if node_id in seen_nodes:
                continue
            if node_id not in node_sequences:
                raise KeyError(f"No ancestral sequence found for LCA node {node_id}.")

            seen_nodes.add(node_id)
            records.append(
                SeqRecord(
                    Seq(node_sequences[node_id]),
                    id=node_id,
                    description=f"clade_id={clade_id};member_count={len(members)}",
                )
            )

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, str(output_fasta), "fasta")
    return len(records)


def run_asr_pipeline(
    alignment_path: Path,
    tree_path: Path,
    assignments_csv: Path,
    output_fasta: Path,
    iqtree_binary: str = "iqtree2",
    output_prefix: Path = Path("data/interim/asr_run"),
    min_clade_size: int = 5,
    reuse_existing: bool = False,
) -> int:
    state_file = output_prefix.with_suffix(".state")
    asr_tree = output_prefix.with_suffix(".treefile")

    if not reuse_existing or not state_file.exists() or not asr_tree.exists():
        run_iqtree_asr(
            alignment_path=alignment_path,
            tree_path=tree_path,
            output_prefix=output_prefix,
            iqtree_binary=iqtree_binary,
        )

    node_sequences = parse_iqtree_state_sequences(state_file)

    return extract_lca_ancestral_sequences(
        tree_path=asr_tree,
        assignments_csv=assignments_csv,
        node_sequences=node_sequences,
        output_fasta=output_fasta,
        min_clade_size=min_clade_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IQ-TREE ASR and extract LCA ancestral sequences.")
    parser.add_argument("--alignment", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--tree", default="data/interim/IPR019888.tree")
    parser.add_argument("--assignments", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--output", default="data/interim/ancestral_sequences.fasta")
    parser.add_argument("--iqtree-binary", default="iqtree2")
    parser.add_argument("--prefix", default="data/interim/asr_run")
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    written = run_asr_pipeline(
        alignment_path=Path(args.alignment),
        tree_path=Path(args.tree),
        assignments_csv=Path(args.assignments),
        output_fasta=Path(args.output),
        iqtree_binary=args.iqtree_binary,
        output_prefix=Path(args.prefix),
        min_clade_size=args.min_clade_size,
        reuse_existing=args.reuse_existing,
    )
    print(f"LCA ancestral sequences written: {written}")


if __name__ == "__main__":
    main()
