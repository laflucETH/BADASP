import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from scipy.cluster.hierarchy import fcluster

try:
    from src.visualization import plot_topological_dendrogram
except ModuleNotFoundError:
    from visualization import plot_topological_dendrogram


def _compute_subtree_heights(clade: Clade) -> Dict[int, float]:
    heights: Dict[int, float] = {}

    def _walk(node: Clade) -> float:
        if node.is_terminal():
            heights[id(node)] = 0.0
            return 0.0

        child_heights: List[float] = []
        for child in node.clades:
            branch_len = child.branch_length or 0.0
            child_heights.append(branch_len + _walk(child))
        node_height = max(child_heights) if child_heights else 0.0
        heights[id(node)] = node_height
        return node_height

    _walk(clade)
    return heights


def tree_to_linkage(tree: Tree) -> Tuple[List[str], List[List[float]]]:
    leaves = tree.get_terminals()
    leaf_index = {id(leaf): i for i, leaf in enumerate(leaves)}
    labels = [leaf.name or f"leaf_{i}" for i, leaf in enumerate(leaves)]

    if len(leaves) < 2:
        raise ValueError("Need at least 2 leaves to build linkage.")

    heights = _compute_subtree_heights(tree.root)
    cluster_map: Dict[int, Tuple[int, int]] = {id(leaf): (leaf_index[id(leaf)], 1) for leaf in leaves}

    next_id = len(leaves)
    linkage_rows: List[List[float]] = []

    for node in tree.get_nonterminals(order="postorder"):
        child_clusters = [cluster_map[id(child)] for child in node.clades if id(child) in cluster_map]
        if len(child_clusters) < 2:
            continue

        node_distance = max(0.0, 2.0 * heights[id(node)])
        merged_id, merged_size = child_clusters[0]

        for child_id, child_size in child_clusters[1:]:
            total = merged_size + child_size
            linkage_rows.append([float(merged_id), float(child_id), float(node_distance), float(total)])
            merged_id = next_id
            merged_size = total
            next_id += 1

        cluster_map[id(node)] = (merged_id, merged_size)

    expected_rows = len(leaves) - 1
    if len(linkage_rows) != expected_rows:
        raise ValueError(
            f"Invalid linkage conversion. Expected {expected_rows} rows, got {len(linkage_rows)}."
        )

    return labels, linkage_rows


def _resolve_lca_label(tree: Tree, members: List[str]) -> str:
    lca = tree.common_ancestor(members)
    if lca.name:
        return lca.name

    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if node is lca:
            return f"InternalNode_{idx}"
    return "InternalNode_unknown"


def cluster_tree_topologically(
    tree_path: Path,
    clusters_output: Path,
    assignments_output: Path,
    distance_threshold: Optional[float] = None,
    dendrogram_output: Optional[Path] = None,
) -> int:
    tree = Phylo.read(str(tree_path), "newick")
    labels, linkage_rows = tree_to_linkage(tree)

    if distance_threshold is None:
        distances = sorted(row[2] for row in linkage_rows)
        pivot_idx = max(0, int(0.25 * len(distances)) - 1)
        distance_threshold = distances[pivot_idx] if distances else 0.0

    cluster_ids = fcluster(linkage_rows, t=distance_threshold, criterion="distance")

    assignments: Dict[int, List[str]] = {}
    for terminal_name, cluster_id in zip(labels, cluster_ids):
        assignments.setdefault(int(cluster_id), []).append(terminal_name)

    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    with assignments_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["terminal_name", "clade_id"])
        for cluster_id in sorted(assignments):
            for terminal in sorted(assignments[cluster_id]):
                writer.writerow([terminal, cluster_id])

    clusters_output.parent.mkdir(parents=True, exist_ok=True)
    with clusters_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clade_id", "member_count", "lca_node"])
        for cluster_id in sorted(assignments):
            members = assignments[cluster_id]
            lca_name = _resolve_lca_label(tree, members)
            writer.writerow([cluster_id, len(members), lca_name])

    if dendrogram_output is not None:
        plot_topological_dendrogram(linkage_rows, dendrogram_output)

    return len(assignments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Topological clustering of a phylogenetic tree.")
    parser.add_argument("--tree", default="data/interim/IPR019888.tree")
    parser.add_argument("--clusters-output", default="results/topological_clustering/tree_clusters.csv")
    parser.add_argument("--assignments-output", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--dendrogram-output", default="results/topological_clustering/tree_dendrogram.svg")
    args = parser.parse_args()

    clade_count = cluster_tree_topologically(
        tree_path=Path(args.tree),
        clusters_output=Path(args.clusters_output),
        assignments_output=Path(args.assignments_output),
        distance_threshold=args.distance_threshold,
        dendrogram_output=Path(args.dendrogram_output),
    )
    print(f"Topological clades generated: {clade_count}")


if __name__ == "__main__":
    main()
