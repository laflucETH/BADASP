import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
import numpy as np
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


def _clade_count_at_threshold(linkage_rows: Sequence[Sequence[float]], threshold: float) -> int:
    cluster_ids = fcluster(linkage_rows, t=threshold, criterion="distance")
    return len(set(int(x) for x in cluster_ids))


def _surviving_clade_count_at_threshold(
    linkage_rows: Sequence[Sequence[float]],
    threshold: float,
    min_clade_size: int,
) -> int:
    cluster_ids = [int(x) for x in fcluster(linkage_rows, t=threshold, criterion="distance")]
    sizes: Dict[int, int] = {}
    for cid in cluster_ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    return sum(1 for size in sizes.values() if size >= min_clade_size)


def choose_distance_threshold(
    linkage_rows: Sequence[Sequence[float]],
    target_min_clades: int = 20,
    target_max_clades: int = 80,
    min_clade_size: int = 1,
    iterations: int = 30,
) -> float:
    if target_min_clades <= 0 or target_max_clades < target_min_clades:
        raise ValueError("Invalid target clade range.")

    z = np.asarray(linkage_rows, dtype=float)
    min_dist = float(np.min(z[:, 2]))
    max_dist = float(np.max(z[:, 2]))

    low = min_dist
    high = max_dist
    midpoint = (target_min_clades + target_max_clades) / 2.0

    sampled: List[Tuple[float, int]] = []

    low_count = _surviving_clade_count_at_threshold(linkage_rows, low, min_clade_size)
    high_count = _surviving_clade_count_at_threshold(linkage_rows, high, min_clade_size)
    sampled.append((low, low_count))
    sampled.append((high, high_count))

    if target_min_clades <= low_count <= target_max_clades:
        return low
    if target_min_clades <= high_count <= target_max_clades:
        return high

    for _ in range(iterations):
        mid = (low + high) / 2.0
        mid_count = _surviving_clade_count_at_threshold(linkage_rows, mid, min_clade_size)
        sampled.append((mid, mid_count))

        if target_min_clades <= mid_count <= target_max_clades:
            return mid

        if mid_count > target_max_clades:
            low = mid
        else:
            high = mid

    best_threshold, _ = min(sampled, key=lambda x: abs(x[1] - midpoint))
    return best_threshold


def cluster_tree_topologically(
    tree_path: Path,
    clusters_output: Path,
    assignments_output: Path,
    distance_threshold: Optional[float] = None,
    target_min_clades: int = 20,
    target_max_clades: int = 80,
    min_clade_size: int = 5,
    dendrogram_output: Optional[Path] = None,
) -> Tuple[int, float]:
    tree = Phylo.read(str(tree_path), "newick")
    labels, linkage_rows = tree_to_linkage(tree)

    if distance_threshold is None:
        distance_threshold = choose_distance_threshold(
            linkage_rows=linkage_rows,
            target_min_clades=target_min_clades,
            target_max_clades=target_max_clades,
            min_clade_size=min_clade_size,
        )

    cluster_ids = [int(x) for x in fcluster(linkage_rows, t=distance_threshold, criterion="distance")]

    assignments: Dict[int, List[str]] = {}
    for terminal_name, cluster_id in zip(labels, cluster_ids):
        assignments.setdefault(cluster_id, []).append(terminal_name)

    filtered_assignments = {
        cluster_id: members for cluster_id, members in assignments.items() if len(members) >= min_clade_size
    }
    if not filtered_assignments:
        raise ValueError("No clades survive the minimum size threshold.")

    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    with assignments_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["terminal_name", "clade_id"])
        for cluster_id in sorted(filtered_assignments):
            for terminal in sorted(filtered_assignments[cluster_id]):
                writer.writerow([terminal, cluster_id])

    clusters_output.parent.mkdir(parents=True, exist_ok=True)
    with clusters_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clade_id", "member_count", "lca_node"])
        for cluster_id in sorted(filtered_assignments):
            members = filtered_assignments[cluster_id]
            lca_name = _resolve_lca_label(tree, members)
            writer.writerow([cluster_id, len(members), lca_name])

    if dendrogram_output is not None:
        plot_topological_dendrogram(linkage_rows, dendrogram_output, color_threshold=distance_threshold)

    return len(filtered_assignments), float(distance_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Topological clustering of a phylogenetic tree.")
    parser.add_argument("--tree", default="data/interim/IPR019888.tree")
    parser.add_argument("--clusters-output", default="results/topological_clustering/tree_clusters.csv")
    parser.add_argument("--assignments-output", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--target-min-clades", type=int, default=20)
    parser.add_argument("--target-max-clades", type=int, default=80)
    parser.add_argument("--min-clade-size", type=int, default=5)
    parser.add_argument("--dendrogram-output", default="results/topological_clustering/tree_dendrogram.svg")
    args = parser.parse_args()

    clade_count, used_threshold = cluster_tree_topologically(
        tree_path=Path(args.tree),
        clusters_output=Path(args.clusters_output),
        assignments_output=Path(args.assignments_output),
        distance_threshold=args.distance_threshold,
        target_min_clades=args.target_min_clades,
        target_max_clades=args.target_max_clades,
        min_clade_size=args.min_clade_size,
        dendrogram_output=Path(args.dendrogram_output),
    )
    print(f"Topological clades generated: {clade_count}")
    print(f"Distance threshold used: {used_threshold:.6f}")


if __name__ == "__main__":
    main()
