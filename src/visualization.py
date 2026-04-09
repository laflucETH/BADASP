import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Phylo, SeqIO
from Bio.Phylo.BaseTree import Clade, Tree
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette


def default_plot_paths() -> Tuple[Path, Path, Path]:
    return (
        Path("results/sequence_filtering/raw_length_dist.svg"),
        Path("results/alignment_qc/msa_gap_profile.svg"),
        Path("results/topological_clustering/tree_dendrogram.svg"),
    )


def default_hierarchical_badasp_plot_paths() -> Tuple[Path, Path]:
    return (
        Path("results/badasp_scoring/hierarchical_distributions.svg"),
        Path("results/badasp_scoring/hierarchical_switch_counts.svg"),
    )


def default_individual_badasp_plot_paths() -> Tuple[Path, Path, Path]:
    return (
        Path("results/badasp_scoring/badasp_score_distribution_groups.svg"),
        Path("results/badasp_scoring/badasp_score_distribution_families.svg"),
        Path("results/badasp_scoring/badasp_score_distribution_subfamilies.svg"),
    )


def default_tree_switch_plot_paths() -> Tuple[Path, Path, Path]:
    return (
        Path("results/badasp_scoring/tree_switches_groups.svg"),
        Path("results/badasp_scoring/tree_switches_families.svg"),
        Path("results/badasp_scoring/tree_switches_subfamilies.svg"),
    )


def _read_fasta_lengths(fasta_path: Path) -> List[int]:
    return [len(record.seq) for record in SeqIO.parse(str(fasta_path), "fasta")]


def plot_sequence_length_distribution(fasta_path: Path, output_svg: Path) -> None:
    lengths = _read_fasta_lengths(fasta_path)
    if not lengths:
        raise ValueError(f"No sequences found in {fasta_path}")

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=80, kde=True, color="#2E86AB")
    plt.title("Sequence Length Distribution")
    plt.xlabel("Sequence Length (AA)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def compute_gap_percentages(msa_path: Path) -> List[float]:
    sequences = [str(record.seq) for record in SeqIO.parse(str(msa_path), "fasta")]
    if not sequences:
        raise ValueError(f"No aligned sequences found in {msa_path}")

    aln_len = len(sequences[0])
    gap_percentages: List[float] = []
    for i in range(aln_len):
        gap_count = sum(1 for seq in sequences if seq[i] == "-")
        gap_percentages.append((gap_count / len(sequences)) * 100.0)
    return gap_percentages


def plot_gap_percentage_per_column(msa_path: Path, output_svg: Path) -> None:
    gap_percentages = compute_gap_percentages(msa_path)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(gap_percentages) + 1), gap_percentages, color="#F18F01", linewidth=1.0)
    plt.title("MSA Gap Percentage per Column")
    plt.xlabel("Alignment Column")
    plt.ylabel("Gap Percentage (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def plot_topological_dendrogram(
    linkage_matrix: Sequence[Sequence[float]],
    output_svg: Path,
    max_leaves: int = 200,
    color_threshold: float = 0.0,
) -> None:
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(linkage_matrix, dtype=float)
    # Use an expanded palette so large clade counts can still receive distinct colors.
    palette = [plt.cm.tab20(i / 20) for i in range(20)] + [plt.cm.Set3(i / 12) for i in range(12)]
    set_link_color_palette([mcolors.to_hex(c) for c in palette])

    plt.figure(figsize=(12, 6))
    dendrogram(
        z,
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="#666666",
    )
    set_link_color_palette(None)
    plt.title("Topological Clustering Dendrogram")
    plt.xlabel("Collapsed Leaf Groups")
    plt.ylabel("Cophenetic Distance")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def _load_score_table(score_path: Path) -> pd.DataFrame:
    df = pd.read_csv(score_path)
    required_columns = {"position", "switch_count", "global_threshold", "badasp_score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {score_path}: {sorted(missing)}")
    return df


def _load_pairwise_table(pairwise_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pairwise_path)
    required_columns = {"pair", "position", "score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {pairwise_path}: {sorted(missing)}")
    return df


def plot_badasp_score_distribution(
    raw_pairwise_path: Path,
    output_svg: Path,
    title: str,
    color: str,
) -> None:
    df = _load_pairwise_table(raw_pairwise_path)
    scores = df["score"].astype(float).to_numpy()
    threshold = float(np.percentile(scores, 95)) if len(scores) else 0.0

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=40, kde=True, stat="density", color=color, alpha=0.35)
    plt.axvline(threshold, color=color, linestyle="--", linewidth=2.0, label=f"95th percentile = {threshold:.3f}")
    plt.title(title)
    plt.xlabel("Raw BADASP Score")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def plot_hierarchical_badasp_distributions(
    group_pairwise: Path,
    family_pairwise: Path,
    subfamily_pairwise: Path,
    output_svg: Path,
) -> None:
    score_tables = {
        "Groups": _load_pairwise_table(group_pairwise),
        "Families": _load_pairwise_table(family_pairwise),
        "Subfamilies": _load_pairwise_table(subfamily_pairwise),
    }
    colors = {
        "Groups": "#1F77B4",
        "Families": "#D95F02",
        "Subfamilies": "#2CA02C",
    }

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))

    for label, df in score_tables.items():
        scores = df["score"].astype(float).to_numpy()
        threshold = float(np.percentile(scores, 95)) if len(scores) else 0.0
        sns.kdeplot(scores, label=label, color=colors[label], linewidth=2.0, fill=False)
        plt.axvline(threshold, color=colors[label], linestyle="--", linewidth=1.5, alpha=0.8)

    threshold_legend = [
        Line2D([0], [0], color=colors[label], linestyle="--", linewidth=1.5, label=f"{label} 95th pct.")
        for label in score_tables
    ]
    density_legend = [
        Line2D([0], [0], color=colors[label], linewidth=2.0, label=label)
        for label in score_tables
    ]
    plt.legend(handles=density_legend + threshold_legend, loc="best", frameon=False, ncol=2)
    plt.title("Hierarchical BADASP Score Distributions")
    plt.xlabel("Raw BADASP Score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def plot_hierarchical_switch_counts(
    group_scores: Path,
    family_scores: Path,
    subfamily_scores: Path,
    output_svg: Path,
) -> None:
    score_tables = [
        ("Groups", _load_score_table(group_scores), "#1F77B4"),
        ("Families", _load_score_table(family_scores), "#D95F02"),
        ("Subfamilies", _load_score_table(subfamily_scores), "#2CA02C"),
    ]

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (label, df, color) in zip(axes, score_tables):
        positions = df["position"].astype(int).to_numpy()
        switch_counts = df["switch_count"].astype(int).to_numpy()
        ax.bar(positions, switch_counts, color=color, width=1.0, alpha=0.9)
        ax.set_ylabel("Switches")
        ax.set_title(label)
        ax.set_xlim(1, int(positions.max()))
        ax.set_ylim(0, max(1, int(switch_counts.max())) + 1)

    axes[-1].set_xlabel("Alignment Position")
    fig.suptitle("Hierarchical BADASP Switch Counts Across the Alignment", y=0.995)
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def plot_individual_hierarchical_badasp_distributions(
    group_pairwise: Path,
    family_pairwise: Path,
    subfamily_pairwise: Path,
    output_group_svg: Path,
    output_family_svg: Path,
    output_subfamily_svg: Path,
) -> None:
    plot_badasp_score_distribution(
        raw_pairwise_path=group_pairwise,
        output_svg=output_group_svg,
        title="Groups BADASP Score Distribution",
        color="#1F77B4",
    )
    plot_badasp_score_distribution(
        raw_pairwise_path=family_pairwise,
        output_svg=output_family_svg,
        title="Families BADASP Score Distribution",
        color="#D95F02",
    )
    plot_badasp_score_distribution(
        raw_pairwise_path=subfamily_pairwise,
        output_svg=output_subfamily_svg,
        title="Subfamilies BADASP Score Distribution",
        color="#2CA02C",
    )


def _ensure_tree_node_names(tree: Tree) -> None:
    for idx, node in enumerate(tree.get_nonterminals(order="preorder"), start=1):
        if not node.name:
            node.name = f"InternalNode_{idx}"


def _build_y_positions(tree: Tree) -> Dict[Clade, float]:
    terminals = tree.get_terminals()
    y_positions: Dict[Clade, float] = {leaf: float(i) for i, leaf in enumerate(reversed(terminals), start=1)}

    def _assign(node: Clade) -> float:
        if node in y_positions:
            return y_positions[node]
        child_ys = [_assign(child) for child in node.clades]
        y_positions[node] = (min(child_ys) + max(child_ys)) / 2.0
        return y_positions[node]

    _assign(tree.root)
    return y_positions


def build_switch_node_map(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_path: Path,
    level: str,
) -> Dict[str, int]:
    level_map = {
        "groups": "group",
        "families": "family",
        "subfamilies": "subfamily",
    }
    if level not in level_map:
        raise ValueError(f"Unsupported level: {level}")

    singular = level_map[level]
    id_col = f"{singular}_id"

    tree = Phylo.read(str(tree_path), "newick")
    _ensure_tree_node_names(tree)

    assignments = pd.read_csv(assignments_path)
    members_by_cluster = assignments.groupby(id_col)["sequence_id"].apply(list).to_dict()

    raw_pairwise = _load_pairwise_table(raw_pairwise_path)
    if raw_pairwise.empty:
        return {}

    threshold = float(np.percentile(raw_pairwise["score"].astype(float), 95))
    switched = raw_pairwise[raw_pairwise["score"] > threshold]
    pair_switch_counts = switched.groupby("pair").size().to_dict()

    node_switch_counts: Dict[str, int] = defaultdict(int)
    for pair, switch_count in pair_switch_counts.items():
        try:
            left_str, right_str = str(pair).split("-")
            left_id = int(left_str)
            right_id = int(right_str)
        except ValueError:
            continue
        if left_id not in members_by_cluster or right_id not in members_by_cluster:
            continue
        pair_members = list(members_by_cluster[left_id]) + list(members_by_cluster[right_id])
        lca = tree.common_ancestor(pair_members)
        if not lca.name:
            continue
        node_switch_counts[lca.name] += int(switch_count)

    return dict(node_switch_counts)


def plot_tree_with_switches(
    tree_path: Path,
    node_switch_counts: Dict[str, int],
    output_svg: Path,
    title: str,
) -> None:
    tree = Phylo.read(str(tree_path), "newick")
    _ensure_tree_node_names(tree)

    depths = tree.depths()
    if not max(depths.values()):
        depths = tree.depths(unit_branch_lengths=True)
    y_positions = _build_y_positions(tree)

    fig, ax = plt.subplots(figsize=(14, 10))

    def _draw(node: Clade) -> None:
        x = depths[node]
        y = y_positions[node]
        for child in node.clades:
            child_x = depths[child]
            child_y = y_positions[child]
            ax.plot([x, child_x], [child_y, child_y], color="#666666", linewidth=0.7)
            ax.plot([x, x], [y, child_y], color="#666666", linewidth=0.7)
            _draw(child)

    _draw(tree.root)

    switched_nodes = [(node, count) for node, count in node_switch_counts.items() if count > 0]
    if switched_nodes:
        node_lookup = {clade.name: clade for clade in tree.find_clades() if clade.name}
        counts = np.array([count for _, count in switched_nodes], dtype=float)
        max_count = float(counts.max()) if len(counts) else 1.0

        xs = []
        ys = []
        sizes = []
        colors = []
        for node_name, count in switched_nodes:
            if node_name not in node_lookup:
                continue
            clade = node_lookup[node_name]
            xs.append(depths[clade])
            ys.append(y_positions[clade])
            sizes.append(30.0 + (220.0 * (count / max_count)))
            colors.append(count)

        scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="OrRd", alpha=0.9, edgecolor="#222222", linewidth=0.3)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Switch count")

    ax.set_title(title)
    ax.set_xlabel("Branch length from root")
    ax.set_ylabel("Taxa / internal nodes")
    ax.set_yticks([])
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_svg, format="svg")
    plt.close(fig)


def generate_tree_switch_plots(
    rooted_tree_path: Path,
    assignments_path: Path,
    output_groups_svg: Path,
    output_families_svg: Path,
    output_subfamilies_svg: Path,
    raw_pairwise_groups: Path,
    raw_pairwise_families: Path,
    raw_pairwise_subfamilies: Path,
) -> None:
    groups_map = build_switch_node_map(
        tree_path=rooted_tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_groups,
        level="groups",
    )
    families_map = build_switch_node_map(
        tree_path=rooted_tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_families,
        level="families",
    )
    subfamilies_map = build_switch_node_map(
        tree_path=rooted_tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_subfamilies,
        level="subfamilies",
    )

    plot_tree_with_switches(
        tree_path=rooted_tree_path,
        node_switch_counts=groups_map,
        output_svg=output_groups_svg,
        title="Switch Events on Tree: Groups",
    )
    plot_tree_with_switches(
        tree_path=rooted_tree_path,
        node_switch_counts=families_map,
        output_svg=output_families_svg,
        title="Switch Events on Tree: Families",
    )
    plot_tree_with_switches(
        tree_path=rooted_tree_path,
        node_switch_counts=subfamilies_map,
        output_svg=output_subfamilies_svg,
        title="Switch Events on Tree: Subfamilies",
    )


def _compute_dendrogram_node_coords(
    linkage_matrix: np.ndarray,
    leaves_order: List[int],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, frozenset]]:
    leaf_x = {leaf_idx: 5.0 + (10.0 * rank) for rank, leaf_idx in enumerate(leaves_order)}
    x_by_node: Dict[int, float] = {}
    y_by_node: Dict[int, float] = {}
    descendants: Dict[int, frozenset] = {}

    n_leaves = linkage_matrix.shape[0] + 1
    for i in range(n_leaves):
        x_by_node[i] = leaf_x[i]
        y_by_node[i] = 0.0
        descendants[i] = frozenset({i})

    for i, row in enumerate(linkage_matrix):
        left = int(row[0])
        right = int(row[1])
        node_id = n_leaves + i
        x_by_node[node_id] = (x_by_node[left] + x_by_node[right]) / 2.0
        y_by_node[node_id] = float(row[2])
        descendants[node_id] = descendants[left] | descendants[right]

    return x_by_node, y_by_node, descendants


def plot_dendrogram_with_switches(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_path: Path,
    level: str,
    output_svg: Path,
    title: str,
    threshold: float,
) -> None:
    from src.tree_cluster import tree_to_linkage

    level_map = {
        "groups": "group",
        "families": "family",
        "subfamilies": "subfamily",
    }
    if level not in level_map:
        raise ValueError(f"Unsupported level: {level}")
    singular = level_map[level]
    id_col = f"{singular}_id"

    tree = Phylo.read(str(tree_path), "newick")
    labels, linkage_rows = tree_to_linkage(tree)
    linkage_matrix = np.asarray(linkage_rows, dtype=float)

    dendro_meta = dendrogram(
        linkage_matrix,
        no_plot=True,
        no_labels=True,
        color_threshold=threshold,
        above_threshold_color="#666666",
    )

    x_by_node, y_by_node, descendants = _compute_dendrogram_node_coords(
        linkage_matrix=linkage_matrix,
        leaves_order=[int(idx) for idx in dendro_meta["leaves"]],
    )
    descendant_to_node = {desc: node_id for node_id, desc in descendants.items() if node_id >= len(labels)}

    assignments = pd.read_csv(assignments_path)
    cluster_members = assignments.groupby(id_col)["sequence_id"].apply(list).to_dict()
    name_to_index = {name: idx for idx, name in enumerate(labels)}

    raw_pairwise = _load_pairwise_table(raw_pairwise_path)
    switched = raw_pairwise[raw_pairwise["score"] > float(threshold)]
    pair_switch_counts = switched.groupby("pair").size().to_dict()

    node_switch_counts: Dict[int, int] = defaultdict(int)
    for pair, count in pair_switch_counts.items():
        try:
            left_str, right_str = str(pair).split("-")
            left_id = int(left_str)
            right_id = int(right_str)
        except ValueError:
            continue
        if left_id not in cluster_members or right_id not in cluster_members:
            continue
        combined_members = cluster_members[left_id] + cluster_members[right_id]
        idx_set = frozenset(name_to_index[name] for name in combined_members if name in name_to_index)
        node_id = descendant_to_node.get(idx_set)
        if node_id is not None:
            node_switch_counts[node_id] += int(count)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    palette = [plt.cm.tab20(i / 20) for i in range(20)] + [plt.cm.Set3(i / 12) for i in range(12)]
    set_link_color_palette([mcolors.to_hex(c) for c in palette])

    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=threshold,
        above_threshold_color="#666666",
    )
    set_link_color_palette(None)

    if node_switch_counts:
        node_ids = list(node_switch_counts.keys())
        switch_values = np.array([node_switch_counts[node_id] for node_id in node_ids], dtype=float)
        max_val = float(switch_values.max()) if len(switch_values) else 1.0
        xs = [x_by_node[node_id] for node_id in node_ids]
        ys = [y_by_node[node_id] for node_id in node_ids]
        sizes = [40.0 + (260.0 * (val / max_val)) for val in switch_values]

        scatter = plt.scatter(
            xs,
            ys,
            c=switch_values,
            s=sizes,
            cmap="OrRd",
            edgecolor="#222222",
            linewidth=0.4,
            alpha=0.9,
            zorder=5,
        )
        cbar = plt.colorbar(scatter, pad=0.01)
        cbar.set_label("Switch count")

    plt.title(title)
    plt.xlabel("Collapsed Leaf Groups")
    plt.ylabel("Cophenetic Distance")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def generate_dendrogram_switch_plots(
    tree_path: Path,
    assignments_path: Path,
    raw_pairwise_groups: Path,
    raw_pairwise_families: Path,
    raw_pairwise_subfamilies: Path,
    output_groups_svg: Path,
    output_families_svg: Path,
    output_subfamilies_svg: Path,
    group_threshold: float,
    family_threshold: float,
    subfamily_threshold: float,
) -> None:
    plot_dendrogram_with_switches(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_groups,
        level="groups",
        output_svg=output_groups_svg,
        title="Groups Dendrogram with Switch Events",
        threshold=group_threshold,
    )
    plot_dendrogram_with_switches(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_families,
        level="families",
        output_svg=output_families_svg,
        title="Families Dendrogram with Switch Events",
        threshold=family_threshold,
    )
    plot_dendrogram_with_switches(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_subfamilies,
        level="subfamilies",
        output_svg=output_subfamilies_svg,
        title="Subfamilies Dendrogram with Switch Events",
        threshold=subfamily_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and hierarchical BADASP visualizations.")
    default_length_out, default_gap_out, _ = default_plot_paths()
    default_hier_dist_out, default_hier_switch_out = default_hierarchical_badasp_plot_paths()
    default_group_dist_out, default_family_dist_out, default_subfamily_dist_out = default_individual_badasp_plot_paths()
    default_tree_groups_out, default_tree_families_out, default_tree_subfamilies_out = default_tree_switch_plot_paths()
    parser.add_argument("--fasta", default=None, help="Input FASTA for length distribution plot.")
    parser.add_argument("--length-output", default=str(default_length_out))
    parser.add_argument("--msa", default=None, help="Input MSA FASTA for gap-per-column plot.")
    parser.add_argument("--gap-output", default=str(default_gap_out))
    parser.add_argument("--group-scores", default="results/badasp_scoring/badasp_scores_groups.csv")
    parser.add_argument("--family-scores", default="results/badasp_scoring/badasp_scores_families.csv")
    parser.add_argument("--subfamily-scores", default="results/badasp_scoring/badasp_scores_subfamilies.csv")
    parser.add_argument("--group-pairwise", default="results/badasp_scoring/raw_pairwise_groups.csv")
    parser.add_argument("--family-pairwise", default="results/badasp_scoring/raw_pairwise_families.csv")
    parser.add_argument("--subfamily-pairwise", default="results/badasp_scoring/raw_pairwise_subfamilies.csv")
    parser.add_argument("--rooted-tree", default="results/topological_clustering/midpoint_rooted.tree")
    parser.add_argument("--assignments", default="results/topological_clustering/tree_cluster_assignments.csv")
    parser.add_argument("--hierarchical-distribution-output", default=str(default_hier_dist_out))
    parser.add_argument("--hierarchical-switch-output", default=str(default_hier_switch_out))
    parser.add_argument("--group-distribution-output", default=str(default_group_dist_out))
    parser.add_argument("--family-distribution-output", default=str(default_family_dist_out))
    parser.add_argument("--subfamily-distribution-output", default=str(default_subfamily_dist_out))
    parser.add_argument("--tree-switch-groups-output", default=str(default_tree_groups_out))
    parser.add_argument("--tree-switch-families-output", default=str(default_tree_families_out))
    parser.add_argument("--tree-switch-subfamilies-output", default=str(default_tree_subfamilies_out))
    parser.add_argument("--dendrogram-switch-groups-output", default="results/badasp_scoring/dendrogram_switches_groups.svg")
    parser.add_argument("--dendrogram-switch-families-output", default="results/badasp_scoring/dendrogram_switches_families.svg")
    parser.add_argument("--dendrogram-switch-subfamilies-output", default="results/badasp_scoring/dendrogram_switches_subfamilies.svg")
    parser.add_argument("--group-threshold", type=float, default=8.579924)
    parser.add_argument("--family-threshold", type=float, default=6.929765)
    parser.add_argument("--subfamily-threshold", type=float, default=4.729553)
    parser.add_argument("--hierarchical-only", action="store_true")
    args = parser.parse_args()

    if args.fasta and not args.hierarchical_only:
        plot_sequence_length_distribution(Path(args.fasta), Path(args.length_output))
        print(f"Saved length distribution: {args.length_output}")

    if args.msa and not args.hierarchical_only:
        plot_gap_percentage_per_column(Path(args.msa), Path(args.gap_output))
        print(f"Saved gap profile: {args.gap_output}")

    if Path(args.group_pairwise).exists() and Path(args.family_pairwise).exists() and Path(args.subfamily_pairwise).exists():
        plot_individual_hierarchical_badasp_distributions(
            group_pairwise=Path(args.group_pairwise),
            family_pairwise=Path(args.family_pairwise),
            subfamily_pairwise=Path(args.subfamily_pairwise),
            output_group_svg=Path(args.group_distribution_output),
            output_family_svg=Path(args.family_distribution_output),
            output_subfamily_svg=Path(args.subfamily_distribution_output),
        )
        print(f"Saved group score distribution: {args.group_distribution_output}")
        print(f"Saved family score distribution: {args.family_distribution_output}")
        print(f"Saved subfamily score distribution: {args.subfamily_distribution_output}")

        plot_hierarchical_badasp_distributions(
            group_pairwise=Path(args.group_pairwise),
            family_pairwise=Path(args.family_pairwise),
            subfamily_pairwise=Path(args.subfamily_pairwise),
            output_svg=Path(args.hierarchical_distribution_output),
        )
        print(f"Saved hierarchical score distributions: {args.hierarchical_distribution_output}")

        plot_hierarchical_switch_counts(
            group_scores=Path(args.group_scores),
            family_scores=Path(args.family_scores),
            subfamily_scores=Path(args.subfamily_scores),
            output_svg=Path(args.hierarchical_switch_output),
        )
        print(f"Saved hierarchical switch counts: {args.hierarchical_switch_output}")

        generate_dendrogram_switch_plots(
            tree_path=Path(args.rooted_tree),
            assignments_path=Path(args.assignments),
            raw_pairwise_groups=Path(args.group_pairwise),
            raw_pairwise_families=Path(args.family_pairwise),
            raw_pairwise_subfamilies=Path(args.subfamily_pairwise),
            output_groups_svg=Path(args.dendrogram_switch_groups_output),
            output_families_svg=Path(args.dendrogram_switch_families_output),
            output_subfamilies_svg=Path(args.dendrogram_switch_subfamilies_output),
            group_threshold=float(args.group_threshold),
            family_threshold=float(args.family_threshold),
            subfamily_threshold=float(args.subfamily_threshold),
        )
        print(f"Saved dendrogram switches plot (groups): {args.dendrogram_switch_groups_output}")
        print(f"Saved dendrogram switches plot (families): {args.dendrogram_switch_families_output}")
        print(f"Saved dendrogram switches plot (subfamilies): {args.dendrogram_switch_subfamilies_output}")


if __name__ == "__main__":
    main()
