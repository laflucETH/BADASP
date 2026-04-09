import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
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


def plot_badasp_score_distribution(
    score_path: Path,
    output_svg: Path,
    title: str,
    color: str,
) -> None:
    df = _load_score_table(score_path)
    scores = df["badasp_score"].astype(float).to_numpy()
    threshold = float(df["global_threshold"].iloc[0])

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
    group_scores: Path,
    family_scores: Path,
    subfamily_scores: Path,
    output_svg: Path,
) -> None:
    score_tables = {
        "Groups": _load_score_table(group_scores),
        "Families": _load_score_table(family_scores),
        "Subfamilies": _load_score_table(subfamily_scores),
    }
    colors = {
        "Groups": "#1F77B4",
        "Families": "#D95F02",
        "Subfamilies": "#2CA02C",
    }

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))

    for label, df in score_tables.items():
        scores = df["badasp_score"].astype(float).to_numpy()
        threshold = float(df["global_threshold"].iloc[0])
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
    group_scores: Path,
    family_scores: Path,
    subfamily_scores: Path,
    output_group_svg: Path,
    output_family_svg: Path,
    output_subfamily_svg: Path,
) -> None:
    plot_badasp_score_distribution(
        score_path=group_scores,
        output_svg=output_group_svg,
        title="Groups BADASP Score Distribution",
        color="#1F77B4",
    )
    plot_badasp_score_distribution(
        score_path=family_scores,
        output_svg=output_family_svg,
        title="Families BADASP Score Distribution",
        color="#D95F02",
    )
    plot_badasp_score_distribution(
        score_path=subfamily_scores,
        output_svg=output_subfamily_svg,
        title="Subfamilies BADASP Score Distribution",
        color="#2CA02C",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QC and hierarchical BADASP visualizations.")
    default_length_out, default_gap_out, _ = default_plot_paths()
    default_hier_dist_out, default_hier_switch_out = default_hierarchical_badasp_plot_paths()
    default_group_dist_out, default_family_dist_out, default_subfamily_dist_out = default_individual_badasp_plot_paths()
    parser.add_argument("--fasta", default=None, help="Input FASTA for length distribution plot.")
    parser.add_argument("--length-output", default=str(default_length_out))
    parser.add_argument("--msa", default=None, help="Input MSA FASTA for gap-per-column plot.")
    parser.add_argument("--gap-output", default=str(default_gap_out))
    parser.add_argument("--group-scores", default="results/badasp_scoring/badasp_scores_groups.csv")
    parser.add_argument("--family-scores", default="results/badasp_scoring/badasp_scores_families.csv")
    parser.add_argument("--subfamily-scores", default="results/badasp_scoring/badasp_scores_subfamilies.csv")
    parser.add_argument("--hierarchical-distribution-output", default=str(default_hier_dist_out))
    parser.add_argument("--hierarchical-switch-output", default=str(default_hier_switch_out))
    parser.add_argument("--group-distribution-output", default=str(default_group_dist_out))
    parser.add_argument("--family-distribution-output", default=str(default_family_dist_out))
    parser.add_argument("--subfamily-distribution-output", default=str(default_subfamily_dist_out))
    parser.add_argument("--hierarchical-only", action="store_true")
    args = parser.parse_args()

    if args.fasta and not args.hierarchical_only:
        plot_sequence_length_distribution(Path(args.fasta), Path(args.length_output))
        print(f"Saved length distribution: {args.length_output}")

    if args.msa and not args.hierarchical_only:
        plot_gap_percentage_per_column(Path(args.msa), Path(args.gap_output))
        print(f"Saved gap profile: {args.gap_output}")

    if Path(args.group_scores).exists() and Path(args.family_scores).exists() and Path(args.subfamily_scores).exists():
        plot_individual_hierarchical_badasp_distributions(
            group_scores=Path(args.group_scores),
            family_scores=Path(args.family_scores),
            subfamily_scores=Path(args.subfamily_scores),
            output_group_svg=Path(args.group_distribution_output),
            output_family_svg=Path(args.family_distribution_output),
            output_subfamily_svg=Path(args.subfamily_distribution_output),
        )
        print(f"Saved group score distribution: {args.group_distribution_output}")
        print(f"Saved family score distribution: {args.family_distribution_output}")
        print(f"Saved subfamily score distribution: {args.subfamily_distribution_output}")

        plot_hierarchical_badasp_distributions(
            group_scores=Path(args.group_scores),
            family_scores=Path(args.family_scores),
            subfamily_scores=Path(args.subfamily_scores),
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


if __name__ == "__main__":
    main()
