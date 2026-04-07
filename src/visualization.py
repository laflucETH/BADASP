import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Bio import SeqIO
from scipy.cluster.hierarchy import dendrogram


def default_plot_paths() -> Tuple[Path, Path, Path]:
    return (
        Path("results/sequence_filtering/raw_length_dist.svg"),
        Path("results/alignment_qc/msa_gap_profile.svg"),
        Path("results/topological_clustering/tree_dendrogram.svg"),
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
    color_threshold: Optional[float] = None,
) -> None:
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    z = np.asarray(linkage_matrix, dtype=float)

    plt.figure(figsize=(12, 6))
    dendrogram(
        z,
        truncate_mode="lastp",
        p=max_leaves,
        no_labels=True,
        color_threshold=color_threshold,
    )
    plt.title("Topological Clustering Dendrogram")
    plt.xlabel("Collapsed Leaf Groups")
    plt.ylabel("Cophenetic Distance")
    plt.tight_layout()
    plt.savefig(output_svg, format="svg")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="QC visualizations for sequence lengths and MSA gap profile.")
    default_length_out, default_gap_out, _ = default_plot_paths()
    parser.add_argument("--fasta", default=None, help="Input FASTA for length distribution plot.")
    parser.add_argument("--length-output", default=str(default_length_out))
    parser.add_argument("--msa", default=None, help="Input MSA FASTA for gap-per-column plot.")
    parser.add_argument("--gap-output", default=str(default_gap_out))
    args = parser.parse_args()

    if args.fasta:
        plot_sequence_length_distribution(Path(args.fasta), Path(args.length_output))
        print(f"Saved length distribution: {args.length_output}")

    if args.msa:
        plot_gap_percentage_per_column(Path(args.msa), Path(args.gap_output))
        print(f"Saved gap profile: {args.gap_output}")


if __name__ == "__main__":
    main()
