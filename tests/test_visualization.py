from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import linkage

from src.visualization import (
    default_plot_paths,
    compute_gap_percentages,
    plot_gap_percentage_per_column,
    plot_sequence_length_distribution,
    plot_topological_dendrogram,
)


def test_plot_sequence_length_distribution_writes_svg(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    out_svg = tmp_path / "lengths.svg"

    fasta.write_text(
        ">a\nAAAA\n>b\nAAAAAA\n>c\nAAAAAAAA\n",
        encoding="utf-8",
    )

    plot_sequence_length_distribution(fasta, out_svg)

    assert out_svg.exists()
    assert out_svg.stat().st_size > 0


def test_compute_gap_percentages_returns_expected_values(tmp_path: Path) -> None:
    aln = tmp_path / "input.aln"
    aln.write_text(
        ">s1\nA-C\n>s2\nACC\n>s3\nA-C\n",
        encoding="utf-8",
    )

    gap_pct = compute_gap_percentages(aln)

    assert len(gap_pct) == 3
    assert gap_pct[0] == 0.0
    assert round(gap_pct[1], 2) == 66.67
    assert gap_pct[2] == 0.0


def test_plot_gap_percentage_per_column_writes_svg(tmp_path: Path) -> None:
    aln = tmp_path / "input.aln"
    out_svg = tmp_path / "gap_profile.svg"

    aln.write_text(
        ">s1\nA-C\n>s2\nACC\n>s3\nA-C\n",
        encoding="utf-8",
    )

    plot_gap_percentage_per_column(aln, out_svg)

    assert out_svg.exists()
    assert out_svg.stat().st_size > 0


def test_default_plot_paths_use_descriptive_result_directories() -> None:
    length_out, gap_out, dendrogram_out = default_plot_paths()

    assert str(length_out).endswith("results/sequence_filtering/raw_length_dist.svg")
    assert str(gap_out).endswith("results/alignment_qc/msa_gap_profile.svg")
    assert str(dendrogram_out).endswith("results/topological_clustering/tree_dendrogram.svg")


def test_plot_topological_dendrogram_writes_svg(tmp_path: Path) -> None:
    out_svg = tmp_path / "dendrogram.svg"
    linkage_matrix = linkage([[0.0], [0.1], [1.0], [1.1]], method="average")

    plot_topological_dendrogram(linkage_matrix, out_svg, max_leaves=4, color_threshold=0.5)

    assert out_svg.exists()
    assert out_svg.stat().st_size > 0


def test_plot_hierarchical_badasp_distributions_writes_svg(tmp_path: Path) -> None:
    from src.visualization import plot_hierarchical_badasp_distributions

    group_scores = tmp_path / "groups.csv"
    family_scores = tmp_path / "families.csv"
    subfamily_scores = tmp_path / "subfamilies.csv"
    output_svg = tmp_path / "hierarchical_distributions.svg"

    pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.5, 1.0, 1.5],
            "switch_count": [1, 2, 3],
            "global_threshold": [1.2, 1.2, 1.2],
            "badasp_score": [0.5, 1.0, 1.5],
        }
    ).to_csv(group_scores, index=False)
    pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.2, 0.7, 1.1],
            "switch_count": [0, 1, 2],
            "global_threshold": [0.9, 0.9, 0.9],
            "badasp_score": [0.2, 0.7, 1.1],
        }
    ).to_csv(family_scores, index=False)
    pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.1, 0.4, 0.9],
            "switch_count": [2, 1, 0],
            "global_threshold": [0.6, 0.6, 0.6],
            "badasp_score": [0.1, 0.4, 0.9],
        }
    ).to_csv(subfamily_scores, index=False)

    plot_hierarchical_badasp_distributions(
        group_scores=group_scores,
        family_scores=family_scores,
        subfamily_scores=subfamily_scores,
        output_svg=output_svg,
    )

    assert output_svg.exists()
    assert output_svg.stat().st_size > 0


def test_plot_hierarchical_switch_counts_writes_svg(tmp_path: Path) -> None:
    from src.visualization import plot_hierarchical_switch_counts

    group_scores = tmp_path / "groups.csv"
    family_scores = tmp_path / "families.csv"
    subfamily_scores = tmp_path / "subfamilies.csv"
    output_svg = tmp_path / "hierarchical_switch_counts.svg"

    pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.5, 1.0, 1.5],
            "switch_count": [1, 2, 3],
            "global_threshold": [1.2, 1.2, 1.2],
            "badasp_score": [0.5, 1.0, 1.5],
        }
    ).to_csv(group_scores, index=False)
    pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.2, 0.7, 1.1],
            "switch_count": [0, 1, 2],
            "global_threshold": [0.9, 0.9, 0.9],
            "badasp_score": [0.2, 0.7, 1.1],
        }
    ).to_csv(family_scores, index=False)
    pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.1, 0.4, 0.9],
            "switch_count": [2, 1, 0],
            "global_threshold": [0.6, 0.6, 0.6],
            "badasp_score": [0.1, 0.4, 0.9],
        }
    ).to_csv(subfamily_scores, index=False)

    plot_hierarchical_switch_counts(
        group_scores=group_scores,
        family_scores=family_scores,
        subfamily_scores=subfamily_scores,
        output_svg=output_svg,
    )

    assert output_svg.exists()
    assert output_svg.stat().st_size > 0
