from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import linkage

from src.visualization import (
    _compute_95th_threshold,
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

    group_pairwise = tmp_path / "raw_pairwise_groups.csv"
    family_pairwise = tmp_path / "raw_pairwise_families.csv"
    subfamily_pairwise = tmp_path / "raw_pairwise_subfamilies.csv"
    output_svg = tmp_path / "hierarchical_distributions.svg"

    pd.DataFrame(
        {
            "pair": ["1-2", "1-2", "1-2"],
            "position": [1, 2, 3],
            "score": [0.5, 1.0, 1.5],
        }
    ).to_csv(group_pairwise, index=False)
    pd.DataFrame(
        {
            "pair": ["3-4", "3-4", "3-4"],
            "position": [1, 2, 3],
            "score": [0.2, 0.7, 1.1],
        }
    ).to_csv(family_pairwise, index=False)
    pd.DataFrame(
        {
            "pair": ["5-6", "5-6", "5-6"],
            "position": [1, 2, 3],
            "score": [0.1, 0.4, 0.9],
        }
    ).to_csv(subfamily_pairwise, index=False)

    plot_hierarchical_badasp_distributions(
        group_pairwise=group_pairwise,
        family_pairwise=family_pairwise,
        subfamily_pairwise=subfamily_pairwise,
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


def test_plot_badasp_score_distribution_writes_svg(tmp_path: Path) -> None:
    from src.visualization import plot_badasp_score_distribution

    raw_pairwise_csv = tmp_path / "raw_pairwise_groups.csv"
    output_svg = tmp_path / "badasp_score_distribution_groups.svg"

    pd.DataFrame(
        {
            "pair": ["1-2", "1-2", "1-2", "1-2"],
            "position": [1, 2, 3, 4],
            "score": [0.5, 1.0, 1.5, 2.0],
        }
    ).to_csv(raw_pairwise_csv, index=False)

    plot_badasp_score_distribution(
        raw_pairwise_path=raw_pairwise_csv,
        output_svg=output_svg,
        title="Groups BADASP Score Distribution",
        color="#1F77B4",
    )

    assert output_svg.exists()
    assert output_svg.stat().st_size > 0


def test_build_switch_node_map_counts_threshold_exceeding_events(tmp_path: Path) -> None:
    from src.visualization import build_switch_node_map

    tree_path = tmp_path / "tree.nwk"
    assignments_path = tmp_path / "assignments.csv"
    pairwise_path = tmp_path / "raw_pairwise_groups.csv"

    tree_path.write_text("((A:0.1,B:0.1)N1:0.2,(C:0.1,D:0.1)N2:0.2)Root;\n", encoding="utf-8")
    pd.DataFrame(
        {
            "sequence_id": ["A", "B", "C", "D"],
            "group_id": [1, 1, 2, 2],
            "group_lca_node": ["N1", "N1", "N2", "N2"],
            "family_id": [10, 10, 20, 20],
            "family_lca_node": ["N1", "N1", "N2", "N2"],
            "subfamily_id": [100, 100, 200, 200],
            "subfamily_lca_node": ["N1", "N1", "N2", "N2"],
        }
    ).to_csv(assignments_path, index=False)
    pd.DataFrame(
        {
            "pair": ["1-2", "1-2", "1-2", "1-2"],
            "position": [1, 2, 3, 4],
            "score": [0.1, 0.2, 0.3, 10.0],
        }
    ).to_csv(pairwise_path, index=False)

    node_switches = build_switch_node_map(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=pairwise_path,
        level="groups",
    )

    assert node_switches
    assert sum(node_switches.values()) == 1


def test_plot_tree_with_switches_writes_svg(tmp_path: Path) -> None:
    from src.visualization import plot_tree_with_switches

    tree_path = tmp_path / "tree.nwk"
    out_svg = tmp_path / "tree_switches.svg"

    tree_path.write_text("((A:0.1,B:0.1)N1:0.2,(C:0.1,D:0.1)N2:0.2)Root;\n", encoding="utf-8")

    plot_tree_with_switches(
        tree_path=tree_path,
        node_switch_counts={"N1": 2, "N2": 5},
        output_svg=out_svg,
        title="Switch Events on Tree",
    )

    assert out_svg.exists()
    assert out_svg.stat().st_size > 0


def test_plot_dendrogram_with_switches_writes_svg(tmp_path: Path) -> None:
    from src.visualization import plot_dendrogram_with_switches

    tree_path = tmp_path / "tree.nwk"
    assignments_path = tmp_path / "assignments.csv"
    raw_pairwise_path = tmp_path / "raw_pairwise_groups.csv"
    out_svg = tmp_path / "dendrogram_switches_groups.svg"

    tree_path.write_text("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);\n", encoding="utf-8")
    pd.DataFrame(
        {
            "sequence_id": ["A", "B", "C", "D"],
            "group_id": [1, 1, 2, 2],
            "group_lca_node": ["NA", "NA", "NB", "NB"],
            "family_id": [10, 10, 20, 20],
            "family_lca_node": ["NA", "NA", "NB", "NB"],
            "subfamily_id": [100, 100, 200, 200],
            "subfamily_lca_node": ["NA", "NA", "NB", "NB"],
        }
    ).to_csv(assignments_path, index=False)
    pd.DataFrame(
        {
            "pair": ["1-2", "1-2", "1-2", "1-2"],
            "position": [1, 2, 3, 4],
            "score": [0.1, 0.2, 0.3, 10.0],
        }
    ).to_csv(raw_pairwise_path, index=False)

    plot_dendrogram_with_switches(
        tree_path=tree_path,
        assignments_path=assignments_path,
        raw_pairwise_path=raw_pairwise_path,
        level="groups",
        output_svg=out_svg,
        title="Groups Dendrogram with Switches",
        color_threshold=0.2,
        line_color="#1F77B4",
        terminal_colors={"A": "#1F77B4", "B": "#1F77B4", "C": "#D95F02", "D": "#D95F02"},
    )

    svg_text = out_svg.read_text(encoding="utf-8").lower()
    assert out_svg.exists()
    assert out_svg.stat().st_size > 0
    assert "#1f77b4" in svg_text or "rgb(31, 119, 180)" in svg_text
    assert "branch length from root" in svg_text


def test_plot_topological_tree_dendrogram_keeps_level_color(tmp_path: Path) -> None:
    from src.visualization import plot_topological_tree_dendrogram

    tree_path = tmp_path / "tree.nwk"
    out_svg = tmp_path / "tree_dendrogram.svg"

    tree_path.write_text("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);\n", encoding="utf-8")

    plot_topological_tree_dendrogram(
        tree_path=tree_path,
        output_svg=out_svg,
        title="Tree Dendrogram",
        line_color="#1F77B4",
        terminal_colors={"A": "#1F77B4", "B": "#1F77B4", "C": "#D95F02", "D": "#D95F02"},
    )

    svg_text = out_svg.read_text(encoding="utf-8").lower()
    assert out_svg.exists()
    assert "#1f77b4" in svg_text or "rgb(31, 119, 180)" in svg_text
    assert "branch length from root" in svg_text


def test_compute_95th_threshold_isolated_per_level() -> None:
    groups = _compute_95th_threshold([0.01, 0.02, 0.03, 0.04])
    families = _compute_95th_threshold([0.1, 0.2, 0.3, 0.4])
    subfamilies = _compute_95th_threshold([1.0, 2.0, 3.0, 4.0])

    assert groups != families
    assert families != subfamilies
    assert groups != subfamilies


def test_plot_duplication_distributions_and_switches_write_svg(tmp_path: Path) -> None:
    from src.visualization import (
        plot_duplication_badasp_distribution,
        plot_duplication_switch_counts,
    )

    pairwise = tmp_path / "raw_pairwise_duplications.csv"
    scores = tmp_path / "badasp_scores_duplications.csv"
    dist_svg = tmp_path / "badasp_score_distribution_duplications.svg"
    switch_svg = tmp_path / "switch_counts_duplications.svg"

    pd.DataFrame(
        {
            "pair": ["Node1_L-Node1_R"] * 4,
            "position": [1, 2, 3, 4],
            "score": [0.1, 0.2, 0.3, 1.8],
            "lca_node_name": ["N1"] * 4,
        }
    ).to_csv(pairwise, index=False)

    pd.DataFrame(
        {
            "position": [1, 2, 3, 4],
            "switch_count": [0, 1, 2, 1],
            "global_threshold": [1.0, 1.0, 1.0, 1.0],
            "badasp_score": [0.0, 0.5, 1.2, 0.7],
        }
    ).to_csv(scores, index=False)

    plot_duplication_badasp_distribution(pairwise, dist_svg)
    plot_duplication_switch_counts(scores, switch_svg)

    assert dist_svg.exists()
    assert dist_svg.stat().st_size > 0
    assert switch_svg.exists()
    assert switch_svg.stat().st_size > 0


def test_build_duplication_switch_node_map_counts_lca_threshold_events(tmp_path: Path) -> None:
    from src.visualization import build_duplication_switch_node_map

    pairwise = tmp_path / "raw_pairwise_duplications.csv"
    pd.DataFrame(
        {
            "pair": ["Node1_L-Node1_R"] * 5,
            "position": [1, 2, 3, 4, 5],
            "score": [0.1, 0.2, 0.3, 0.4, 5.0],
            "lca_node_name": ["N1", "N1", "N2", "N2", "N2"],
        }
    ).to_csv(pairwise, index=False)

    node_switches = build_duplication_switch_node_map(pairwise)

    assert sum(node_switches.values()) == 1
    assert node_switches["N2"] == 1
