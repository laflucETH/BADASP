from pathlib import Path

import numpy as np
import pandas as pd

from src.evolutionary_analysis import (
    _collect_architecture_switch_values,
    _plot_architecture_boxplot,
    _plot_master_dendrogram,
    assign_coevolution_communities,
    calculate_ca_distance_matrix,
    calculate_lca_depth,
    count_switches_per_domain,
    compute_coevolution_matrix,
    classify_physicochemical_shift,
    extract_taxon_label,
    rank_top_functional_sdps,
)


def test_calculate_lca_depth_for_named_node(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    # Root -> N1 has branch length 2.0
    tree_path.write_text("((A:1,B:1)N1:2,(C:1,D:1)N2:2)Root;\n")

    depth = calculate_lca_depth(tree_path=tree_path, member_names=["A", "B"])
    assert np.isclose(depth, 2.0)


def test_calculate_ca_distance_matrix_from_pdb(tmp_path: Path) -> None:
    pdb_path = tmp_path / "toy.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  SER A   3       0.000   4.000   0.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    matrix = calculate_ca_distance_matrix(pdb_path=pdb_path, residue_numbers=[1, 2, 3])
    assert list(matrix.index) == [1, 2, 3]
    assert list(matrix.columns) == [1, 2, 3]
    assert np.isclose(matrix.loc[1, 2], 3.0)
    assert np.isclose(matrix.loc[1, 3], 4.0)
    assert np.isclose(matrix.loc[2, 3], 5.0)


def test_compute_coevolution_matrix_from_branch_events() -> None:
    events = pd.DataFrame(
        {
            "branch_id": ["N1", "N1", "N2", "N3", "N3"],
            "position": [10, 20, 10, 20, 30],
        }
    )

    matrix = compute_coevolution_matrix(events_df=events)
    assert set(matrix.index) == {10, 20, 30}
    assert set(matrix.columns) == {10, 20, 30}
    assert np.isclose(matrix.loc[10, 10], 1.0)
    assert np.isclose(matrix.loc[20, 20], 1.0)
    assert matrix.loc[10, 20] > matrix.loc[10, 30]
    assert np.isclose(matrix.loc[10, 20], matrix.loc[20, 10])


def test_classify_physicochemical_shift_multiple() -> None:
    category = classify_physicochemical_shift(
        charge_change="neutral->positive",
        hydrophobicity_change="polar->hydrophobic",
        volume_delta=60.0,
    )
    assert category == "multiple_complex"


def test_rank_top_functional_sdps_synthesizes_components() -> None:
    subfamily_scores = pd.DataFrame(
        {
            "position": [10, 20, 30],
            "switch_count": [8, 5, 2],
            "max_score": [1.3, 1.2, 1.1],
        }
    )
    coevo = pd.DataFrame(
        [[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]],
        index=[10, 20, 30],
        columns=[10, 20, 30],
    )
    shifts = pd.DataFrame(
        {
            "position": [10, 20],
            "major_transition_count": [7, 1],
            "charge_change": ["neutral->positive", "neutral->neutral"],
            "hydrophobicity_change": ["polar->hydrophobic", "polar->polar"],
            "volume_change": [55.0, 2.0],
        }
    )

    ranked = rank_top_functional_sdps(
        subfamily_scores_df=subfamily_scores,
        coevolution_matrix_df=coevo,
        shifts_df=shifts,
        top_n=3,
    )

    assert list(ranked.columns).count("position") == 1
    assert ranked.iloc[0]["position"] == 10
    assert ranked.iloc[0]["shift_type"] == "multiple_complex"
    assert ranked.iloc[0]["functional_sdp_score"] >= ranked.iloc[1]["functional_sdp_score"]


def test_count_switches_per_domain() -> None:
    events = pd.DataFrame({"position": [5, 20, 22, 200, 240, 241]})
    domains = {
        "RAM_domain": [1, 170],
        "DNA_binding_domain": [171, 280],
        "Recognition_helix": [230, 245],
    }
    counts = count_switches_per_domain(events, domains)

    assert counts["RAM_domain"] == 3
    assert counts["DNA_binding_domain"] == 3
    assert counts["Recognition_helix"] == 2


def test_collect_architecture_switch_values_uses_raw_counts() -> None:
    scores = pd.DataFrame(
        {
            "position": list(range(1, 41)),
            "switch_count": [2] * 10 + [0] * 10 + [1] * 10 + [3] * 10,
        }
    )
    domains = {
        "HTH_Scaffold": [1, 10],
        "Recognition_Helix": [11, 20],
        "HTH_Linker": [21, 30],
        "RAM_domain": [31, 40],
    }

    values = _collect_architecture_switch_values(scores, domains)

    assert values["HTH_Scaffold"] == [2] * 10
    assert values["Recognition_Helix"] == [0] * 10
    assert values["HTH_Linker"] == [1] * 10
    assert values["RAM_domain"] == [3] * 10


def test_plot_architecture_boxplot_writes_mean_based_svg(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        {
            "position": list(range(1, 41)),
            "switch_count": [2] * 10 + [0] * 10 + [1] * 10 + [3] * 10,
        }
    )
    domains = {
        "HTH_Scaffold": [1, 10],
        "Recognition_Helix": [11, 20],
        "HTH_Linker": [21, 30],
        "RAM_domain": [31, 40],
    }
    output_svg = tmp_path / "architectural_boxplot_groups.svg"

    _plot_architecture_boxplot(scores, domains, output_svg, level="groups")

    svg_text = output_svg.read_text()
    assert output_svg.exists()
    assert "Switch Count" in svg_text
    assert "Architectural Domain" in svg_text
    assert "Holm-corrected" not in svg_text


def test_assign_coevolution_communities() -> None:
    matrix = pd.DataFrame(
        [
            [1.0, 0.85, 0.05, 0.01],
            [0.85, 1.0, 0.04, 0.02],
            [0.05, 0.04, 1.0, 0.8],
            [0.01, 0.02, 0.8, 1.0],
        ],
        index=[10, 11, 50, 51],
        columns=[10, 11, 50, 51],
    )

    communities = assign_coevolution_communities(matrix, distance_cut=0.4)
    assert set(communities["position"].astype(int)) == {10, 11, 50, 51}
    assert communities["community_id"].nunique() == 2


def test_extract_taxon_label() -> None:
    header = "sp|Q9XYZ1|ARAC_ECOLI AraC transcriptional regulator OS=Escherichia coli OX=562"
    assert extract_taxon_label(header) == "Escherichia coli"


def test_plot_master_dendrogram_exports_svg(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    tree_path.write_text("((A:0.1,B:0.1)N1:0.3,(C:0.1,D:0.1)N2:0.3)Root;\n")

    events_by_level = {
        "groups": pd.DataFrame({"branch_id": ["N1"], "position": [10], "score": [1.2]}),
        "families": pd.DataFrame({"branch_id": ["N2", "N2"], "position": [20, 21], "score": [1.0, 1.1]}),
        "subfamilies": pd.DataFrame({"branch_id": ["N1"], "position": [30], "score": [1.3]}),
    }

    output_svg = tmp_path / "master.svg"
    _plot_master_dendrogram(tree_path=tree_path, events_by_level=events_by_level, output_svg=output_svg)

    assert output_svg.exists()
    assert output_svg.stat().st_size > 0
