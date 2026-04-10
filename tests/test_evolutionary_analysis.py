from pathlib import Path

import numpy as np
import pandas as pd

from src.evolutionary_analysis import (
    calculate_ca_distance_matrix,
    calculate_lca_depth,
    compute_coevolution_matrix,
    classify_physicochemical_shift,
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
