from pathlib import Path

import pandas as pd
from Bio import Phylo
import pytest
from scipy.cluster.hierarchy import fcluster, linkage

from src.tree_cluster import choose_distance_threshold, cluster_tree_topologically


def test_choose_distance_threshold_targets_reasonable_clade_range() -> None:
    points = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [10.0], [10.2], [10.3], [15.0], [15.1], [15.2]]
    linkage_matrix = linkage(points, method="average")
    linkage_rows = linkage_matrix.tolist()

    threshold = choose_distance_threshold(
        linkage_rows,
        target_min_clades=3,
        target_max_clades=6,
        min_clade_size=1,
    )
    cluster_ids = fcluster(linkage_matrix, t=threshold, criterion="distance")
    clade_count = len(set(int(x) for x in cluster_ids))

    assert 3 <= clade_count <= 6


def test_cluster_tree_topologically_writes_multilevel_assignments(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"
    rooted_tree_out = tmp_path / "midpoint_rooted.tree"

    tree_path.write_text(
        "(((A:0.05,B:0.05):0.2,(C:0.05,D:0.05):0.2):0.4,((E:0.05,F:0.05):0.2,(G:0.05,H:0.05):0.2):0.4);\n",
        encoding="utf-8",
    )

    level_counts, level_thresholds = cluster_tree_topologically(
        tree_path=tree_path,
        clusters_output=clusters_csv,
        assignments_output=assignments_csv,
        min_clade_size=1,
        rooted_tree_output=rooted_tree_out,
        group_target_min_clades=2,
        group_target_max_clades=3,
        family_target_min_clades=3,
        family_target_max_clades=5,
        subfamily_target_min_clades=4,
        subfamily_target_max_clades=8,
    )

    assert clusters_csv.exists()
    assert assignments_csv.exists()
    assert rooted_tree_out.exists()
    parsed_tree = Phylo.read(str(rooted_tree_out), "newick")
    assert parsed_tree.root is not None

    assert set(level_counts) == {"group", "family", "subfamily"}
    assert set(level_thresholds) == {"group", "family", "subfamily"}

    # Coarse-to-fine hierarchy should produce non-decreasing cluster counts.
    assert level_counts["group"] <= level_counts["family"] <= level_counts["subfamily"]

    assignments = pd.read_csv(assignments_csv)
    expected_cols = {
        "sequence_id",
        "group_id",
        "group_lca_node",
        "family_id",
        "family_lca_node",
        "subfamily_id",
        "subfamily_lca_node",
    }
    assert set(assignments.columns) == expected_cols
    assert len(assignments) == 8

    # Every sequence should map through all three hierarchy levels.
    assert assignments["group_id"].notna().all()
    assert assignments["family_id"].notna().all()
    assert assignments["subfamily_id"].notna().all()

    # Nesting rules: one subfamily -> one family, one family -> one group.
    sub_to_family = assignments.groupby("subfamily_id")["family_id"].nunique()
    fam_to_group = assignments.groupby("family_id")["group_id"].nunique()
    assert (sub_to_family == 1).all()
    assert (fam_to_group == 1).all()


def test_cluster_tree_topologically_filters_small_clades(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"

    tree_path.write_text("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No clades survive"):
        cluster_tree_topologically(
            tree_path=tree_path,
            clusters_output=clusters_csv,
            assignments_output=assignments_csv,
            min_clade_size=5,
            group_target_min_clades=2,
            group_target_max_clades=2,
            family_target_min_clades=3,
            family_target_max_clades=3,
            subfamily_target_min_clades=4,
            subfamily_target_max_clades=4,
        )