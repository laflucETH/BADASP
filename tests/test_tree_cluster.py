from pathlib import Path

from scipy.cluster.hierarchy import fcluster, linkage

from src.tree_cluster import choose_distance_threshold, cluster_tree_topologically


def test_cluster_tree_topologically_creates_monophyletic_clusters_and_lcas(tmp_path: Path) -> None:
    tree_path = tmp_path / "toy.tree"
    clusters_csv = tmp_path / "tree_clusters.csv"
    assignments_csv = tmp_path / "tree_cluster_assignments.csv"

    tree_path.write_text("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);\n", encoding="utf-8")

    clade_count, used_threshold = cluster_tree_topologically(
        tree_path=tree_path,
        clusters_output=clusters_csv,
        assignments_output=assignments_csv,
        distance_threshold=0.25,
    )

    assert clade_count == 2
    assert used_threshold == 0.25
    assert clusters_csv.exists()
    assert assignments_csv.exists()

    clusters_content = clusters_csv.read_text(encoding="utf-8")
    assert "clade_id,member_count,lca_node" in clusters_content
    assert ",2," in clusters_content

    assignments_content = assignments_csv.read_text(encoding="utf-8")
    assert "terminal_name,clade_id" in assignments_content
    for terminal in ("A", "B", "C", "D"):
        assert f"{terminal}," in assignments_content


def test_choose_distance_threshold_targets_reasonable_clade_range() -> None:
    points = [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [10.0], [10.2], [10.3], [15.0], [15.1], [15.2]]
    linkage_matrix = linkage(points, method="average")
    linkage_rows = linkage_matrix.tolist()

    threshold = choose_distance_threshold(linkage_rows, target_min_clades=3, target_max_clades=6)
    cluster_ids = fcluster(linkage_matrix, t=threshold, criterion="distance")
    clade_count = len(set(int(x) for x in cluster_ids))

    assert 3 <= clade_count <= 6