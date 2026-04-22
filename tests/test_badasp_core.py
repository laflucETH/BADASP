"""Tests for Phase 5 multi-level BADASP scoring (Groups/Families/Subfamilies)."""

from pathlib import Path
import tempfile

import pandas as pd
import pytest

from src.badasp_core import (
    BADASPCore,
    build_hierarchical_sister_pairs,
    calculate_ancestral_conservation,
    compute_multilevel_badasp_scores,
    identify_sdps,
    load_state_file,
    load_reconciliation_events,
    _filter_pairs_by_reconciliation,
    _resolve_hierarchical_lca_nodes,
    _validate_lca_coverage,
)


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_alignment(temp_data_dir):
    alignment_path = temp_data_dir / "test_alignment.aln"
    alignment_path.write_text(
        ">A1\nACDE\n"
        ">A2\nACDE\n"
        ">B1\nACDF\n"
        ">B2\nACDF\n"
        ">C1\nACGE\n"
        ">C2\nACGE\n"
        ">D1\nACGF\n"
        ">D2\nACGF\n",
        encoding="utf-8",
    )
    return alignment_path


@pytest.fixture
def sample_assignments(temp_data_dir):
    assignments_path = temp_data_dir / "assignments.csv"
    df = pd.DataFrame(
        {
            "sequence_id": ["A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2"],
            "group_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "group_lca_node": ["G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2"],
            "family_id": [10, 10, 11, 11, 20, 20, 21, 21],
            "family_lca_node": ["F10", "F10", "F11", "F11", "F20", "F20", "F21", "F21"],
            "subfamily_id": [100, 100, 110, 110, 200, 200, 210, 210],
            "subfamily_lca_node": ["S100", "S100", "S110", "S110", "S200", "S200", "S210", "S210"],
        }
    )
    df.to_csv(assignments_path, index=False)
    return assignments_path


@pytest.fixture
def sample_tree(temp_data_dir):
    tree_path = temp_data_dir / "asr.treefile"
    tree_path.write_text(
        "(((A1:0.1,A2:0.1)S100:0.1,(B1:0.1,B2:0.1)S110:0.1)F10:0.2,"
        "((C1:0.1,C2:0.1)S200:0.1,(D1:0.1,D2:0.1)S210:0.1)F20:0.2)G1:0.3,"
        "((((A1:0.1,A2:0.1)S100:0.1,(B1:0.1,B2:0.1)S110:0.1)F11:0.2,"
        "((C1:0.1,C2:0.1)S200:0.1,(D1:0.1,D2:0.1)S210:0.1)F21:0.2)G2:0.3)Root;",
        encoding="utf-8",
    )
    return tree_path


@pytest.fixture
def sample_ancestral_sequences(temp_data_dir):
    ancestral_path = temp_data_dir / "ancestors.fasta"
    ancestral_path.write_text(
        ">G1\nACDE\n"
        ">G2\nACDF\n"
        ">F10\nACDE\n"
        ">F11\nACDF\n"
        ">F20\nACGE\n"
        ">F21\nACGF\n"
        ">S100\nACDE\n"
        ">S110\nACDF\n"
        ">S200\nACGE\n"
        ">S210\nACGF\n",
        encoding="utf-8",
    )
    return ancestral_path


@pytest.fixture
def sample_state_file(temp_data_dir):
    state_path = temp_data_dir / "asr.state"
    state_path.write_text(
        "Node\tSite\tState\tp_A\tp_R\tp_N\tp_D\tp_C\tp_Q\tp_E\tp_G\tp_H\tp_I\tp_L\tp_K\tp_M\tp_F\tp_P\tp_S\tp_T\tp_W\tp_Y\tp_V\n"
        "G1\t1\tA\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "G1\t2\tC\t0.01\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "G1\t3\tD\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "G1\t4\tE\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.95\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "G2\t1\tA\t0.93\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "G2\t2\tC\t0.01\t0.01\t0.01\t0.01\t0.93\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "G2\t3\tD\t0.01\t0.01\t0.01\t0.93\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\n"
        "G2\t4\tF\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.93\t0.01\t0.01\t0.01\t0.01\t0.01\n",
        encoding="utf-8",
    )
    return state_path


@pytest.fixture
def sample_reconciliation_csv(temp_data_dir):
    reconciliation_path = temp_data_dir / "duplication_nodes.csv"
    pd.DataFrame(
        {
            "node_name": ["G1", "G2", "F10", "F20", "S100", "S110", "S200", "S210"],
            "event_type": ["Speciation", "Duplication", "Duplication", "Duplication", "Duplication", "Speciation", "Duplication", "Duplication"],
        }
    ).to_csv(reconciliation_path, index=False)
    return reconciliation_path


def test_load_state_file_parses_correctly(sample_state_file):
    state_data = load_state_file(sample_state_file)
    assert "G1" in state_data
    assert "G2" in state_data


def test_load_reconciliation_events_parses_csv(sample_reconciliation_csv):
    events = load_reconciliation_events(sample_reconciliation_csv)
    assert events["G1"] == "Speciation"
    assert events["G2"] == "Duplication"
    assert events["S110"] == "Speciation"


def test_filter_pairs_by_reconciliation_skips_speciation_pairs(sample_reconciliation_csv):
    events = load_reconciliation_events(sample_reconciliation_csv)
    pairs = [(1, 2), (3, 4)]
    level_lcas = {1: "G1", 2: "G2", 3: "F10", 4: "F20"}

    kept_pairs, skipped_pairs, skipped_speciation_pairs = _filter_pairs_by_reconciliation(
        pairs=pairs,
        level_lcas=level_lcas,
        reconciliation_events=events,
    )

    assert kept_pairs == [(3, 4)]
    assert skipped_pairs == 1
    assert skipped_speciation_pairs == 1


def test_resolve_hierarchical_lca_nodes_ignores_missing_tree_members(temp_data_dir):
    tree_path = temp_data_dir / "tiny.tree"
    tree_path.write_text("((A:0.1,B:0.1)Node1:0.2,C:0.3)Root;", encoding="utf-8")

    assignments = pd.DataFrame(
        {
            "sequence_id": ["A", "B", "Missing"],
            "group_lca_node": ["G1", "G1", "G1"],
            "family_lca_node": ["F1", "F1", "F1"],
        }
    )

    from Bio import Phylo

    tree = Phylo.read(tree_path, "newick")
    resolved = _resolve_hierarchical_lca_nodes(assignments, tree)

    assert resolved["G1"] == "Node1"
    assert resolved["F1"] == "Node1"


def test_validate_lca_coverage_raises_on_low_coverage():
    with pytest.raises(ValueError, match="coverage"):
        _validate_lca_coverage(
            level="group",
            level_lcas={1: "Node1", 2: "Node2", 3: "Node3", 4: "Node4"},
            ancestral_seqs={"Node1": "AC"},
            min_coverage=0.95,
        )


def test_calculate_ancestral_conservation_binary():
    assert calculate_ancestral_conservation("A", "A") == 1
    assert calculate_ancestral_conservation("A", "F") == -1


def test_build_hierarchical_sister_pairs_respects_parent_constraints(sample_assignments, sample_tree):
    assignments = pd.read_csv(sample_assignments)
    pairs = build_hierarchical_sister_pairs(assignments, sample_tree)

    assert set(pairs.keys()) == {"groups", "families", "subfamilies"}
    assert all(len(pair) == 2 for pair in pairs["groups"])

    # Families should not be paired across different groups.
    group_by_family = assignments.drop_duplicates("family_id").set_index("family_id")["group_id"].to_dict()
    for fam_a, fam_b in pairs["families"]:
        assert group_by_family[fam_a] == group_by_family[fam_b]

    # Subfamilies should not be paired across different families.
    family_by_sub = assignments.drop_duplicates("subfamily_id").set_index("subfamily_id")["family_id"].to_dict()
    for sub_a, sub_b in pairs["subfamilies"]:
        assert family_by_sub[sub_a] == family_by_sub[sub_b]


def test_build_hierarchical_sister_pairs_avoids_all_vs_all_within_parent(temp_data_dir):
    assignments_path = temp_data_dir / "assignments_three_families.csv"
    tree_path = temp_data_dir / "three_families.tree"

    pd.DataFrame(
        {
            "sequence_id": ["A1", "A2", "B1", "B2", "C1", "C2"],
            "group_id": [1, 1, 1, 1, 1, 1],
            "group_lca_node": ["G1", "G1", "G1", "G1", "G1", "G1"],
            "family_id": [10, 10, 11, 11, 12, 12],
            "family_lca_node": ["F10", "F10", "F11", "F11", "F12", "F12"],
            "subfamily_id": [100, 100, 110, 110, 120, 120],
            "subfamily_lca_node": ["S100", "S100", "S110", "S110", "S120", "S120"],
        }
    ).to_csv(assignments_path, index=False)

    # F10 and F11 are nearest sisters; F12 is farther from both.
    tree_path.write_text(
        "(((A1:0.1,A2:0.1)S100:0.1,(B1:0.1,B2:0.1)S110:0.1)F11:0.1,(C1:0.1,C2:0.1)S120:0.6)Root;",
        encoding="utf-8",
    )

    assignments = pd.read_csv(assignments_path)
    pairs = build_hierarchical_sister_pairs(assignments, tree_path)

    # Families in one parent with 3 clades would produce 3 pairs in all-vs-all.
    # Nearest-sister logic should produce fewer unique pairs.
    assert len(pairs["families"]) < 3


def test_compute_multilevel_badasp_scores_structure(
    sample_alignment,
    sample_assignments,
    sample_ancestral_sequences,
    sample_state_file,
    sample_tree,
):
    results = compute_multilevel_badasp_scores(
        alignment_path=sample_alignment,
        assignments_path=sample_assignments,
        ancestral_path=sample_ancestral_sequences,
        state_path=sample_state_file,
        tree_path=sample_tree,
        min_clade_size=1,
    )

    assert set(results.keys()) == {"groups", "families", "subfamilies"}
    for level in results:
        df = results[level]["scores"]
        assert set(df.columns) >= {"position", "max_score", "switch_count", "global_threshold", "badasp_score"}
        assert len(df) == 4


def test_compute_multilevel_badasp_scores_filters_speciation_pairs(
    sample_alignment,
    sample_assignments,
    sample_ancestral_sequences,
    sample_state_file,
    sample_tree,
    sample_reconciliation_csv,
):
    results = compute_multilevel_badasp_scores(
        alignment_path=sample_alignment,
        assignments_path=sample_assignments,
        ancestral_path=sample_ancestral_sequences,
        state_path=sample_state_file,
        tree_path=sample_tree,
        reconciliation_csv=sample_reconciliation_csv,
        min_clade_size=1,
    )

    assert results["families"]["filtered_speciation_pairs"] == 1
    assert not results["families"]["pairwise"].empty
    assert results["groups"]["filtered_speciation_pairs"] >= 0
    assert results["subfamilies"]["filtered_speciation_pairs"] >= 0


def test_identify_sdps_prefers_switch_count():
    scores_df = pd.DataFrame(
        {
            "position": [1, 2, 3, 4],
            "max_score": [0.8, 0.6, 1.2, 1.0],
            "switch_count": [2, 5, 5, 1],
            "global_threshold": [0.75, 0.75, 0.75, 0.75],
            "badasp_score": [0.8, 0.6, 1.2, 1.0],
        }
    )
    sdps, threshold = identify_sdps(scores_df)
    assert threshold == pytest.approx(0.75)
    assert set(sdps["position"]) == {2, 3}


def test_identify_sdps_returns_empty_when_no_switches():
    scores_df = pd.DataFrame(
        {
            "position": [1, 2, 3],
            "max_score": [0.0, 0.0, 0.0],
            "switch_count": [0, 0, 0],
            "global_threshold": [0.0, 0.0, 0.0],
            "badasp_score": [0.0, 0.0, 0.0],
        }
    )
    sdps, threshold = identify_sdps(scores_df)
    assert sdps.empty
    assert threshold == 0.0


def test_badasp_core_writes_three_level_outputs(
    sample_alignment,
    sample_assignments,
    sample_ancestral_sequences,
    sample_state_file,
    sample_tree,
    temp_data_dir,
):
    output_dir = temp_data_dir / "out"

    core = BADASPCore(
        alignment_path=sample_alignment,
        assignments_path=sample_assignments,
        ancestral_path=sample_ancestral_sequences,
        state_path=sample_state_file,
        tree_path=sample_tree,
        min_clade_size=1,
    )

    results = core.compute_scores()
    assert set(results.keys()) == {"groups", "families", "subfamilies"}

    sdps, thresholds = core.identify_sdps()
    assert set(sdps.keys()) == {"groups", "families", "subfamilies"}
    assert set(thresholds.keys()) == {"groups", "families", "subfamilies"}

    core.save_results(output_dir)
    assert (output_dir / "badasp_scores_groups.csv").exists()
    assert (output_dir / "badasp_scores_families.csv").exists()
    assert (output_dir / "badasp_scores_subfamilies.csv").exists()
    assert (output_dir / "raw_pairwise_groups.csv").exists()
    assert (output_dir / "raw_pairwise_families.csv").exists()
    assert (output_dir / "raw_pairwise_subfamilies.csv").exists()


def test_filter_pairs_by_reconciliation_skips_filter_on_name_mismatch():
    """Test that reconciliation filter is disabled when node names don't match.
    
    Regression test for bug where reconciliation from topology tree had Event_* names
    but BADASP used ASR tree Node_* names, causing all pairs to filter.
    """
    pairs = [(1, 2), (3, 4)]
    level_lcas = {1: "Node1", 2: "Node2", 3: "Node3", 4: "Node4"}
    reconciliation_events = {"Event_1": "Duplication", "Event_2": "Duplication"}
    
    filtered_pairs, skipped, skipped_spec = _filter_pairs_by_reconciliation(
        pairs, level_lcas, reconciliation_events
    )
    
    # When no node names match, all pairs should be kept with warning
    assert len(filtered_pairs) == len(pairs)
    assert skipped == 0
    assert skipped_spec == 0


def test_filter_pairs_by_reconciliation_applies_filter_on_name_match():
    """Test that reconciliation filter works normally when node names match."""
    pairs = [(1, 2), (3, 4)]
    level_lcas = {1: "Event_1", 2: "Event_2", 3: "Event_3", 4: "Event_4"}
    reconciliation_events = {
        "Event_1": "Duplication",
        "Event_2": "Duplication",
        "Event_3": "Duplication",
        "Event_4": "Speciation",  # This pair will be skipped
    }
    
    filtered_pairs, skipped, skipped_spec = _filter_pairs_by_reconciliation(
        pairs, level_lcas, reconciliation_events
    )
    
    # Only pair (1,2) should be kept since (3,4) has a Speciation event
    assert len(filtered_pairs) == 1
    assert filtered_pairs[0] == (1, 2)
    assert skipped == 1
    assert skipped_spec == 1
