"""
TDD-first tests for restricted BADASP scoring (Phase 5).

Tests cover the Bradley-style adaptation:
- Recent conservation (RC) calculation using substitution similarity across modern sequences
- Ancestral conservation (AC) as a binary identical/different call for sister clades
- Posterior probability extraction from IQ-TREE state files
- Score assembly using Score = RC - (AC * p(AC))
- Specificity Determining Position (SDP) identification
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from Bio import SeqIO, AlignIO
import numpy as np
from src.badasp_core import (
    BADASPCore,
    load_state_file,
    parse_clade_assignments,
    calculate_recent_conservation,
    calculate_ancestral_conservation,
    extract_posterior_probability,
    compute_badasp_scores,
    identify_sdps,
)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_alignment(temp_data_dir):
    """Create a minimal test alignment for BADASP testing."""
    alignment_path = temp_data_dir / "test_alignment.aln"
    align_content = """>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
ACDEFGHIKLMNPQRSTVWY
>seq3
ADDEFGHIKLMNPQRSTVWY
>seq4
ACDEFGHIKLMNPQRSTVWY
>seq5
ACDEFGHIKLMSPQRSTVWY
"""
    with open(alignment_path, "w") as f:
        f.write(align_content)
    return alignment_path


@pytest.fixture
def sample_clade_assignments(temp_data_dir):
    """Create test clade assignment file with 3 clades."""
    assignments_path = temp_data_dir / "test_assignments.csv"
    assignments = pd.DataFrame({
        "terminal_name": ["seq1", "seq2", "seq3", "seq4", "seq5"],
        "clade_id": [0, 0, 1, 1, 2],
    })
    assignments.to_csv(assignments_path, index=False)
    return assignments_path, assignments


@pytest.fixture
def sample_ancestral_sequences(temp_data_dir):
    """Create test ancestral sequences FASTA."""
    ancestral_path = temp_data_dir / "test_ancestral.fasta"
    ancestral_content = """>Node10
ACDEFGHIKLMNPQRSTVWY
>Node11
AC-EFGHIKLMNPQRSTVWY
>Node12
ACDEFGHIKL-NPQRSTVWY
"""
    with open(ancestral_path, "w") as f:
        f.write(ancestral_content)
    return ancestral_path


@pytest.fixture
def sample_clusters(temp_data_dir):
    """Create a minimal clade-to-LCA mapping table."""
    clusters_path = temp_data_dir / "test_clusters.csv"
    clusters = pd.DataFrame({
        "clade_id": [0, 1, 2],
        "member_count": [2, 2, 1],
        "lca_node": ["Node10", "Node11", "Node12"],
    })
    clusters.to_csv(clusters_path, index=False)
    return clusters_path, clusters


@pytest.fixture
def sample_state_file(temp_data_dir):
    """Create a minimal IQ-TREE state file with posterior probabilities."""
    state_path = temp_data_dir / "test.state"
    # Format: Node Site State p_A p_R p_N p_D p_C p_Q p_E p_G p_H p_I p_L p_K p_M p_F p_P p_S p_T p_W p_Y p_V
    state_content = """# Ancestral state reconstruction for test
# Node:  Node name in the tree
# Site:  Alignment site ID
# State: Most likely state assignment
# p_X:   Posterior probability for state X
Node10\t1\tA\t0.9\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01
Node10\t2\tC\t0.01\t0.01\t0.01\t0.01\t0.9\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01
Node11\t1\tA\t0.85\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01
Node11\t2\tC\t0.01\t0.01\t0.01\t0.01\t0.88\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01
Node12\t1\tA\t0.92\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01
Node12\t2\tC\t0.01\t0.01\t0.01\t0.01\t0.91\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01\t0.01
"""
    with open(state_path, "w") as f:
        f.write(state_content)
    return state_path


def test_load_state_file_parses_correctly(sample_state_file):
    """Test that state file is parsed into a dictionary of DataFrames by node."""
    state_data = load_state_file(sample_state_file)
    assert "Node10" in state_data
    assert "Node11" in state_data
    assert "Node12" in state_data
    
    # Check Node10 data
    node10_df = state_data["Node10"]
    assert len(node10_df) == 2
    assert list(node10_df["Site"]) == [1, 2]
    assert list(node10_df["State"]) == ["A", "C"]


def test_parse_clade_assignments_valid_file(sample_clade_assignments):
    """Test parsing clade assignments from CSV."""
    assignments_path, expected_df = sample_clade_assignments
    assignments = parse_clade_assignments(assignments_path)
    
    pd.testing.assert_frame_equal(assignments, expected_df)


def test_calculate_recent_conservation_basic():
    """Test RC calculation should reflect substitution similarity, not raw identity fraction."""
    conservative_site = ["A", "A", "S"]
    divergent_site = ["A", "A", "V"]

    rc_conservative = calculate_recent_conservation(conservative_site, position=0)
    rc_divergent = calculate_recent_conservation(divergent_site, position=0)

    assert rc_conservative > rc_divergent, (
        "RC should assign higher similarity to conservative substitutions than to more divergent ones"
    )
    assert rc_conservative != pytest.approx(rc_divergent), (
        "RC should not collapse to the same raw identity fraction for distinct substitution patterns"
    )


def test_calculate_ancestral_conservation_binary():
    """Test AC must be binary for sister clade ancestral residues."""
    assert calculate_ancestral_conservation("A", "A") == 1
    assert calculate_ancestral_conservation("A", "W") == -1


def test_extract_posterior_probability_basic(sample_state_file):
    """Test extraction of posterior probability from state file."""
    state_data = load_state_file(sample_state_file)
    
    # Extract probability for Node10, Site 1, amino acid A
    prob = extract_posterior_probability(state_data, "Node10", 1, "A")
    assert prob == pytest.approx(0.9)
    
    # Extract probability for Node10, Site 2, amino acid C
    prob = extract_posterior_probability(state_data, "Node10", 2, "C")
    assert prob == pytest.approx(0.9)


def test_extract_posterior_probability_missing_node():
    """Test extraction with missing node returns 0."""
    state_data = {}
    prob = extract_posterior_probability(state_data, "NonexistentNode", 1, "A")
    assert prob == 0.0


def test_compute_badasp_scores_structure(sample_alignment, sample_clade_assignments,
                                          sample_ancestral_sequences, sample_state_file,
                                          sample_clusters):
    """Test that BADASP scoring produces expected output structure."""
    assignments_path, _ = sample_clade_assignments
    clusters_path, _ = sample_clusters
    
    scores = compute_badasp_scores(
        alignment_path=sample_alignment,
        assignments_path=assignments_path,
        ancestral_path=sample_ancestral_sequences,
        state_path=sample_state_file,
        clusters_path=clusters_path,
        min_clade_size=1
    )
    
    # Check that result is a DataFrame
    assert isinstance(scores, pd.DataFrame)
    
    # Check for expected columns
    expected_cols = {"position", "rc", "ac", "p_ac", "badasp_score"}
    assert set(scores.columns) >= expected_cols, f"Missing columns. Expected {expected_cols}, got {set(scores.columns)}"
    
    # Check that we have rows for alignment positions
    assert len(scores) > 0, "Should have scores for positions"


def test_compute_badasp_scores_uses_bradley_formula(
    sample_alignment,
    sample_clade_assignments,
    sample_ancestral_sequences,
    sample_state_file,
    sample_clusters,
    monkeypatch,
):
    """Test the Phase 5 score must use Score = RC - (AC * p(AC))."""
    assignments_path, _ = sample_clade_assignments
    clusters_path, _ = sample_clusters

    monkeypatch.setattr("src.badasp_core.calculate_recent_conservation", lambda sequences, position: 0.8)
    monkeypatch.setattr("src.badasp_core.calculate_ancestral_conservation", lambda aa1, aa2: 1)
    monkeypatch.setattr("src.badasp_core.extract_posterior_probability", lambda state_data, node, site, aa: 0.9)

    scores = compute_badasp_scores(
        alignment_path=sample_alignment,
        assignments_path=assignments_path,
        ancestral_path=sample_ancestral_sequences,
        state_path=sample_state_file,
        clusters_path=clusters_path,
        min_clade_size=1,
    )

    expected_score = 0.8 - (1 * 0.9)
    assert scores.loc[0, "badasp_score"] == pytest.approx(expected_score)
    assert scores.loc[0, "rc"] == pytest.approx(0.8)
    assert scores.loc[0, "ac"] == pytest.approx(1)
    assert scores.loc[0, "p_ac"] == pytest.approx(0.9)


def test_identify_sdps_threshold():
    """Test that SDPs are identified at 95th percentile."""
    # Create mock scores with known distribution
    np.random.seed(42)  # For reproducibility
    low_scores = np.random.uniform(0, 0.3, 95)
    high_scores = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
    all_scores = np.concatenate([low_scores, high_scores])
    
    scores_df = pd.DataFrame({
        "position": range(1, len(all_scores) + 1),
        "badasp_score": all_scores
    })
    
    sdps, threshold = identify_sdps(scores_df)
    
    # 95th percentile should filter approximately top 5% (5% of 100 ≈ 5 positions)
    assert 4 <= len(sdps) <= 6, f"Expected around 5 SDPs (top 5%), got {len(sdps)}"
    assert threshold > 0, f"Threshold should be positive, got {threshold}"
    assert all(sdps["badasp_score"] >= threshold), "All SDPs should be at or above threshold"


def test_identify_sdps_dataframe_columns():
    """Test that SDP output has correct columns."""
    scores_df = pd.DataFrame({
        "position": [1, 2, 3, 4, 5],
        "badasp_score": [0.1, 0.2, 0.3, 0.4, 0.9]
    })
    
    sdps, _ = identify_sdps(scores_df)
    
    assert "position" in sdps.columns
    assert "badasp_score" in sdps.columns


def test_badasp_core_integration(sample_alignment, sample_clade_assignments,
                                  sample_ancestral_sequences, sample_state_file):
    """Integration test: BADASPCore runs end-to-end without errors."""
    assignments_path, _ = sample_clade_assignments
    
    core = BADASPCore(
        alignment_path=sample_alignment,
        assignments_path=assignments_path,
        ancestral_path=sample_ancestral_sequences,
        state_path=sample_state_file,
        min_clade_size=1
    )
    
    scores = core.compute_scores()
    sdps, threshold = core.identify_sdps()
    
    assert isinstance(scores, pd.DataFrame)
    assert isinstance(sdps, pd.DataFrame)
    assert isinstance(threshold, (int, float))
    assert threshold >= 0, "Threshold should be non-negative"
    # Note: threshold may be 0 in edge cases with small test data
