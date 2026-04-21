from pathlib import Path

from Bio import SeqIO
import pytest

from src.asr_runner import extract_lca_ancestral_sequences, parse_iqtree_state_sequences, run_asr_pipeline, run_iqtree_asr, wait_for_file


def test_parse_iqtree_state_sequences_builds_node_sequences(tmp_path: Path) -> None:
    state_file = tmp_path / "asr.state"
    state_file.write_text(
        "Node\tSite\tState\n"
        "Node1\t1\tA\n"
        "Node1\t2\tC\n"
        "Node2\t1\tG\n"
        "Node2\t2\tT\n",
        encoding="utf-8",
    )

    node_sequences = parse_iqtree_state_sequences(state_file)

    assert node_sequences["Node1"] == "AC"
    assert node_sequences["Node2"] == "GT"


def test_extract_lca_ancestral_sequences_writes_hierarchical_unique_nodes(tmp_path: Path) -> None:
    tree_file = tmp_path / "tree.nwk"
    assignments_csv = tmp_path / "assignments.csv"
    out_fasta = tmp_path / "ancestors.fasta"

    tree_file.write_text("((A:0.1,B:0.1)Node1:0.2,(C:0.1,D:0.1)Node2:0.2)Root;\n", encoding="utf-8")
    assignments_csv.write_text(
        "sequence_id,group_id,group_lca_node,family_id,family_lca_node,subfamily_id,subfamily_lca_node\n"
        "A,1,Root,10,Node1,100,Node1\n"
        "B,1,Root,10,Node1,101,Node1\n"
        "C,1,Root,11,Node2,102,Node2\n"
        "D,1,Root,11,Node2,103,Node2\n",
        encoding="utf-8",
    )

    node_sequences = {"Node1": "AC", "Node2": "GT", "Root": "TT"}

    written = extract_lca_ancestral_sequences(
        tree_path=tree_file,
        assignments_csv=assignments_csv,
        node_sequences=node_sequences,
        output_fasta=out_fasta,
        min_clade_size=2,
    )

    records = list(SeqIO.parse(str(out_fasta), "fasta"))
    assert written == 3
    assert {record.id for record in records} == {"Node1", "Node2", "Root"}


def test_extract_lca_ancestral_sequences_uses_asr_mapping_csv(tmp_path: Path) -> None:
    tree_file = tmp_path / "tree.nwk"
    assignments_csv = tmp_path / "assignments.csv"
    mapping_csv = tmp_path / "tree_clusters_asr_mapped.csv"
    out_fasta = tmp_path / "ancestors.fasta"

    tree_file.write_text("((A:0.1,B:0.1)Node1:0.2,(C:0.1,D:0.1)Node2:0.2)Root;\n", encoding="utf-8")
    assignments_csv.write_text(
        "sequence_id,group_id,group_lca_node,family_id,family_lca_node,subfamily_id,subfamily_lca_node\n"
        "A,1,InternalNode_1,10,InternalNode_2,100,InternalNode_3\n"
        "B,1,InternalNode_1,10,InternalNode_2,101,InternalNode_3\n",
        encoding="utf-8",
    )
    mapping_csv.write_text(
        "clade_id,member_count,lca_node,lca_node_asr\n"
        "1,2,InternalNode_1,Node1\n"
        "2,2,InternalNode_2,Node2\n"
        "3,2,InternalNode_3,Root\n",
        encoding="utf-8",
    )

    node_sequences = {"Node1": "AC", "Node2": "GT", "Root": "TT"}

    written = extract_lca_ancestral_sequences(
        tree_path=tree_file,
        assignments_csv=assignments_csv,
        node_sequences=node_sequences,
        output_fasta=out_fasta,
        asr_mapping_csv=mapping_csv,
        min_clade_size=1,
    )

    records = list(SeqIO.parse(str(out_fasta), "fasta"))
    assert written == 3
    assert {record.id for record in records} == {"Node1", "Node2", "Root"}


def test_extract_lca_ancestral_sequences_uses_cdhit_cluster_mapping(tmp_path: Path) -> None:
    tree_file = tmp_path / "tree.nwk"
    assignments_csv = tmp_path / "assignments.csv"
    cluster_map = tmp_path / "clustered.fasta.clstr"
    out_fasta = tmp_path / "ancestors.fasta"

    tree_file.write_text("((RepA:0.1,RepB:0.1)Node1:0.2,RepC:0.2)Root;\n", encoding="utf-8")
    assignments_csv.write_text(
        "sequence_id,group_id,group_lca_node,family_id,family_lca_node,subfamily_id,subfamily_lca_node\n"
        "raw1,1,Node1,10,Node1,100,Node1\n"
        "raw2,1,Node1,10,Node1,100,Node1\n",
        encoding="utf-8",
    )
    cluster_map.write_text(
        ">Cluster 0\n"
        "0\t100aa, >RepA *\n"
        "1\t100aa, >raw1 at 99.00%\n"
        ">Cluster 1\n"
        "0\t100aa, >RepB *\n"
        "1\t100aa, >raw2 at 99.00%\n",
        encoding="utf-8",
    )

    node_sequences = {"RepA": "AC", "RepB": "GT", "Node1": "TT"}

    written = extract_lca_ancestral_sequences(
        tree_path=tree_file,
        assignments_csv=assignments_csv,
        node_sequences=node_sequences,
        output_fasta=out_fasta,
        cluster_mapping_csv=cluster_map,
        min_clade_size=1,
    )

    records = list(SeqIO.parse(str(out_fasta), "fasta"))
    assert written == 1
    assert [record.id for record in records] == ["Node1"]


def test_extract_lca_ancestral_sequences_raises_when_hierarchical_lca_missing(tmp_path: Path) -> None:
    tree_file = tmp_path / "tree.nwk"
    assignments_csv = tmp_path / "assignments.csv"
    out_fasta = tmp_path / "ancestors.fasta"

    tree_file.write_text("((A:0.1,B:0.1)Node1:0.2,(C:0.1,D:0.1)Node2:0.2)Root;\n", encoding="utf-8")
    assignments_csv.write_text(
        "sequence_id,group_id,group_lca_node,family_id,family_lca_node,subfamily_id,subfamily_lca_node\n"
        "A,1,Root,10,Node1,100,NodeMissing\n",
        encoding="utf-8",
    )

    node_sequences = {"Node1": "AC", "Node2": "GT", "Root": "TT"}

    with pytest.raises(KeyError, match="A"):
        extract_lca_ancestral_sequences(
            tree_path=tree_file,
            assignments_csv=assignments_csv,
            node_sequences=node_sequences,
            output_fasta=out_fasta,
            min_clade_size=1,
        )


def test_run_iqtree_asr_uses_openmp_threads_flag(monkeypatch, tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.aln"
    tree = tmp_path / "tree.nwk"
    prefix = tmp_path / "asr_run"

    alignment.write_text(">a\nAAAA\n", encoding="utf-8")
    tree.write_text("(a:0.1)Root;\n", encoding="utf-8")

    calls = []

    def _mock_run(cmd, check):
        calls.append(cmd)
        assert check is True

    monkeypatch.setattr("subprocess.run", _mock_run)

    run_iqtree_asr(alignment_path=alignment, tree_path=tree, output_prefix=prefix)

    assert calls
    assert "-T" in calls[0]
    assert "AUTO" in calls[0]
    assert "-nt" not in calls[0]


def test_run_asr_pipeline_runs_once_and_waits_for_state(monkeypatch, tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.aln"
    tree = tmp_path / "tree.nwk"
    assignments = tmp_path / "assignments.csv"
    output_fasta = tmp_path / "ancestors.fasta"
    prefix = tmp_path / "asr_run"
    state_file = prefix.with_suffix(".state")
    treefile = prefix.with_suffix(".treefile")

    alignment.write_text(">a\nAAAA\n", encoding="utf-8")
    tree.write_text("(a:0.1)Root;\n", encoding="utf-8")
    assignments.write_text(
        "sequence_id,group_id,group_lca_node,family_id,family_lca_node,subfamily_id,subfamily_lca_node\n"
        "a,1,Root,1,Root,1,Root\n",
        encoding="utf-8",
    )

    calls = []

    def _mock_run_iqtree_asr(alignment_path, tree_path, output_prefix, iqtree_binary="iqtree2", model="LG+G", threads="AUTO"):
        calls.append((alignment_path, tree_path, output_prefix, iqtree_binary, model, threads))
        state_file.write_text("Node\tSite\tState\nRoot\t1\tA\n", encoding="utf-8")
        treefile.write_text("(a:0.1)Root;\n", encoding="utf-8")

    wait_calls = []

    def _mock_wait_for_file(path, timeout_seconds=300, poll_interval=1.0):
        wait_calls.append(path)
        assert path == state_file
        return path

    def _mock_parse_state(path):
        assert path == state_file
        return {"Root": "A"}

    def _mock_extract(
        tree_path,
        assignments_csv,
        node_sequences,
        output_fasta,
        asr_mapping_csv=None,
        cluster_mapping_csv=None,
        min_clade_size=5,
    ):
        assert tree_path == treefile
        assert assignments_csv == assignments
        assert node_sequences == {"Root": "A"}
        assert asr_mapping_csv is None
        assert cluster_mapping_csv is None
        output_fasta.write_text(">Root\nA\n", encoding="utf-8")
        return 1

    monkeypatch.setattr("src.asr_runner.run_iqtree_asr", _mock_run_iqtree_asr)
    monkeypatch.setattr("src.asr_runner.wait_for_file", _mock_wait_for_file)
    monkeypatch.setattr("src.asr_runner.parse_iqtree_state_sequences", _mock_parse_state)
    monkeypatch.setattr("src.asr_runner.extract_lca_ancestral_sequences", _mock_extract)

    written = run_asr_pipeline(
        alignment_path=alignment,
        tree_path=tree,
        assignments_csv=assignments,
        output_fasta=output_fasta,
        output_prefix=prefix,
        reuse_existing=False,
    )

    assert written == 1
    assert len(calls) == 1
    assert wait_calls == [state_file]
    assert output_fasta.read_text(encoding="utf-8").strip() == ">Root\nA"


def test_wait_for_file_returns_when_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "ready.state"
    path.write_text("done\n", encoding="utf-8")

    resolved = wait_for_file(path, timeout_seconds=1, poll_interval=0.0)

    assert resolved == path
