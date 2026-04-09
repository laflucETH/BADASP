from pathlib import Path

from Bio import SeqIO
import pytest

from src.asr_runner import extract_lca_ancestral_sequences, parse_iqtree_state_sequences


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
