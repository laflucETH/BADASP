from pathlib import Path

from src.sequence_cluster import build_parser, filter_fasta_by_length, run_sequence_clustering


def test_filter_fasta_by_length_keeps_only_expected_range(tmp_path: Path) -> None:
    input_fasta = tmp_path / "raw.fasta"
    output_fasta = tmp_path / "filtered.fasta"

    seq_120 = "A" * 120
    seq_150 = "A" * 150
    seq_190 = "A" * 190
    seq_200 = "A" * 200
    seq_201 = "A" * 201
    input_fasta.write_text(
        (
            f">short\n{seq_120}\n"
            f">ok1\n{seq_150}\n"
            f">ok2\n{seq_190}\n"
            f">ok3\n{seq_200}\n"
            f">long\n{seq_201}\n"
        ),
        encoding="utf-8",
    )

    kept = filter_fasta_by_length(input_fasta, output_fasta, min_len=130, max_len=200)

    assert kept == 3
    filtered_text = output_fasta.read_text(encoding="utf-8")
    assert ">ok1" in filtered_text
    assert ">ok2" in filtered_text
    assert ">ok3" in filtered_text
    assert ">short" not in filtered_text
    assert ">long" not in filtered_text


def test_run_sequence_clustering_invokes_cdhit_and_counts_sequences(monkeypatch, tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    input_fasta.write_text(">a\n" + ("A" * 150) + "\n>b\n" + ("A" * 160) + "\n>c\n" + ("G" * 170) + "\n", encoding="utf-8")

    filtered_fasta = tmp_path / "filtered.fasta"
    output_fasta = tmp_path / "clustered.fasta"

    def _mock_run(cmd, check):
        assert check is True
        assert cmd[0] == "cd-hit"
        assert "-c" in cmd and "0.6" in cmd
        assert "-n" in cmd and "4" in cmd
        assert "-i" in cmd and str(filtered_fasta) in cmd
        assert "-o" in cmd and str(output_fasta) in cmd

        output_fasta.write_text(">cluster1\nAAAA\n>cluster2\nGGGG\n", encoding="utf-8")

    monkeypatch.setattr("subprocess.run", _mock_run)

    surviving = run_sequence_clustering(
        input_fasta=input_fasta,
        filtered_fasta=filtered_fasta,
        output_fasta=output_fasta,
        identity=0.6,
        word_size=4,
        min_len=130,
        max_len=200,
    )

    assert surviving == 2
    assert output_fasta.exists()


def test_build_parser_defaults_identity_to_0_70() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.identity == 0.7
