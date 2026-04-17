from pathlib import Path

from src.msa_builder import build_alignment_and_trim, build_parser


def test_build_alignment_and_trim_invokes_famsa_and_trimal_by_default(monkeypatch, tmp_path: Path) -> None:
    input_fasta = tmp_path / "clustered.fasta"
    aligned_fasta = tmp_path / "aligned.aln"
    trimmed_fasta = tmp_path / "trimmed.aln"

    input_fasta.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")

    calls = []

    def _mock_run(cmd, check, stdout=None):
        calls.append(cmd)
        assert check is True

        if cmd[0] == "famsa":
            out_file = Path(cmd[2])
            out_file.write_text(">s1\nAAAA-\n>s2\nAAA-T\n", encoding="utf-8")
            return

        if cmd[0] == "trimal":
            out_idx = cmd.index("-out") + 1
            out_file = Path(cmd[out_idx])
            out_file.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")
            return

        raise AssertionError("Unexpected command")

    monkeypatch.setattr("subprocess.run", _mock_run)

    trimmed_cols = build_alignment_and_trim(
        input_fasta=input_fasta,
        aligned_output=aligned_fasta,
        trimmed_output=trimmed_fasta,
        gap_threshold=0.5,
    )

    assert calls[0][0] == "famsa"
    assert calls[1][0] == "trimal"
    assert "-gt" in calls[1] and "0.5" in calls[1]
    assert trimmed_cols == 4
    assert trimmed_fasta.exists()


def test_build_alignment_and_trim_uses_default_gap_threshold_0_2(monkeypatch, tmp_path: Path) -> None:
    input_fasta = tmp_path / "clustered.fasta"
    aligned_fasta = tmp_path / "aligned.aln"
    trimmed_fasta = tmp_path / "trimmed.aln"

    input_fasta.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")

    calls = []

    def _mock_run(cmd, check, stdout=None):
        calls.append(cmd)
        assert check is True

        if cmd[0] == "famsa":
            out_file = Path(cmd[2])
            out_file.write_text(">s1\nAAAA-\n>s2\nAAA-T\n", encoding="utf-8")
            return

        if cmd[0] == "trimal":
            out_idx = cmd.index("-out") + 1
            out_file = Path(cmd[out_idx])
            out_file.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")
            return

        raise AssertionError("Unexpected command")

    monkeypatch.setattr("subprocess.run", _mock_run)

    _ = build_alignment_and_trim(
        input_fasta=input_fasta,
        aligned_output=aligned_fasta,
        trimmed_output=trimmed_fasta,
    )

    assert "-gt" in calls[1] and "0.2" in calls[1]


def test_build_alignment_and_trim_invokes_famsa_when_selected(monkeypatch, tmp_path: Path) -> None:
    input_fasta = tmp_path / "clustered.fasta"
    aligned_fasta = tmp_path / "aligned.aln"
    trimmed_fasta = tmp_path / "trimmed.aln"

    input_fasta.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")

    calls = []

    def _mock_run(cmd, check, stdout=None):
        calls.append(cmd)
        assert check is True

        if cmd[0] == "famsa":
            out_file = Path(cmd[2])
            out_file.write_text(">s1\nAAAA-\n>s2\nAAA-T\n", encoding="utf-8")
            return

        if cmd[0] == "trimal":
            out_idx = cmd.index("-out") + 1
            out_file = Path(cmd[out_idx])
            out_file.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")
            return

        raise AssertionError("Unexpected command")

    monkeypatch.setattr("subprocess.run", _mock_run)

    trimmed_cols = build_alignment_and_trim(
        input_fasta=input_fasta,
        aligned_output=aligned_fasta,
        trimmed_output=trimmed_fasta,
        aligner="famsa",
    )

    assert calls[0][0] == "famsa"
    assert calls[1][0] == "trimal"
    assert trimmed_cols == 4


def test_build_alignment_and_trim_rejects_unknown_aligner(tmp_path: Path) -> None:
    input_fasta = tmp_path / "clustered.fasta"
    aligned_fasta = tmp_path / "aligned.aln"
    trimmed_fasta = tmp_path / "trimmed.aln"

    input_fasta.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")

    try:
        _ = build_alignment_and_trim(
            input_fasta=input_fasta,
            aligned_output=aligned_fasta,
            trimmed_output=trimmed_fasta,
            aligner="unknown",
        )
        assert False, "Expected ValueError for unsupported aligner"
    except ValueError as exc:
        assert "Unsupported aligner" in str(exc)


def test_build_parser_sets_default_aligner_to_famsa() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.aligner == "famsa"


def test_build_parser_accepts_famsa_aligner() -> None:
    parser = build_parser()
    args = parser.parse_args(["--aligner", "famsa"])
    assert args.aligner == "famsa"
