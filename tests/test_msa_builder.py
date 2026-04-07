from pathlib import Path

from src.msa_builder import build_alignment_and_trim


def test_build_alignment_and_trim_invokes_mafft_and_trimal(monkeypatch, tmp_path: Path) -> None:
    input_fasta = tmp_path / "clustered.fasta"
    aligned_fasta = tmp_path / "aligned.aln"
    trimmed_fasta = tmp_path / "trimmed.aln"

    input_fasta.write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")

    calls = []

    def _mock_run(cmd, check, stdout=None):
        calls.append(cmd)
        assert check is True

        if cmd[0] == "mafft":
            stdout.write(">s1\nAAAA-\n>s2\nAAA-T\n")
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

    assert calls[0][0] == "mafft"
    assert "--auto" in calls[0]
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

        if cmd[0] == "mafft":
            stdout.write(">s1\nAAAA-\n>s2\nAAA-T\n")
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
