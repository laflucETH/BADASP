from pathlib import Path

import pandas as pd

from src.benchmark_scaling import choose_word_size_for_identity, run_scaling_benchmark


def test_choose_word_size_for_identity() -> None:
    assert choose_word_size_for_identity(0.80) == 5
    assert choose_word_size_for_identity(0.70) == 5
    assert choose_word_size_for_identity(0.65) == 4
    assert choose_word_size_for_identity(0.55) == 3


def test_run_scaling_benchmark_writes_expected_report(monkeypatch, tmp_path: Path) -> None:
    raw_fasta = tmp_path / "raw.fasta"
    raw_fasta.write_text(">a\nAAAA\n>b\nAAAT\n", encoding="utf-8")

    report_csv = tmp_path / "scaling_benchmark_report.csv"
    work_dir = tmp_path / "work"

    calls = []

    class _Timer:
        def __init__(self):
            self._t = 0.0

        def __call__(self):
            self._t += 1.0
            return self._t

    fake_timer = _Timer()

    def _mock_run(cmd, check, stdout=None, env=None):
        calls.append(cmd)
        assert check is True

        if cmd[0] == "cd-hit":
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_text(">rep1\nAAAA\n", encoding="utf-8")
            return

        if cmd[0] == "famsa":
            Path(cmd[2]).write_text(">rep1\nAAAA\n", encoding="utf-8")
            return

        if cmd[0] in {"FastTree", "FastTreeMP"}:
            stdout.write("(rep1:0.1);\n")
            return

        raise AssertionError("Unexpected command")

    monkeypatch.setattr("subprocess.run", _mock_run)
    monkeypatch.setattr("src.benchmark_scaling.time.perf_counter", fake_timer)

    report = run_scaling_benchmark(
        raw_fasta=raw_fasta,
        thresholds=[0.65, 0.70],
        work_dir=work_dir,
        report_csv=report_csv,
    )

    assert report_csv.exists()
    assert len(report) == 2
    assert set(report.columns) == {
        "CD-HIT_Threshold",
        "Sequence_Count",
        "FAMSA_Time(s)",
        "FastTree_Time(s)",
    }

    written = pd.read_csv(report_csv)
    assert len(written) == 2
    assert list(written["CD-HIT_Threshold"]) == [0.65, 0.70]
    assert all(t > 0 for t in written["FAMSA_Time(s)"])
    assert all(t > 0 for t in written["FastTree_Time(s)"])
    assert any(cmd[0] == "cd-hit" for cmd in calls)
    assert any(cmd[0] == "famsa" for cmd in calls)
    assert any(cmd[0] in {"FastTree", "FastTreeMP"} for cmd in calls)
