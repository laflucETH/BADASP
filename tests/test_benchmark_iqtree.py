from pathlib import Path

import pandas as pd

from src.benchmark_iqtree import (
    build_parser,
    build_subset_alignment_and_tree,
    generate_extrapolated_scaling_plot,
    run_iqtree_scaling_benchmark,
)


def test_build_subset_alignment_and_tree_writes_subset_files(tmp_path: Path) -> None:
    alignment = tmp_path / "full.fasta"
    tree = tmp_path / "full.tree"
    work_dir = tmp_path / "work"

    alignment.write_text(
        ">a\nAAAA\n>b\nAAAT\n>c\nAATT\n>d\nTTTT\n>e\nGGGG\n",
        encoding="utf-8",
    )
    tree.write_text("((a:0.1,b:0.1):0.2,(c:0.1,(d:0.1,e:0.1):0.1):0.2)Root;\n", encoding="utf-8")

    subset_alignment, subset_tree, subset_count = build_subset_alignment_and_tree(
        alignment_path=alignment,
        tree_path=tree,
        subset_size=3,
        work_dir=work_dir,
        seed=7,
    )

    assert subset_count == 3
    assert subset_alignment.exists()
    assert subset_tree.exists()
    subset_text = subset_alignment.read_text(encoding="utf-8")
    assert subset_text.count(">") == 3


def test_run_iqtree_scaling_benchmark_writes_report_and_plot(monkeypatch, tmp_path: Path) -> None:
    alignment = tmp_path / "full.fasta"
    tree = tmp_path / "full.tree"
    work_dir = tmp_path / "work"
    report_csv = tmp_path / "iqtree_scaling.csv"
    plot_svg = tmp_path / "iqtree_scaling_plot.svg"

    records = [f">s{i}\nAAAA\n" for i in range(1, 7)]
    alignment.write_text("".join(records), encoding="utf-8")
    tree.write_text("((s1:0.1,s2:0.1):0.2,(s3:0.1,(s4:0.1,(s5:0.1,s6:0.1):0.1):0.1):0.2)Root;\n", encoding="utf-8")

    calls = []

    class _Timer:
        def __init__(self):
            self._value = 0.0

        def __call__(self):
            self._value += 2.5
            return self._value

    monkeypatch.setattr("src.benchmark_iqtree.time.perf_counter", _Timer())

    def _mock_run(cmd, check):
        calls.append(cmd)
        assert check is True
        assert cmd[0] in {"iqtree2", "/opt/homebrew/bin/iqtree2"}
        assert "-asr" in cmd
        assert "-T" in cmd and "AUTO" in cmd

    monkeypatch.setattr("subprocess.run", _mock_run)
    monkeypatch.setattr("src.benchmark_iqtree.resolve_iqtree_binary", lambda: "iqtree2")

    report = run_iqtree_scaling_benchmark(
        alignment_path=alignment,
        tree_path=tree,
        subset_sizes=[2, 4],
        work_dir=work_dir,
        report_csv=report_csv,
        plot_svg=plot_svg,
        seed=7,
    )

    assert report_csv.exists()
    assert plot_svg.exists()
    assert len(report) == 2
    assert set(report.columns) == {"Sequence_Count", "ASR_Time(s)"}
    written = pd.read_csv(report_csv)
    assert list(written["Sequence_Count"]) == [2, 4]
    assert any("-T" in cmd and "AUTO" in cmd for cmd in calls)


def test_build_parser_defaults_subset_sizes() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.subset_sizes == "500,1000,2000,4000"


def test_generate_extrapolated_scaling_plot_reads_report_and_writes_svg(tmp_path: Path) -> None:
    report_csv = tmp_path / "iqtree_scaling.csv"
    plot_svg = tmp_path / "iqtree_scaling_plot_extrapolated.svg"

    report_csv.write_text(
        "Sequence_Count,ASR_Time(s)\n"
        "500,26.0054\n"
        "1000,29.4324\n"
        "2000,32.4825\n"
        "4000,45.2983\n",
        encoding="utf-8",
    )

    generate_extrapolated_scaling_plot(report_csv=report_csv, plot_svg=plot_svg, threshold_count=24608)

    assert plot_svg.exists()
    svg_text = plot_svg.read_text(encoding="utf-8")
    assert "24608" in svg_text
    assert "minutes" in svg_text
    assert "25000" in svg_text