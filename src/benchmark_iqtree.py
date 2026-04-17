import argparse
import copy
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
from Bio.SeqRecord import SeqRecord


def resolve_iqtree_binary() -> str:
    if shutil.which("iqtree2") is not None:
        return "iqtree2"

    bundled_binary = Path("iqtree-2.4.0-macOS-arm") / "bin" / "iqtree2"
    if bundled_binary.exists():
        return str(bundled_binary)

    raise FileNotFoundError("Unable to locate iqtree2 on PATH or in the bundled repository binary.")


def _read_alignment_records(alignment_path: Path) -> List[SeqRecord]:
    records = list(SeqIO.parse(str(alignment_path), "fasta"))
    if not records:
        raise ValueError(f"No alignment records found in {alignment_path}")
    return records


def _sample_records(records: Sequence[SeqRecord], subset_size: int, seed: int) -> List[SeqRecord]:
    if subset_size > len(records):
        raise ValueError(f"Requested subset size {subset_size} exceeds available records {len(records)}.")

    rng = random.Random(seed)
    selected_ids = set(record.id for record in rng.sample(list(records), subset_size))
    return [record for record in records if record.id in selected_ids][:subset_size]


def _prune_tree_to_ids(tree_path: Path, selected_ids: Sequence[str], output_tree: Path) -> Path:
    tree = Phylo.read(str(tree_path), "newick")
    selected = set(selected_ids)
    pruned_tree = copy.deepcopy(tree)
    for terminal in list(pruned_tree.get_terminals()):
        if terminal.name not in selected:
            pruned_tree.prune(terminal)

    output_tree.parent.mkdir(parents=True, exist_ok=True)
    Phylo.write(pruned_tree, str(output_tree), "newick")
    return output_tree


def build_subset_alignment_and_tree(
    alignment_path: Path,
    tree_path: Path,
    subset_size: int,
    work_dir: Path,
    seed: int = 42,
) -> Tuple[Path, Path, int]:
    work_dir.mkdir(parents=True, exist_ok=True)
    records = _read_alignment_records(alignment_path)
    sampled_records = _sample_records(records, subset_size=subset_size, seed=seed)
    selected_ids = [record.id for record in sampled_records]

    subset_alignment = work_dir / f"subset_{subset_size}.aln"
    subset_tree = work_dir / f"subset_{subset_size}.tree"

    SeqIO.write(sampled_records, str(subset_alignment), "fasta")
    _prune_tree_to_ids(tree_path=tree_path, selected_ids=selected_ids, output_tree=subset_tree)
    return subset_alignment, subset_tree, len(sampled_records)


def _run_iqtree_asr(alignment_path: Path, tree_path: Path, output_prefix: Path, iqtree_binary: Optional[str] = None) -> None:
    binary = iqtree_binary or resolve_iqtree_binary()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        "-s",
        str(alignment_path),
        "-te",
        str(tree_path),
        "-m",
        "JTT",
        "-asr",
        "-T",
        "AUTO",
        "--prefix",
        str(output_prefix),
    ]
    subprocess.run(cmd, check=True)


def _fit_scaling_curve(sequence_counts: Sequence[int], times: Sequence[float]) -> np.ndarray:
    x = np.asarray(sequence_counts, dtype=float)
    y = np.asarray(times, dtype=float)
    positive_mask = (x > 0) & (y > 0)
    x = x[positive_mask]
    y = y[positive_mask]
    if len(x) < 2:
        return np.array([])

    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    fitted_x = np.linspace(x.min(), x.max(), 200)
    fitted_y = 10 ** (intercept + slope * np.log10(fitted_x))
    return np.column_stack([fitted_x, fitted_y])


def _plot_scaling_results(report: pd.DataFrame, plot_svg: Path) -> None:
    plot_svg.parent.mkdir(parents=True, exist_ok=True)
    fitted = _fit_scaling_curve(report["Sequence_Count"].tolist(), report["ASR_Time(s)"].tolist())

    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.scatter(report["Sequence_Count"], report["ASR_Time(s)"], color="#4C78A8", s=60, zorder=3)
    if fitted.size:
        axis.plot(fitted[:, 0], fitted[:, 1], color="#C44E52", linewidth=2.0, label="Power-law fit")
        axis.legend(frameon=False, loc="upper left")

    axis.set_xlabel("Sequence Count")
    axis.set_ylabel("Time (s)")
    axis.set_title("IQ-TREE2 ASR Scaling")
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_svg, format="svg")
    plt.close(fig)


def run_iqtree_scaling_benchmark(
    alignment_path: Path,
    tree_path: Path,
    subset_sizes: Iterable[int],
    work_dir: Path,
    report_csv: Path,
    plot_svg: Path,
    seed: int = 42,
    iqtree_binary: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    binary = iqtree_binary or resolve_iqtree_binary()

    for subset_size in subset_sizes:
        subset_alignment, subset_tree, sequence_count = build_subset_alignment_and_tree(
            alignment_path=alignment_path,
            tree_path=tree_path,
            subset_size=subset_size,
            work_dir=work_dir / f"subset_{subset_size}",
            seed=seed,
        )
        prefix = work_dir / f"iqtree_{subset_size}"
        start = time.perf_counter()
        _run_iqtree_asr(
            alignment_path=subset_alignment,
            tree_path=subset_tree,
            output_prefix=prefix,
            iqtree_binary=binary,
        )
        elapsed = time.perf_counter() - start
        rows.append({"Sequence_Count": int(sequence_count), "ASR_Time(s)": round(float(elapsed), 4)})

    report = pd.DataFrame(rows)
    report.to_csv(report_csv, index=False)
    _plot_scaling_results(report, plot_svg)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark IQ-TREE2 ASR scaling on subset alignments and trees.")
    parser.add_argument("--alignment", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--tree", default="data/interim/IPR019888.tree")
    parser.add_argument("--work-dir", default="data/interim/iqtree_scaling")
    parser.add_argument("--output", default="results/iqtree_scaling.csv")
    parser.add_argument("--plot", default="results/iqtree_scaling_plot.svg")
    parser.add_argument("--subset-sizes", default="500,1000,2000,4000")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    subset_sizes = [int(value.strip()) for value in args.subset_sizes.split(",") if value.strip()]

    report = run_iqtree_scaling_benchmark(
        alignment_path=Path(args.alignment),
        tree_path=Path(args.tree),
        subset_sizes=subset_sizes,
        work_dir=Path(args.work_dir),
        report_csv=Path(args.output),
        plot_svg=Path(args.plot),
        seed=args.seed,
    )

    print(report.to_string(index=False))


if __name__ == "__main__":
    main()