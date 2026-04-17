import argparse
import subprocess
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:
    from src.sequence_cluster import count_fasta_records
    from src.tree_builder import build_fasttree
except ModuleNotFoundError:
    from sequence_cluster import count_fasta_records
    from tree_builder import build_fasttree


def choose_word_size_for_identity(identity: float) -> int:
    if identity >= 0.7:
        return 5
    if identity >= 0.6:
        return 4
    if identity >= 0.5:
        return 3
    return 2


def _run_cdhit(input_fasta: Path, output_fasta: Path, identity: float) -> int:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    word_size = choose_word_size_for_identity(identity)
    cmd = [
        "cd-hit",
        "-i",
        str(input_fasta),
        "-o",
        str(output_fasta),
        "-c",
        str(identity),
        "-n",
        str(word_size),
    ]
    subprocess.run(cmd, check=True)
    return count_fasta_records(output_fasta)


def _run_famsa(input_fasta: Path, aligned_output: Path) -> float:
    aligned_output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    subprocess.run(["famsa", str(input_fasta), str(aligned_output)], check=True)
    return time.perf_counter() - start


def _run_fasttree(aligned_fasta: Path, tree_output: Path) -> float:
    tree_output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    build_fasttree(trimmed_alignment=aligned_fasta, tree_output=tree_output)
    return time.perf_counter() - start


def run_scaling_benchmark(
    raw_fasta: Path,
    thresholds: Iterable[float],
    work_dir: Path,
    report_csv: Path,
) -> pd.DataFrame:
    rows: List[dict] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    for threshold in thresholds:
        tag = f"{int(threshold * 100):02d}"
        clustered_fasta = work_dir / f"clustered_c{tag}.fasta"
        aligned_fasta = work_dir / f"aligned_c{tag}.aln"
        tree_output = work_dir / f"tree_c{tag}.nwk"

        sequence_count = _run_cdhit(raw_fasta, clustered_fasta, threshold)
        famsa_time = _run_famsa(clustered_fasta, aligned_fasta)
        fasttree_time = _run_fasttree(aligned_fasta, tree_output)

        rows.append(
            {
                "CD-HIT_Threshold": float(threshold),
                "Sequence_Count": int(sequence_count),
                "FAMSA_Time(s)": round(float(famsa_time), 4),
                "FastTree_Time(s)": round(float(fasttree_time), 4),
            }
        )

    report = pd.DataFrame(rows)
    report.to_csv(report_csv, index=False)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark scaling behavior across CD-HIT thresholds with FAMSA and FastTree.")
    parser.add_argument("--input", default="data/raw/IPR019888.fasta")
    parser.add_argument("--work-dir", default="data/interim/scaling_benchmark")
    parser.add_argument("--output", default="results/scaling_benchmark_report.csv")
    parser.add_argument("--thresholds", default="0.65,0.70,0.75,0.80")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    thresholds = [float(token.strip()) for token in args.thresholds.split(",") if token.strip()]
    report = run_scaling_benchmark(
        raw_fasta=Path(args.input),
        thresholds=thresholds,
        work_dir=Path(args.work_dir),
        report_csv=Path(args.output),
    )

    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
