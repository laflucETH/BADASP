import argparse
import subprocess
from pathlib import Path

from Bio import SeqIO


def count_fasta_records(fasta_path: Path) -> int:
    return sum(1 for line in fasta_path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))


def filter_fasta_by_length(
    input_fasta: Path,
    output_fasta: Path,
    min_len: int = 130,
    max_len: int = 200,
) -> int:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with output_fasta.open("w", encoding="utf-8") as handle:
        for record in SeqIO.parse(str(input_fasta), "fasta"):
            seq_len = len(record.seq)
            if min_len <= seq_len <= max_len:
                SeqIO.write(record, handle, "fasta")
                kept += 1
    return kept


def run_sequence_clustering(
    input_fasta: Path,
    filtered_fasta: Path,
    output_fasta: Path,
    identity: float = 0.8,
    word_size: int = 4,
    min_len: int = 130,
    max_len: int = 200,
) -> int:
    filter_fasta_by_length(
        input_fasta=input_fasta,
        output_fasta=filtered_fasta,
        min_len=min_len,
        max_len=max_len,
    )

    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "cd-hit",
        "-i",
        str(filtered_fasta),
        "-o",
        str(output_fasta),
        "-c",
        str(identity),
        "-n",
        str(word_size),
    ]
    subprocess.run(cmd, check=True)

    return count_fasta_records(output_fasta)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster raw sequences with CD-HIT.")
    parser.add_argument("--input", default="data/raw/IPR019888.fasta")
    parser.add_argument("--filtered", default="data/interim/IPR019888_length_filtered.fasta")
    parser.add_argument("--output", default="data/interim/IPR019888_clustered.fasta")
    parser.add_argument("--identity", type=float, default=0.8)
    parser.add_argument("--word-size", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=130)
    parser.add_argument("--max-len", type=int, default=200)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    surviving = run_sequence_clustering(
        input_fasta=Path(args.input),
        filtered_fasta=Path(args.filtered),
        output_fasta=Path(args.output),
        identity=args.identity,
        word_size=args.word_size,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    kept = count_fasta_records(Path(args.filtered))
    print(f"Length-filtered sequences kept: {kept}")
    print(f"CD-HIT representative sequences: {surviving}")


if __name__ == "__main__":
    main()
