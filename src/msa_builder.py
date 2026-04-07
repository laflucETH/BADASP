import argparse
import subprocess
from pathlib import Path

from Bio import SeqIO


def _alignment_column_count(alignment_path: Path) -> int:
    lengths = [len(record.seq) for record in SeqIO.parse(str(alignment_path), "fasta")]
    if not lengths:
        return 0
    if len(set(lengths)) != 1:
        raise ValueError("Trimmed alignment has inconsistent sequence lengths.")
    return lengths[0]


def build_alignment_and_trim(
    input_fasta: Path,
    aligned_output: Path,
    trimmed_output: Path,
    gap_threshold: float = 0.2,
) -> int:
    aligned_output.parent.mkdir(parents=True, exist_ok=True)
    trimmed_output.parent.mkdir(parents=True, exist_ok=True)

    mafft_cmd = ["mafft", "--auto", str(input_fasta)]
    with aligned_output.open("w", encoding="utf-8") as handle:
        subprocess.run(mafft_cmd, check=True, stdout=handle)

    trimal_cmd = [
        "trimal",
        "-in",
        str(aligned_output),
        "-out",
        str(trimmed_output),
        "-gt",
        str(gap_threshold),
    ]
    subprocess.run(trimal_cmd, check=True)

    return _alignment_column_count(trimmed_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align clustered FASTA with MAFFT and trim with trimAl.")
    parser.add_argument("--input", default="data/interim/IPR019888_clustered.fasta")
    parser.add_argument("--aligned", default="data/interim/IPR019888_aligned.aln")
    parser.add_argument("--output", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--gap-threshold", type=float, default=0.2)
    args = parser.parse_args()

    trimmed_cols = build_alignment_and_trim(
        input_fasta=Path(args.input),
        aligned_output=Path(args.aligned),
        trimmed_output=Path(args.output),
        gap_threshold=args.gap_threshold,
    )
    print(f"Trimmed alignment columns: {trimmed_cols}")


if __name__ == "__main__":
    main()
