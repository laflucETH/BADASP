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
    aligner: str = "famsa",
) -> int:
    aligned_output.parent.mkdir(parents=True, exist_ok=True)
    trimmed_output.parent.mkdir(parents=True, exist_ok=True)

    if aligner == "mafft":
        mafft_cmd = ["mafft", "--auto", str(input_fasta)]
        with aligned_output.open("w", encoding="utf-8") as handle:
            subprocess.run(mafft_cmd, check=True, stdout=handle)
    elif aligner == "famsa":
        famsa_cmd = ["famsa", str(input_fasta), str(aligned_output)]
        subprocess.run(famsa_cmd, check=True)
    else:
        raise ValueError(f"Unsupported aligner: {aligner}. Expected one of: mafft, famsa")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align clustered FASTA with MAFFT or FAMSA and trim with trimAl.")
    parser.add_argument("--input", default="data/interim/IPR019888_clustered.fasta")
    parser.add_argument("--aligned", default="data/interim/IPR019888_aligned.aln")
    parser.add_argument("--output", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--gap-threshold", type=float, default=0.2)
    parser.add_argument("--aligner", choices=["mafft", "famsa"], default="famsa")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    trimmed_cols = build_alignment_and_trim(
        input_fasta=Path(args.input),
        aligned_output=Path(args.aligned),
        trimmed_output=Path(args.output),
        gap_threshold=args.gap_threshold,
        aligner=args.aligner,
    )
    print(f"Trimmed alignment columns: {trimmed_cols}")


if __name__ == "__main__":
    main()
