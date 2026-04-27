from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pdb_mapper import PDBMapper
from src.visualization import plot_duplication_switch_counts


DEFAULT_PERCENTILES: Tuple[int, int] = (97, 99)


def _summarize_threshold(raw_df: pd.DataFrame, percentile: float) -> Tuple[float, pd.DataFrame]:
    scores = pd.to_numeric(raw_df["score"], errors="coerce")
    valid_scores = scores[np.isfinite(scores)]
    threshold = float(np.percentile(valid_scores.to_numpy(dtype=float), percentile)) if not valid_scores.empty else 0.0

    filtered = raw_df[scores >= threshold].copy()
    if filtered.empty:
        empty = pd.DataFrame(columns=["position", "switch_count", "max_score", "global_threshold", "badasp_score"])
        return threshold, empty

    summary = (
        filtered.groupby("position", as_index=False)
        .agg(switch_count=("score", "size"), max_score=("score", "max"))
        .sort_values(["switch_count", "max_score", "position"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    summary["global_threshold"] = threshold
    summary["badasp_score"] = summary["max_score"]
    return threshold, summary


def _write_chimerax_script(
    mapper: PDBMapper,
    alignment_path: Path,
    sdp_csv: Path,
    output_cxc: Path,
    percentile: int,
) -> Path:
    rows, no_switch_reason = mapper._all_switch_rows_from_csv(sdp_csv)
    _, max_switch_count = mapper._switch_count_bounds(rows)
    mapping = mapper.map_alignment_to_structure(alignment_path)

    low_hex = "#FBE6E6"
    high_hex = "#7A0000"
    residue_pairs = mapper._residue_color_pairs(
        rows,
        mapping,
        low_hex=low_hex,
        high_hex=high_hex,
        min_value=0,
        max_value=max_switch_count,
    )
    if rows and not residue_pairs:
        top_alignment_col, top_switch = max(rows, key=lambda item: float(item[1]))
        no_switch_reason = (
            f"{len(rows)} switched alignment columns found, but none mapped to PDB residues "
            f"(top alignment col {int(top_alignment_col)}, switch_count={int(round(top_switch))})"
        )

    level_label = f"Duplications p{percentile}"
    return mapper._build_chimerax_script(
        pdb_path=Path(mapper.download_pdb()),
        residue_pairs=residue_pairs,
        output_path=output_cxc,
        level_label=level_label,
        min_switch_count=0,
        max_switch_count=max_switch_count,
        low_hex=low_hex,
        high_hex=high_hex,
        no_switch_reason=no_switch_reason,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore stricter duplication BADASP thresholds")
    parser.add_argument("--raw-pairwise", default="results/badasp_scoring/raw_pairwise_duplications.csv")
    parser.add_argument("--scores-dir", default="results/badasp_scoring")
    parser.add_argument("--structural-dir", default="results/structural_mapping")
    parser.add_argument("--alignment", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--pdb-id", default="2cg4")
    parser.add_argument("--pdb-file", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raw_pairwise_path = Path(args.raw_pairwise)
    scores_dir = Path(args.scores_dir)
    structural_dir = Path(args.structural_dir)
    alignment_path = Path(args.alignment)

    raw_df = pd.read_csv(raw_pairwise_path)
    mapper = PDBMapper(pdb_id=args.pdb_id, pdb_file=args.pdb_file)

    for percentile in DEFAULT_PERCENTILES:
        threshold, summary = _summarize_threshold(raw_df, percentile)
        sdp_csv = scores_dir / f"badasp_sdps_duplications_p{percentile}.csv"
        svg_path = scores_dir / f"switch_counts_duplications_p{percentile}.svg"
        cxc_path = structural_dir / f"highlight_sdps_duplications_p{percentile}.cxc"

        scores_dir.mkdir(parents=True, exist_ok=True)
        structural_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(sdp_csv, index=False)

        plot_duplication_switch_counts(
            raw_pairwise_path=raw_pairwise_path,
            output_svg=svg_path,
            percentile=float(percentile),
        )
        _write_chimerax_script(
            mapper=mapper,
            alignment_path=alignment_path,
            sdp_csv=sdp_csv,
            output_cxc=cxc_path,
            percentile=percentile,
        )

        print(f"Percentile {percentile}: threshold={threshold:.12f}")
        print("Top 5 positions")
        for _, row in summary.head(5).iterrows():
            print(f"- position={int(row['position'])} switch_count={int(row['switch_count'])}")
        print(f"Saved SDP table: {sdp_csv}")
        print(f"Saved switch SVG: {svg_path}")
        print(f"Saved ChimeraX script: {cxc_path}")


if __name__ == "__main__":
    main()