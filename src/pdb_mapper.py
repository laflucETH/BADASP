from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBList, PDBParser, Polypeptide


class PDBMapper:
    """Map BADASP alignment positions to structure residue numbering."""

    def __init__(
        self,
        pdb_id: str,
        pdb_file: Optional[str] = None,
        cache_dir: str = "data/raw",
    ) -> None:
        self.pdb_id = pdb_id.lower()
        self.pdb_file = Path(pdb_file) if pdb_file else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_pdb(self) -> str:
        """Download a PDB file using Biopython or reuse a cached local copy."""
        if self.pdb_file:
            return str(self.pdb_file)

        canonical_pdb = self.cache_dir / f"{self.pdb_id}.pdb"
        if canonical_pdb.exists() and canonical_pdb.stat().st_size > 0:
            self.pdb_file = canonical_pdb
            return str(self.pdb_file)

        pdbl = PDBList(verbose=False)
        fetched = pdbl.retrieve_pdb_file(
            self.pdb_id,
            pdir=str(self.cache_dir),
            file_format="pdb",
            overwrite=False,
        )
        fetched_path = Path(fetched)

        # Normalize to a predictable cache filename for stable reuse.
        if fetched_path != canonical_pdb:
            canonical_pdb.write_text(fetched_path.read_text())

        self.pdb_file = canonical_pdb
        return str(self.pdb_file)

    def _extract_representative_msa_sequence(self, alignment_path: Path) -> str:
        records = list(SeqIO.parse(str(alignment_path), "fasta"))
        if not records:
            raise ValueError(f"No sequences found in alignment: {alignment_path}")

        # Use the sequence with the fewest gaps as representative.
        representative = min(records, key=lambda r: str(r.seq).count("-"))
        return str(representative.seq)

    def _extract_pdb_sequence_and_residue_numbers(self, pdb_path: Path) -> Tuple[str, List[int]]:
        if pdb_path.suffix.lower() == ".cif":
            structure = MMCIFParser(QUIET=True).get_structure(self.pdb_id, str(pdb_path))
        else:
            structure = PDBParser(QUIET=True).get_structure(self.pdb_id, str(pdb_path))

        residues: List[Tuple[str, int]] = []
        for model in structure:
            for chain in model:
                chain_residues: List[Tuple[str, int]] = []
                for residue in chain:
                    if not Polypeptide.is_aa(residue, standard=True):
                        continue
                    resseq = int(residue.id[1])
                    aa = Polypeptide.protein_letters_3to1.get(residue.resname.upper(), None)
                    if aa is None:
                        continue
                    chain_residues.append((aa, resseq))
                if chain_residues:
                    residues = chain_residues
                    break
            if residues:
                break

        if not residues:
            raise ValueError(f"No protein residues found in structure: {pdb_path}")

        sequence = "".join(aa for aa, _ in residues)
        numbers = [n for _, n in residues]
        return sequence, numbers

    def map_alignment_to_structure(self, alignment_path: Path) -> Dict[int, int]:
        """Map alignment column index (1-based) to PDB residue number."""
        pdb_path = Path(self.download_pdb())
        msa_seq_gapped = self._extract_representative_msa_sequence(alignment_path)
        pdb_seq, pdb_residue_numbers = self._extract_pdb_sequence_and_residue_numbers(pdb_path)

        ungapped_to_msa_col: List[int] = []
        msa_ungapped_chars: List[str] = []
        for idx, aa in enumerate(msa_seq_gapped, start=1):
            if aa != "-":
                msa_ungapped_chars.append(aa)
                ungapped_to_msa_col.append(idx)
        msa_seq_ungapped = "".join(msa_ungapped_chars)

        if not msa_seq_ungapped or not pdb_seq:
            return {}

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -0.5

        alignment = aligner.align(msa_seq_ungapped, pdb_seq)[0]
        mapping: Dict[int, int] = {}

        msa_blocks, pdb_blocks = alignment.aligned
        for (msa_start, msa_end), (pdb_start, pdb_end) in zip(msa_blocks, pdb_blocks):
            block_len = min(msa_end - msa_start, pdb_end - pdb_start)
            for offset in range(block_len):
                msa_ungapped_index = msa_start + offset
                pdb_index = pdb_start + offset
                if msa_ungapped_index >= len(ungapped_to_msa_col):
                    continue
                if pdb_index >= len(pdb_residue_numbers):
                    continue
                msa_col = ungapped_to_msa_col[msa_ungapped_index]
                mapping[msa_col] = int(pdb_residue_numbers[pdb_index])

        return mapping

    def _top_positions_from_csv(self, csv_path: Path, top_n: int) -> List[int]:
        if not csv_path.exists():
            return []

        df = pd.read_csv(csv_path)
        if "position" not in df.columns:
            return []

        if "switch_count" in df.columns and "max_score" in df.columns:
            df = df.sort_values(["switch_count", "max_score"], ascending=[False, False])
        elif "switch_count" in df.columns:
            df = df.sort_values(["switch_count"], ascending=[False])

        return [int(p) for p in df["position"].head(top_n).tolist()]

    def _top_switch_rows_from_csv(self, csv_path: Path, top_n: int) -> List[Tuple[int, float]]:
        """Return top positions with switch counts (or fallback score) for gradient coloring."""
        if not csv_path.exists():
            return []

        df = pd.read_csv(csv_path)
        if "position" not in df.columns:
            return []

        value_col = "switch_count" if "switch_count" in df.columns else "max_score"
        if value_col is None or value_col not in df.columns:
            return []

        if "switch_count" in df.columns and "max_score" in df.columns:
            df = df.sort_values(["switch_count", "max_score"], ascending=[False, False])
        else:
            df = df.sort_values([value_col], ascending=[False])

        top_df = df[["position", value_col]].head(top_n)
        return [(int(pos), float(val)) for pos, val in top_df.itertuples(index=False, name=None)]

    @staticmethod
    def _hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
        color = color_hex.lstrip("#")
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

    def _interpolate_hex(self, low_hex: str, high_hex: str, fraction: float) -> str:
        low = self._hex_to_rgb(low_hex)
        high = self._hex_to_rgb(high_hex)
        frac = max(0.0, min(1.0, fraction))
        rgb = (
            int(round(low[0] + (high[0] - low[0]) * frac)),
            int(round(low[1] + (high[1] - low[1]) * frac)),
            int(round(low[2] + (high[2] - low[2]) * frac)),
        )
        return self._rgb_to_hex(rgb)

    def _residue_color_pairs(
        self,
        top_rows: Sequence[Tuple[int, float]],
        mapping: Dict[int, int],
        low_hex: str,
        high_hex: str,
    ) -> List[Tuple[int, str]]:
        scored_residues: Dict[int, float] = {}
        for msa_pos, value in top_rows:
            if msa_pos not in mapping:
                continue
            residue = mapping[msa_pos]
            scored_residues[residue] = max(value, scored_residues.get(residue, float("-inf")))

        if not scored_residues:
            return []

        values = list(scored_residues.values())
        min_val = min(values)
        max_val = max(values)
        scale = max(max_val - min_val, 1e-9)

        residue_colors: List[Tuple[int, str]] = []
        for residue, value in sorted(scored_residues.items()):
            fraction = (value - min_val) / scale
            residue_colors.append((residue, self._interpolate_hex(low_hex, high_hex, fraction)))
        return residue_colors

    @staticmethod
    def _format_residue_selector(residues: Sequence[int]) -> str:
        return "+".join(str(r) for r in sorted(set(residues)))

    def generate_pymol_script(
        self,
        alignment_path: Path,
        sdp_csv_groups: Path,
        sdp_csv_families: Path,
        sdp_csv_subfamilies: Path,
        output_pml: Path,
        top_n: int = 10,
    ) -> Path:
        """Generate a PyMOL script highlighting mapped SDP residues by hierarchy."""
        output_pml.parent.mkdir(parents=True, exist_ok=True)
        pdb_path = Path(self.download_pdb())
        mapping = self.map_alignment_to_structure(alignment_path)

        group_msa = self._top_positions_from_csv(sdp_csv_groups, top_n=top_n)
        family_msa = self._top_positions_from_csv(sdp_csv_families, top_n=top_n)
        subfamily_msa = self._top_positions_from_csv(sdp_csv_subfamilies, top_n=top_n)

        group_res = [mapping[p] for p in group_msa if p in mapping]
        family_res = [mapping[p] for p in family_msa if p in mapping]
        subfamily_res = [mapping[p] for p in subfamily_msa if p in mapping]

        lines: List[str] = [
            "# BADASP hierarchical SDP highlighting",
            f"# source_pdb: {pdb_path}",
            "# groups: red, families: blue, subfamilies: green",
            f"load {pdb_path}",
            "hide everything",
            "show cartoon",
            "color white, polymer.protein",
        ]

        if group_res:
            selector = self._format_residue_selector(group_res)
            lines.append(f"select group_sdps, resi {selector}")
            lines.append("color red, group_sdps")
        else:
            lines.append("# group_sdps: no mapped residues")

        if family_res:
            selector = self._format_residue_selector(family_res)
            lines.append(f"select family_sdps, resi {selector}")
            lines.append("color blue, family_sdps")
        else:
            lines.append("# family_sdps: no mapped residues")

        if subfamily_res:
            selector = self._format_residue_selector(subfamily_res)
            lines.append(f"select subfamily_sdps, resi {selector}")
            lines.append("color green, subfamily_sdps")
        else:
            lines.append("# subfamily_sdps: no mapped residues")

        output_pml.write_text("\n".join(lines) + "\n")
        return output_pml

    def generate_chimerax_script(
        self,
        alignment_path: Path,
        sdp_csv_groups: Path,
        sdp_csv_families: Path,
        sdp_csv_subfamilies: Path,
        output_cxc: Path,
        top_n: int = 10,
    ) -> Path:
        """Generate a ChimeraX command script for mapped SDP highlighting."""
        output_cxc.parent.mkdir(parents=True, exist_ok=True)
        pdb_path = Path(self.download_pdb())
        mapping = self.map_alignment_to_structure(alignment_path)

        group_rows = self._top_switch_rows_from_csv(sdp_csv_groups, top_n=top_n)
        family_rows = self._top_switch_rows_from_csv(sdp_csv_families, top_n=top_n)
        subfamily_rows = self._top_switch_rows_from_csv(sdp_csv_subfamilies, top_n=top_n)

        # Family and subfamily are always combined on the same structure.
        family_pairs = self._residue_color_pairs(
            family_rows,
            mapping,
            low_hex="#FDE047",  # yellow
            high_hex="#DC2626",  # red
        )
        subfamily_pairs = self._residue_color_pairs(
            subfamily_rows,
            mapping,
            low_hex="#67E8F9",  # cyan
            high_hex="#1D4ED8",  # blue
        )
        # Optional group overlay as an accent gradient.
        group_pairs = self._residue_color_pairs(
            group_rows,
            mapping,
            low_hex="#E9D5FF",  # light purple
            high_hex="#7E22CE",  # purple
        )

        all_highlighted = sorted(
            set([r for r, _ in family_pairs] + [r for r, _ in subfamily_pairs] + [r for r, _ in group_pairs])
        )

        lines = [
            "# BADASP Phase 6 publication-quality structural mapping",
            f"open {pdb_path}",
            "set bgColor white",
            "lighting soft",
            "graphics silhouettes true color black width 1.5",
            "material dull",
            "show cartoon",
            "hide atoms",
            "color protein white",
        ]

        if all_highlighted:
            residue_selector = ",".join(str(r) for r in all_highlighted)
            lines.append(f"show :{residue_selector} atoms")
            lines.append(f"style :{residue_selector} stick")
        else:
            lines.append("# no mapped SDP residues to display as atoms")

        lines.append("# Family SDPs warm gradient (yellow -> red) by switch_count")
        for residue, color_hex in family_pairs:
            lines.append(f"color :{residue} {color_hex}")

        lines.append("# Subfamily SDPs cool gradient (cyan -> blue) by switch_count")
        for residue, color_hex in subfamily_pairs:
            lines.append(f"color :{residue} {color_hex}")

        if group_pairs:
            lines.append("# Group SDPs optional accent gradient (light purple -> purple)")
            for residue, color_hex in group_pairs:
                lines.append(f"color :{residue} {color_hex}")

        lines.extend(
            [
                "# legend",
                f"# family_residues: {','.join(str(r) for r, _ in family_pairs) if family_pairs else 'none'}",
                f"# subfamily_residues: {','.join(str(r) for r, _ in subfamily_pairs) if subfamily_pairs else 'none'}",
                f"# group_residues: {','.join(str(r) for r, _ in group_pairs) if group_pairs else 'none'}",
            ]
        )

        output_cxc.write_text("\n".join(lines) + "\n")
        return output_cxc


def _resolve_sdp_csv(base_dir: Path, level: str) -> Path:
    preferred = base_dir / f"badasp_sdps_{level}.csv"
    if preferred.exists():
        return preferred
    return base_dir / f"badasp_scores_{level}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6 structural mapping for BADASP SDPs")
    parser.add_argument("--pdb-id", default="2cg4", help="Target PDB identifier")
    parser.add_argument("--pdb-file", default=None, help="Optional local PDB/CIF path")
    parser.add_argument(
        "--alignment",
        default="data/interim/IPR019888_trimmed.aln",
        help="Trimmed alignment FASTA path",
    )
    parser.add_argument(
        "--scores-dir",
        default="results/badasp_scoring",
        help="Directory containing BADASP score/SDP CSVs",
    )
    parser.add_argument(
        "--output-cxc",
        default="results/structural_mapping/highlight_sdps.cxc",
        help="Output ChimeraX script path",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N SDPs per level")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    mapper = PDBMapper(pdb_id=args.pdb_id, pdb_file=args.pdb_file)
    scores_dir = Path(args.scores_dir)
    alignment = Path(args.alignment)

    groups_csv = _resolve_sdp_csv(scores_dir, "groups")
    families_csv = _resolve_sdp_csv(scores_dir, "families")
    subfamilies_csv = _resolve_sdp_csv(scores_dir, "subfamilies")

    output_cxc = mapper.generate_chimerax_script(
        alignment_path=alignment,
        sdp_csv_groups=groups_csv,
        sdp_csv_families=families_csv,
        sdp_csv_subfamilies=subfamilies_csv,
        output_cxc=Path(args.output_cxc),
        top_n=args.top_n,
    )
    print(f"Generated ChimeraX script: {output_cxc}")


if __name__ == "__main__":
    main()
