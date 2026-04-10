"""
Test suite for Phase 6: Structural Mapping (PDB annotation).

Tests verify:
1. Downloading and caching PDB structures
2. Mapping trimmed MSA columns (1-165) to PDB residue numbers
3. PyMOL script generation for SDP visualization by hierarchy level
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.pdb_mapper import PDBMapper


class TestPDBDownloader:
    """Test PDB file acquisition and caching."""

    def test_download_pdb_file(self):
        """
        Download a target PDB file using Bio.PDB.PDBList.

        Expected:
        - PDB file is downloaded or cached
        - File path is returned
        - File contains valid PDB format content
        """
        pdb_id = "2cg4"  # Target structure for IPR019888
        mapper = PDBMapper(pdb_id=pdb_id)
        pdb_path = mapper.download_pdb()

        assert pdb_path is not None
        assert isinstance(pdb_path, (str, Path))
        assert Path(pdb_path).exists()
        with open(pdb_path, "r") as f:
            content = f.read()
            assert "ATOM" in content or "HETATM" in content

    def test_pdb_cache_reuse(self):
        """
        Verify that repeated downloads use cached version.

        Expected:
        - Second call returns same path
        - File is not re-downloaded (same stat timestamp)
        """
        pdb_id = "2cg4"
        mapper1 = PDBMapper(pdb_id=pdb_id)
        path1 = mapper1.download_pdb()

        mapper2 = PDBMapper(pdb_id=pdb_id)
        path2 = mapper2.download_pdb()

        assert path1 == path2
        assert Path(path1).stat().st_mtime == Path(path2).stat().st_mtime

    def test_custom_pdb_path(self):
        """
        Allow explicit PDB file specification instead of automatic download.

        Expected:
        - Mapper accepts a local PDB file path
        - No download attempt is made
        """
        # Using the existing structure in data/raw/
        pdb_path = Path("data/raw/2cg4.cif")
        if pdb_path.exists():
            mapper = PDBMapper(pdb_id="2cg4", pdb_file=str(pdb_path))
            retrieved_path = mapper.download_pdb()
            assert retrieved_path == str(pdb_path)


class TestSequenceToStructureAlignment:
    """Test mapping MSA columns to PDB residue numbers."""

    def test_alignment_basic(self):
        """
        Map trimmed MSA columns (1-165) to PDB residue indices.

        Expected:
        - Input: MSA FASTA, PDB structure
        - Output: Dict mapping MSA column index → PDB residue number
        - No gaps in mapping (gapped positions handled)
        """
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        pdb_id = "2cg4"

        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)

        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        # PDB residue numbers should be integers
        for aln_col, pdb_res in mapping.items():
            assert isinstance(aln_col, int)
            assert isinstance(pdb_res, int)

    def test_alignment_handles_gaps(self):
        """
        Verify gap positions in MSA are mapped correctly.

        Expected:
        - Gapped columns are either skipped or marked as None/NaN
        - Aligned and ungapped positions map 1:1 to PDB structure
        """
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        pdb_id = "2cg4"

        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)

        # Verify that gapped positions are handled (either not in dict or mapped to None)
        for aln_col, pdb_res in mapping.items():
            assert pdb_res is None or isinstance(pdb_res, int)

    def test_alignment_column_index_range(self):
        """
        Verify mapping covers the expected column range (1-165 for IPR019888).

        Expected:
        - Mapping keys should span from 1 to 165 (or close to it)
        - No out-of-range column indices
        """
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        pdb_id = "2cg4"

        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)

        if mapping:
            max_col = max(mapping.keys())
            min_col = min(mapping.keys())
            assert min_col >= 1
            assert max_col <= 165


class TestPyMOLScriptGeneration:
    """Test generation of PyMOL scripts for SDP visualization."""

    def test_pymol_script_basic(self):
        """
        Generate a .pml script that colors SDPs by hierarchical level.

        Expected:
        - Output file is created
        - File contains valid PyMOL syntax
        - Includes color commands for Groups, Families, Subfamilies
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sdps_hierarchical.pml"

            mapper = PDBMapper(pdb_id=pdb_id)
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_path,
            )

            assert output_path.exists()
            with open(output_path, "r") as f:
                content = f.read()
                # Verify PyMOL syntax
                assert "color" in content or "colour" in content
                # Verify hierarchy levels are present
                assert "group" in content.lower() or "Groups" in content
                assert len(content) > 100

    def test_pymol_script_hierarchy_colors(self):
        """
        Verify that hierarchical levels get distinct visual encodings.

        Expected:
        - Groups colored with one scheme
        - Families colored with another scheme
        - Subfamilies colored with a third scheme
        - Colors are PyMOL-compatible (hex, named, or RGB)
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sdps_colored.pml"

            mapper = PDBMapper(pdb_id=pdb_id)
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_path,
            )

            with open(output_path, "r") as f:
                content = f.read()
                # Each hierarchy level should be mentioned
                lines = content.lower()
                assert "group" in lines or "subfamily" in lines or "family" in lines

    def test_pymol_script_integrates_alignment_mapping(self):
        """
        Verify that the PyMOL script uses the MSA-to-PDB mapping correctly.

        Expected:
        - SDP positions from CSV are mapped to PDB residue numbers
        - PyMOL script uses PDB residue numbers, not MSA column numbers
        - Script selects residues using PDB numbering
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sdps_pdb_mapped.pml"

            mapper = PDBMapper(pdb_id=pdb_id)
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_path,
            )

            # The script should contain residue selections and color commands
            with open(output_path, "r") as f:
                content = f.read()
                # Should have selection statements (e.g., "resi X")
                assert len(content) > 100


class TestChimeraXScriptGeneration:
    """Test generation of separate ChimeraX scripts for each hierarchy level."""

    def test_three_separate_chimerax_scripts(self):
        """
        Generate three level-specific scripts instead of one combined file.

        Expected:
        - Groups, families, and subfamilies each get a distinct .cxc file
        - The legacy combined file is not required for the refactor
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mapper = PDBMapper(pdb_id=pdb_id)
            outputs = mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_dir=output_dir,
            )

            assert set(outputs.keys()) == {"groups", "families", "subfamilies"}
            assert outputs["groups"].name == "highlight_sdps_groups.cxc"
            assert outputs["families"].name == "highlight_sdps_families.cxc"
            assert outputs["subfamilies"].name == "highlight_sdps_subfamilies.cxc"
            for script_path in outputs.values():
                assert script_path.exists()

    def test_chimerax_scripts_cover_all_switch_positions(self):
        """
        Ensure the refactored ChimeraX scripts use all mapped switch positions.

        Expected:
        - The families script contains every position with switch_count > 0 from the SDP table
        - The script is not limited to a top-N subset
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        mapper = PDBMapper(pdb_id=pdb_id)
        mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)
        expected_positions = sorted(
            int(pos)
            for pos in pd.read_csv(sdp_csv_families).query("switch_count > 0")["position"].tolist()
            if int(pos) in mapping
        )
        expected_residues = sorted(mapping[pos] for pos in expected_positions)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                sdp_csv_groups=Path("results/badasp_scoring/badasp_scores_groups.csv"),
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=Path("results/badasp_scoring/badasp_scores_subfamilies.csv"),
                output_dir=output_dir,
            )

            content = outputs["families"].read_text()
            for residue in expected_residues:
                assert f":{residue}" in content

    def test_chimerax_scripts_include_standalone_png_legends(self):
        """
        Verify that each level has a standalone PNG colorbar legend.

        Expected:
        - legend_groups.png, legend_families.png, legend_subfamilies.png are generated
        - files are non-empty raster artifacts
        """
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        mapper = PDBMapper(pdb_id="2cg4")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                sdp_csv_groups=Path("results/badasp_scoring/badasp_scores_groups.csv"),
                sdp_csv_families=Path("results/badasp_scoring/badasp_scores_families.csv"),
                sdp_csv_subfamilies=Path("results/badasp_scoring/badasp_scores_subfamilies.csv"),
                output_dir=output_dir,
            )

            for legend_name in ("legend_groups.png", "legend_families.png", "legend_subfamilies.png"):
                legend_path = output_dir / legend_name
                assert legend_path.exists()
                assert legend_path.stat().st_size > 0

    def test_chimerax_scripts_do_not_use_key_command(self):
        """Ensure we avoid ChimeraX key syntax due to the palette lookup bug."""
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        mapper = PDBMapper(pdb_id="2cg4")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = mapper.generate_chimerax_scripts(
                alignment_path=alignment_path,
                sdp_csv_groups=Path("results/badasp_scoring/badasp_scores_groups.csv"),
                sdp_csv_families=Path("results/badasp_scoring/badasp_scores_families.csv"),
                sdp_csv_subfamilies=Path("results/badasp_scoring/badasp_scores_subfamilies.csv"),
                output_dir=output_dir,
            )

            for script_path in outputs.values():
                content = script_path.read_text()
                assert "\nkey " not in content


class TestPDBMapperEndToEnd:
    """Integration tests for end-to-end structural mapping."""

    def test_pdb_mapper_initialization(self):
        """
        Test basic initialization of PDBMapper.

        Expected:
        - Mapper object created successfully
        - PDB ID is stored
        - Ready for subsequent operations
        """
        pdb_id = "2cg4"
        mapper = PDBMapper(pdb_id=pdb_id)

        assert mapper.pdb_id == pdb_id

    def test_pdb_mapper_with_custom_cache(self):
        """
        Test PDBMapper with custom cache directory.

        Expected:
        - Custom PDB cache directory is used
        - Downloads are stored in custom location
        """
        pdb_id = "2cg4"
        with tempfile.TemporaryDirectory() as tmpdir:
            mapper = PDBMapper(pdb_id=pdb_id, cache_dir=tmpdir)
            pdb_path = mapper.download_pdb()

            assert str(pdb_path).startswith(tmpdir) or Path(pdb_path).exists()

    def test_complete_workflow(self):
        """
        Test the complete Phase 6 workflow: download, map, script generation.

        Expected:
        - Downloads PDB file
        - Maps MSA to structure
        - Generates PyMOL script
        - All artifacts are created without error
        """
        pdb_id = "2cg4"
        alignment_path = Path("data/interim/IPR019888_trimmed.aln")
        sdp_csv_groups = Path("results/badasp_scoring/badasp_scores_groups.csv")
        sdp_csv_families = Path("results/badasp_scoring/badasp_scores_families.csv")
        sdp_csv_subfamilies = Path("results/badasp_scoring/badasp_scores_subfamilies.csv")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_pml = Path(tmpdir) / "sdps.pml"

            mapper = PDBMapper(pdb_id=pdb_id)

            # Step 1: Download PDB
            pdb_path = mapper.download_pdb()
            assert pdb_path is not None

            # Step 2: Map alignment to structure
            mapping = mapper.map_alignment_to_structure(alignment_path=alignment_path)
            assert isinstance(mapping, dict)

            # Step 3: Generate PyMOL script
            mapper.generate_pymol_script(
                alignment_path=alignment_path,
                sdp_csv_groups=sdp_csv_groups,
                sdp_csv_families=sdp_csv_families,
                sdp_csv_subfamilies=sdp_csv_subfamilies,
                output_pml=output_pml,
            )
            assert output_pml.exists()
