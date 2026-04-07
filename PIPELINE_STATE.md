# PIPELINE_STATE

## Project
- Name: BADASP replication pipeline for IPR019888 (PF13404 track included)
- Mode: Reproducible, modular Python workflow
- Current date: 2026-04-07

## Architecture Snapshot
- Active root workspace now contains a fresh Phase 1 scaffold.
- Archived legacy work is preserved in `_archive_v1/` and was not modified.
- Project instructions are now codified in `.github/copilot-instructions.md` with explicit phase approval stops, descriptive results directories, vector-output policy, TDD, and root-venv enforcement.

## Directory Layout (Phase 1)
- `src/` : pipeline source modules
- `tests/` : pytest suite
- `data/raw/` : downloaded FASTA and source files (gitignored)
- `data/interim/` : intermediate artifacts (gitignored)
- `data/processed/` : processed tables/artifacts (gitignored)
- `docs/` : project documentation
- `notebooks/` : analysis notebooks
- `results/` : generated outputs (gitignored except `.gitkeep` scaffolds)
  - `results/sequence_filtering/` : sequence-length QC plots
  - `results/alignment_qc/` : alignment quality plots
  - `results/topological_clustering/` : tree-based clade outputs and dendrograms

## Completed Work
- Created Phase 1 scaffold directories and `.gitkeep` placeholders.
- Hardened `.gitignore` to exclude raw/interim/processed data, results, local env/caches, and large artifacts while preserving scaffold placeholders.
- Implemented TDD-first ingestion module:
  - Test file: `tests/test_data_fetcher.py`
  - Source file: `src/data_fetcher.py`
- Executed tests: `3 passed`.
- Executed ingestion command for IPR019888:
  - Command: `python src/data_fetcher.py --interpro-id IPR019888 --output data/raw/IPR019888.fasta`
  - Result: `117246` FASTA records written.
- Implemented TDD-first Phase 2 modules:
  - Tests: `tests/test_sequence_cluster.py`, `tests/test_msa_builder.py`, `tests/test_tree_builder.py`
  - Source: `src/sequence_cluster.py`, `src/msa_builder.py`, `src/tree_builder.py`
- Executed full test suite: `6 passed`.
- Executed Phase 2 pipeline:
  - Clustering: `python src/sequence_cluster.py --input data/raw/IPR019888.fasta --output data/interim/IPR019888_clustered.fasta --identity 0.7 --word-size 4`
  - Alignment + trimming: `python src/msa_builder.py --input data/interim/IPR019888_clustered.fasta --aligned data/interim/IPR019888_aligned.aln --output data/interim/IPR019888_trimmed.aln --gap-threshold 0.5`
  - Phylogeny: `python src/tree_builder.py --input data/interim/IPR019888_trimmed.aln --output data/interim/IPR019888.tree`
- Implemented retroactive QC remediation for Phase 2 failure:
  - Added visual QC module: `src/visualization.py`
  - Added strict length pre-filtering in `src/sequence_cluster.py` (130-200 AA)
  - Switched aggressive clustering default to `cd-hit -c 0.60 -n 4`
  - Added alignment gap-per-column QC plotting
  - Re-ran clustering, MAFFT, trimAl (`-gt 0.5`), and FastTree
  - Added tests for filtering and QC plots; full suite now `10 passed`
  - Corrected rerun performed in root `venv` with SVG QC outputs
- Housekeeping completed before Phase 3:
  - Added and populated `README.md` with purpose, status, methodology, and structure
  - Reorganized `results/` into descriptive analysis subdirectories
  - Routed visualization defaults to descriptive result paths

## Data Metrics (Current)
- IPR019888 raw sequence count: 117246
- Raw FASTA path: `data/raw/IPR019888.fasta`
- Length-filtered sequence count (130-200 AA): 110022
- CD-HIT representative sequence count (`-c 0.60 -n 4`): 5925
- Clustered FASTA path: `data/interim/IPR019888_clustered.fasta`
- Trimmed alignment column length (trimAl -gt 0.5): 60
- Trimmed alignment path: `data/interim/IPR019888_trimmed.aln`
- Phase 2 tree path: `data/interim/IPR019888.tree`

## QC Metrics
- Raw sequence length QC plot: `results/sequence_filtering/raw_length_dist.svg`
- MSA gap-per-column QC plot: `results/alignment_qc/msa_gap_profile.svg`
- Length-filtered sequence count (130-200 AA): 110022
- Clustered representative sequence count (`-c 0.60 -n 4`): 5925
- Trimmed alignment column length after corrected rerun (`trimAl -gt 0.5`): 60

## Phase Status
- Phase 1 (Architecture & Data Ingestion): complete
  - Architecture scaffold: complete
  - Git hygiene: complete
  - Data fetcher module + tests: complete
  - Raw sequence ingestion: complete
- Phase 2 (Alignment & Phylogeny): complete, awaiting user review
  - Sequence clustering with CD-HIT: complete
  - MAFFT alignment + trimAl trimming: complete
  - FastTree ML phylogeny: complete
  - QC remediation pass: complete

## Pending (Before Phase 3)
- User review/approval of housekeeping updates and descriptive results routing.
