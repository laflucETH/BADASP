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
- Critical Phase 2/3 bug-fix pass:
  - Updated trimAl aggressiveness from `-gt 0.5` to `-gt 0.2` in `src/msa_builder.py`
  - Added programmatic trimmed-alignment column counting from FASTA parsing (no hardcoded values)
  - Re-ran MAFFT + trimAl + FastTree and logged verified trimmed alignment length
  - Applied midpoint rooting (`tree.root_at_midpoint()`) before topological clustering in `src/tree_cluster.py`
  - Re-ran clustering and regenerated dendrogram with explicit `color_threshold` equal to the exact `fcluster` cut distance
  - Full test suite after bug-fix pass: `18 passed`
- Housekeeping completed before Phase 3:
  - Added and populated `README.md` with purpose, status, methodology, and structure
  - Reorganized `results/` into descriptive analysis subdirectories
  - Routed visualization defaults to descriptive result paths
- Implemented TDD-first Phase 3 topological clustering:
  - Tests: `tests/test_tree_cluster.py` and visualization extensions
  - Source: `src/tree_cluster.py`
  - Method: topology-derived hierarchical linkage + adaptive `scipy.cluster.hierarchy.fcluster` threshold selection targeting major clades
  - QC filter: dropped clades with fewer than 5 sequences before downstream analysis
  - LCA identification: computed per generated clade using phylogenetic common ancestor
  - Dendrogram output generated as SVG in `results/topological_clustering/` with `color_threshold` matched to the clustering cut threshold
  - Full test suite after changes: `17 passed`
- Implemented TDD-first Phase 4 ancestral sequence reconstruction:
  - Test: `tests/test_asr_runner.py`
  - Source: `src/asr_runner.py`
  - Tooling: IQ-TREE2 `-asr` with extraction of ancestral sequences for LCA nodes of valid clades
  - Output: `data/interim/ancestral_sequences.fasta`

## Data Metrics (Current)
- IPR019888 raw sequence count: 117246
- Raw FASTA path: `data/raw/IPR019888.fasta`
- Length-filtered sequence count (130-200 AA): 110022
- CD-HIT representative sequence count (`-c 0.60 -n 4`): 5925
- Clustered FASTA path: `data/interim/IPR019888_clustered.fasta`
- Trimmed alignment column length (trimAl -gt 0.2): 165
- Trimmed alignment path: `data/interim/IPR019888_trimmed.aln`
- Phase 2 tree path: `data/interim/IPR019888.tree`

## QC Metrics
- Raw sequence length QC plot: `results/sequence_filtering/raw_length_dist.svg`
- MSA gap-per-column QC plot: `results/alignment_qc/msa_gap_profile.svg`
- Length-filtered sequence count (130-200 AA): 110022
- Clustered representative sequence count (`-c 0.60 -n 4`): 5925
- Trimmed alignment column length after corrected rerun (`trimAl -gt 0.2`): 165

## Phase 3 Metrics (Topological Subfamily Clustering)
- Input tree path: `data/interim/IPR019888.tree`
- Midpoint-rooted tree artifact: `results/topological_clustering/midpoint_rooted.tree`
- Topological clades generated (after min-size filter >=5): 34
- Minimum clade size retained: 5
- Tree rooting mode for clustering: midpoint-rooted
- Distance threshold selection: adaptive search targeting 20-80 clades
- Distance threshold used in latest rerun: 6.929765
- Clade summary output: `results/topological_clustering/tree_clusters.csv`
- Clade assignment output: `results/topological_clustering/tree_cluster_assignments.csv`
- Dendrogram output: `results/topological_clustering/tree_dendrogram.svg`

## Phase 4 Metrics (Ancestral Sequence Reconstruction)
- ASR engine: IQ-TREE2 (`-asr`)
- ASR run prefix: `data/interim/asr_run`
- Tree used for refreshed ASR run: `results/topological_clustering/midpoint_rooted.tree`
- Assignments used for refreshed ASR run: `results/topological_clustering/tree_cluster_assignments.csv`
- LCA ancestral sequences written: 34
- LCA ancestral FASTA output: `data/interim/ancestral_sequences.fasta`
- Refresh status: completed after rooted-tree and 34-clade Phase 3 rerun.

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
- Phase 3 (Topological Subfamily Clustering): complete for initial implementation, awaiting user review
  - Topological clustering on FastTree output: complete
  - LCA identification per clade: complete
  - Topological dendrogram visualization: complete

- Phase 4 (Ancestral Sequence Reconstruction): complete for initial implementation, awaiting user review
  - IQ-TREE ASR execution: complete
  - LCA sequence extraction for valid clades: complete

## Pending (Before Phase 5)
- User review/approval of rooted-tree Phase 3 artifacts and refreshed Phase 4 ASR outputs.
