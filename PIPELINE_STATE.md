# PIPELINE_STATE

## Project
- Name: BADASP replication pipeline for IPR019888 (PF13404 track included)
- Mode: Reproducible, modular Python workflow
- Current date: 2026-04-10

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
- Per-level rooted tree artifacts:
  - `results/topological_clustering/midpoint_rooted_groups.tree`
  - `results/topological_clustering/midpoint_rooted_families.tree`
  - `results/topological_clustering/midpoint_rooted_subfamilies.tree`
- Topological clades generated (after min-size filter >=5): 34
- Minimum clade size retained: 5
- Tree rooting mode for clustering: midpoint-rooted
- Distance threshold selection: adaptive search targeting 20-80 clades
- Distance threshold used in latest rerun: 6.929765
- Clade summary output: `results/topological_clustering/tree_clusters.csv`
- Clade assignment output: `results/topological_clustering/tree_cluster_assignments.csv`
- Per-level clade summary outputs:
  - `results/topological_clustering/tree_clusters_groups.csv`
  - `results/topological_clustering/tree_clusters_families.csv`
  - `results/topological_clustering/tree_clusters_subfamilies.csv`
- Per-level assignment outputs:
  - `results/topological_clustering/tree_cluster_assignments_groups.csv`
  - `results/topological_clustering/tree_cluster_assignments_families.csv`
  - `results/topological_clustering/tree_cluster_assignments_subfamilies.csv`
- Dendrogram output: `results/topological_clustering/tree_dendrogram.svg`
- Per-level dendrograms:
  - `results/topological_clustering/tree_dendrogram_groups.svg`
  - `results/topological_clustering/tree_dendrogram_families.svg`
  - `results/topological_clustering/tree_dendrogram_subfamilies.svg`

## Phase 4 Metrics (Ancestral Sequence Reconstruction)
- ASR engine: IQ-TREE2 (`-asr`)
- ASR run prefix: `data/interim/asr_run`
- Tree used for refreshed ASR run: `results/topological_clustering/midpoint_rooted.tree`
- Assignments used for refreshed ASR run: `results/topological_clustering/tree_cluster_assignments.csv`
- LCA ancestral sequences written: 34
- LCA ancestral FASTA output: `data/interim/ancestral_sequences.fasta`
- Refresh status: completed after rooted-tree and 34-clade Phase 3 rerun.
- Hierarchical extraction support: `src/asr_runner.py` now reads all unique `*_lca_node` values across group, family, and subfamily assignments for multilevel reruns.
- Hierarchical rerun written: 142 ancestral sequences from the current multilevel assignment table.

## Phase 5 Metrics (Restricted BADASP Scoring)
- Implementation: TDD-first with `src/badasp_core.py` and `tests/test_badasp_core.py` for the multilevel hierarchy contract
- Method: three-level BADASP scoring for group, family, and subfamily comparisons
  - Sister-pairing: strict nearest phylogenetic sister-clade events only (topology-driven), restricted by parent hierarchy constraints
  - RC (Recent Conservation): average BLOSUM62-based similarity across the compared clades
  - AC (Ancestral Conservation): binary identical/different call between ancestral residues (1 or -1)
  - p(AC): posterior probability from the IQ-TREE state file
  - Score: `RC - (AC * p(AC))`
  - Thresholding: pooled 95th percentile threshold per hierarchy level
  - SDP selection: highest switch-count positions per level
- Outputs written by the refactor:
  - `results/badasp_scoring/badasp_scores_groups.csv`
  - `results/badasp_scoring/badasp_scores_families.csv`
  - `results/badasp_scoring/badasp_scores_subfamilies.csv`
- Raw pairwise score exports:
  - `results/badasp_scoring/raw_pairwise_groups.csv`
  - `results/badasp_scoring/raw_pairwise_families.csv`
  - `results/badasp_scoring/raw_pairwise_subfamilies.csv`
- Corrected rerun thresholds and max switches:
  - Groups: threshold 1.312925, max switch_count 2 at position 123 (max_score = 1.454876)
  - Families: threshold 1.272619, max switch_count 9 at position 46 (max_score = 1.581377)
  - Subfamilies: threshold 1.290963, max switch_count 18 at position 44 (max_score = 1.563795)
- Hierarchical comparative visualizations:
  - Score distributions: `results/badasp_scoring/hierarchical_distributions.svg`
  - Switch counts across the alignment: `results/badasp_scoring/hierarchical_switch_counts.svg`
- Individual hierarchical score distributions (raw counts on the Y-axis):
  - `results/badasp_scoring/badasp_score_distribution_groups.svg`
  - `results/badasp_scoring/badasp_score_distribution_families.svg`
  - `results/badasp_scoring/badasp_score_distribution_subfamilies.svg`
- Unified dendrogram switch-event visualizations (Supplementary S2-style):
  - `results/badasp_scoring/dendrogram_switches_groups.svg`
  - `results/badasp_scoring/dendrogram_switches_families.svg`
  - `results/badasp_scoring/dendrogram_switches_subfamilies.svg`
  - Mapping validation (debug rerun):
    - Groups: 4 pairs with switches, 4 mapped scatter coordinates
    - Families: 23 pairs with switches, 21 mapped scatter coordinates
    - Subfamilies: 78 pairs with switches, 72 mapped scatter coordinates
- Distribution source for all score-distribution plots: raw pairwise score tables (`raw_pairwise_*.csv`), with histograms rendered as absolute counts and 95th percentile lines derived from the same raw distributions
- Test coverage: 45 tests passing in the current suite

## Phase 6 Metrics (Structural Mapping)
- Implementation: TDD-first with `src/pdb_mapper.py` and `tests/test_pdb_mapper.py`
- PDB download support: `Bio.PDB.PDBList` with local caching in `data/raw/`
- Sequence-to-structure mapping:
  - Representative sequence selected from `data/interim/IPR019888_trimmed.aln`
  - Pairwise alignment engine: `Bio.Align.PairwiseAligner` (global mode)
  - Mapping output contract: `msa_column_index -> pdb_residue_number`
- Structural script outputs:
  - ChimeraX scripts:
    - `results/structural_mapping/highlight_sdps_groups.cxc`
    - `results/structural_mapping/highlight_sdps_families.cxc`
    - `results/structural_mapping/highlight_sdps_subfamilies.cxc`
  - PyMOL script support implemented in module API (`generate_pymol_script`)
  - Source tables: `results/badasp_scoring/badasp_scores_{groups,families,subfamilies}.csv` filtered to `switch_count > 0`
- Publication-quality ChimeraX rendering profile (Figure 2-style upgrade):
  - Separate per-level scripts instead of one combined file
  - Every mapped switch position (`switch_count > 0`) is colored with a yellow -> red gradient
  - Rendering directives embedded in script header:
    - `set bgColor white`
    - `lighting soft`
    - `graphics silhouettes true color black width 1.5`
    - `material dull`
    - `show cartoon`
    - `hide atoms`
    - `show :<sdp_residues> atoms` and stick styling for SDP sidechains
- CLI execution status:
  - Command run: `python -m src.pdb_mapper --pdb-id 2cg4 --alignment data/interim/IPR019888_trimmed.aln --scores-dir results/badasp_scoring --output-cxc results/structural_mapping/highlight_sdps.cxc`
  - Confirmation: `Generated ChimeraX scripts: results/structural_mapping/highlight_sdps_groups.cxc, results/structural_mapping/highlight_sdps_families.cxc, results/structural_mapping/highlight_sdps_subfamilies.cxc`

## Phase Status
- Phase 1 (Architecture & Data Ingestion): complete
  - Architecture scaffold: complete
  - Git hygiene: complete
  - Data fetcher module + tests: complete
  - Raw sequence ingestion: complete
- Phase 2 (Alignment & Phylogeny): complete, user-approved
  - Sequence clustering with CD-HIT: complete
  - MAFFT alignment + trimAl trimming: complete
  - FastTree ML phylogeny: complete
  - QC remediation pass: complete
- Phase 3 (Topological Subfamily Clustering): complete, user-approved
  - Topological clustering on FastTree output: complete
  - LCA identification per clade: complete
  - Topological dendrogram visualization: complete
  - Rooted tree artifact saved: complete
- Phase 4 (Ancestral Sequence Reconstruction): complete, user-approved
  - IQ-TREE ASR execution: complete
  - LCA sequence extraction for valid clades: complete
  - Refresh with rooted tree: complete
- Phase 5 (Restricted BADASP Scoring): complete, user-approved
  - TDD-first core implementation: complete
  - Multilevel hierarchy refactor (Groups/Families/Subfamilies): complete
  - Hierarchical ASR node extraction support: complete
  - Strict nearest-sister switch-count aggregation: complete
  - BADASP score calculation: complete
  - Hierarchical comparative visualizations: complete
  - Standalone per-level score distribution plots: complete
  - Unified dendrogram switch visualizations: complete
  - Min_clade_size filter audit logging: complete
  - Visual gray-out of ignored clades: complete
  - All tests passing: complete (33/33)
- **Phase 6 (Structural Mapping): complete, user review pending**
  - TDD scaffold setup: complete
  - PDB downloader function: complete
  - Sequence-to-structure alignment: complete
  - PyMOL/ChimeraX script generation: complete
  - Structural annotation module implementation: complete
  - Validation: complete (`tests/test_pdb_mapper.py` 12/12, full suite 45/45)
- **Phase 7 (Evolutionary & Physicochemical Analysis): in progress**
  - Evolutionary timeline (Age of Switches): pending
  - Structural clustering analysis: pending
  - Co-evolution network analysis: pending
  - Physicochemical trajectory analysis: pending

## Pending (Before Phase 8)
- User review/approval of Phase 6 structural mapping outputs
- Phase 7 implementation and validation
