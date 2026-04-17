# BADASP Pipeline (IPR019888)

## Purpose
This repository implements a reproducible BADASP-inspired computational pipeline focused on the IPR019888 transcription factor family. The workflow performs sequence ingestion, quality-controlled alignment/phylogeny generation, and topological subfamily clustering to support downstream specificity-determining position analysis.

## Current Status
- **Phase 1 (Architecture & Data Ingestion)**: ✓ Complete
- **Phase 2 (Alignment & Phylogeny)**: ✓ Complete — CD-HIT (default 0.80), FAMSA/MAFFT, trimAl, native OpenMP FastTreeMP
- **Phase 3 (Topological Subfamily Clustering)**: ✓ Complete — Hierarchical tree clustering, LCA identification
- **Phase 4 (Ancestral Sequence Reconstruction)**: ✓ Complete — single-pass global IQ-TREE2 ASR with hierarchical LCA extraction
- **Phase 5 (Restricted BADASP Scoring)**: ✓ Complete — Multilevel sister-clade scoring, SDP identification (45/45 tests passing)
- **Phase 6 (Structural Mapping)**: ✓ Complete — PyMOL/ChimeraX script generation, sequence-to-structure alignment mapping
- **Phase 7 (Evolutionary & Physicochemical Analysis)**: ✓ Complete — Evolutionary timeline, structural clustering, co-evolution networks, multi-level synthesis
- **Phase 7b (Advanced Synthesis)**: ✓ Complete — Architectural domain mapping, community extraction, taxonomic distribution
- **Dendrogram Visualizations**: ✓ Complete & Refined — Orientation standardization, style cleanup (endpoint removal), architecture normalization
- All development uses TDD and the root virtual environment (`venv/`). Full test suite: **89/89 passing**.

## Methodology Summary

### Phase 1-2: Sequence Ingestion & Alignment
1. Fetch IPR019888 sequences from UniProt/InterPro (117k raw sequences).
2. Filter sequences by domain length (130-200 AA; 110k retained).
3. Perform CD-HIT representative clustering (5.9k clusters at 80% identity default for the tuned pipeline; benchmarked across 0.65-0.80).
4. Build MSA with FAMSA by default (MAFFT remains available); trim columns with trimAl (`-gt 0.2`; 165 columns).
5. Build ML phylogeny with native OpenMP FastTreeMP on Apple Silicon, with automatic fallback to FastTree if needed.

### Phase 3: Topological Subfamily Clustering
6. Root tree with the canonical MAD Python implementation (`venv/bin/mad.py`); if unavailable, fall back to midpoint rooting.
7. Cut hierarchy into monophyletic clades by cophenetic-distance threshold (adaptive search).
8. Retain clades ≥5 sequences (3-level hierarchy: Groups, Families, Subfamilies).
  - Filtering is applied independently per hierarchy level (group/family/subfamily), not as a cross-level intersection.
9. Identify and extract clade Last Common Ancestors (LCAs).

### Phase 4-5: Ancestral Reconstruction & Scoring
10. Run IQ-TREE2 ASR once on the full alignment/tree (`-asr -T AUTO`) to infer ancestral amino acid sequences, then wait for the single global `.state` file and parse it into per-node sequences.
11. Map hierarchical LCA nodes from the clustering assignments onto the ASR tree and extract the corresponding ancestral sequences from the master reconstruction.
12. Compute restricted BADASP scores for nearest-sister clade pairs within hierarchy.
13. Score formula: `RC - (AC * p(AC))` where RC=conservation, AC=ancestral call, p(AC)=posterior probability.
14. Calculate 95th percentile threshold on raw pairwise scores; identify Specificity Determining Positions (SDPs).
15. Generate dendrogram switch-event overlays and hierarchical score distributions.
  - Dendrogram circle sizes reflect aggregated threshold-exceeding pairwise switch events at each LCA node; the per-position `switch_count` tables remain the source for summary statistics.

### Phase 6-7: Structural & Evolutionary Analysis (Complete)
16. Map trimmed alignment columns to PDB residue numbers; generate PyMOL/ChimeraX scripts for SDP visualization.
17. Analyze SDP evolution: phylogenetic depth timeline, 3D spatial clustering, co-evolution networks, physicochemical trajectories.
18. Perform multilevel (Groups/Families/Subfamilies) architectural domain mapping, community extraction from coevolution matrices, and taxonomic SDP distribution analysis.
19. Generate publication-ready visualizations with architectural switch distributions, compact count-based boxplots, and hierarchical dendrograms with refined styling.

## Repository Structure
- `src/`: pipeline modules
  - `data_fetcher.py`: InterPro/UniProt sequence ingestion
  - `sequence_cluster.py`: length filtering + CD-HIT clustering
  - `msa_builder.py`: MAFFT alignment + trimAl trimming
  - `tree_builder.py`: FastTree tree construction
  - `tree_cluster.py`: topological clade clustering + LCA reporting
  - `badasp_core.py`: multilevel BADASP scoring + SDP identification
  - `asr_runner.py`: IQ-TREE2 ancestral sequence reconstruction
  - `pdb_mapper.py`: sequence-to-structure alignment + PyMOL/ChimeraX script generation
  - `evolutionary_analysis.py`: evolutionary timeline, structural clustering, coevolution, physicochemical analysis, multilevel synthesis
  - `visualization.py`: QC and clustering visual outputs including dendrogram rendering
- `tests/`: pytest suite for all core modules
- `data/raw/`: source sequence inputs (gitignored)
- `data/interim/`: intermediate artifacts (gitignored)
- `data/processed/`: processed artifacts (gitignored)
- `results/`: vector graphics and tabular outputs organized by analysis
  - `results/sequence_filtering/`
  - `results/alignment_qc/`
  - `results/topological_clustering/`
  - `results/badasp_scoring/`
  - `results/structural_mapping/`
  - `results/evolutionary_analysis/`

## Results Organization Policy
Results are grouped by analysis purpose and never by phase number:
- `results/sequence_filtering/`: sequence-length QC outputs
- `results/alignment_qc/`: MSA quality outputs
- `results/topological_clustering/`: tree-clade assignments, LCA summaries, and dendrograms (rotated, color-refined, architecture-normalized)
- `results/badasp_scoring/`: hierarchical BADASP scores, switch dendrograms, score distributions, and switch statistics analysis
  - `switch_statistics_summary.csv`: aggregate metrics across hierarchy levels
  - `clade_switch_involvement.csv`: per-clade switch involvement data
  - `plot_*.svg`: publication-ready visualizations (7 plots showing position counts, distributions, top positions, BADASP correlations, frequency data)
- `results/structural_mapping/`: ChimeraX/PyMOL visualization scripts, PDB mappings, and legends
- `results/evolutionary_analysis/`: phylogenetic timelines, structural clustering heatmaps, coevolution matrices, physicochemical shifts, architectural domain distributions, compact count-based boxplots, taxonomic SDP mapping, and multilevel synthesized outputs

Generated CSV outputs under `results/` are treated as local analysis artifacts and are not tracked in git; tree files and SVG figures remain available for committed outputs when needed.

## Reproducibility Notes
- Use root virtual environment commands, for example: `./venv/bin/python -m pytest -q`.
- Generate vector figures as SVG by default.
- `_archive_v1/` is excluded from active development and execution.
- The pipeline intentionally uses a single MAD execution path through `venv/bin/mad.py` (no separate binary-mode integration in pipeline code).
- The tree-building stage now prefers a natively compiled `venv/bin/FastTreeMP` built from source with OpenMP for multicore Apple Silicon execution; single-threaded FastTree is only a fallback.
- IQ-TREE2 benchmark outputs are written to `results/iqtree_scaling.csv` and `results/iqtree_scaling_plot.svg`; the benchmark samples 500/1000/2000/4000-sequence subsets from the full alignment/tree.
- IQ-TREE2 extrapolation plotting now marks the 24,608-sequence 0.80 threshold and saves the result to `results/iqtree_scaling_plot_extrapolated.svg`.
