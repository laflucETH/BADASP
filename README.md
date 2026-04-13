# BADASP Pipeline (IPR019888)

## Purpose
This repository implements a reproducible BADASP-inspired computational pipeline focused on the IPR019888 transcription factor family. The workflow performs sequence ingestion, quality-controlled alignment/phylogeny generation, and topological subfamily clustering to support downstream specificity-determining position analysis.

## Current Status
- **Phase 1 (Architecture & Data Ingestion)**: ✓ Complete
- **Phase 2 (Alignment & Phylogeny)**: ✓ Complete — CD-HIT, MAFFT, trimAl, FastTree
- **Phase 3 (Topological Subfamily Clustering)**: ✓ Complete — Hierarchical tree clustering, LCA identification
- **Phase 4 (Ancestral Sequence Reconstruction)**: ✓ Complete — IQ-TREE2 ASR for internal nodes
- **Phase 5 (Restricted BADASP Scoring)**: ✓ Complete — Multilevel sister-clade scoring, SDP identification (45/45 tests passing)
- **Phase 6 (Structural Mapping)**: ✓ Complete — PyMOL/ChimeraX script generation, sequence-to-structure alignment mapping
- **Phase 7 (Evolutionary & Physicochemical Analysis)**: ✓ Complete — Evolutionary timeline, structural clustering, co-evolution networks, multi-level synthesis
- **Phase 7b (Advanced Synthesis)**: ✓ Complete — Architectural domain mapping, community extraction, taxonomic distribution
- **Dendrogram Visualizations**: ✓ Complete & Refined — Orientation standardization, style cleanup (endpoint removal), architecture normalization
- All development uses TDD and the root virtual environment (`venv/`). Full test suite: **60/60 passing**.

## Methodology Summary

### Phase 1-2: Sequence Ingestion & Alignment
1. Fetch IPR019888 sequences from UniProt/InterPro (117k raw sequences).
2. Filter sequences by domain length (130-200 AA; 110k retained).
3. Perform CD-HIT representative clustering (5.9k clusters at 60% identity).
4. Build MSA with MAFFT; trim columns with trimAl (`-gt 0.2`; 165 columns).
5. Build ML phylogeny with FastTree.

### Phase 3: Topological Subfamily Clustering
6. Root tree at midpoint; convert to hierarchical linkage representation.
7. Cut hierarchy into monophyletic clades by cophenetic-distance threshold (adaptive search).
8. Retain clades ≥5 sequences (3-level hierarchy: Groups, Families, Subfamilies).
9. Identify and extract clade Last Common Ancestors (LCAs).

### Phase 4-5: Ancestral Reconstruction & Scoring
10. Run IQ-TREE2 ASR to infer ancestral amino acid sequences at internal nodes.
11. Compute restricted BADASP scores for nearest-sister clade pairs within hierarchy.
12. Score formula: `RC - (AC * p(AC))` where RC=conservation, AC=ancestral call, p(AC)=posterior probability.
13. Calculate 95th percentile threshold on raw pairwise scores; identify Specificity Determining Positions (SDPs).
14. Generate dendrogram switch-event overlays and hierarchical score distributions.

### Phase 6-7: Structural & Evolutionary Analysis (Complete)
15. Map trimmed alignment columns to PDB residue numbers; generate PyMOL/ChimeraX scripts for SDP visualization.
16. Analyze SDP evolution: phylogenetic depth timeline, 3D spatial clustering, co-evolution networks, physicochemical trajectories.
17. Perform multilevel (Groups/Families/Subfamilies) architectural domain mapping, community extraction from coevolution matrices, and taxonomic SDP distribution analysis.
18. Generate publication-ready visualizations with normalized (per-residue) architectural switch distributions and hierarchical dendrograms with refined styling.

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
- `results/badasp_scoring/`: hierarchical BADASP scores, switch dendrograms, and score distributions
- `results/structural_mapping/`: ChimeraX/PyMOL visualization scripts, PDB mappings, and legends
- `results/evolutionary_analysis/`: phylogenetic timelines, structural clustering heatmaps, coevolution matrices, physicochemical shifts, architectural domain distributions (raw and normalized), taxonomic SDP mapping, and multilevel synthesized outputs

## Reproducibility Notes
- Use root virtual environment commands, for example: `./venv/bin/python -m pytest -q`.
- Generate vector figures as SVG by default.
- `_archive_v1/` is excluded from active development and execution.
