# BADASP Pipeline (IPR019888)

## Purpose
This repository implements a reproducible BADASP-inspired computational pipeline focused on the IPR019888 transcription factor family. The workflow performs sequence ingestion, quality-controlled alignment/phylogeny generation, and topological subfamily clustering to support downstream specificity-determining position analysis.

## Current Status
- Phase 1 complete: architecture and sequence ingestion are implemented and tested.
- Phase 2 complete: length filtering, CD-HIT reduction, MAFFT alignment, trimAl trimming, and FastTree phylogeny are implemented and tested.
- Phase 3 in progress: topological, tree-based subfamily clustering and LCA identification implemented with tests and execution outputs.
- All development uses TDD and the root virtual environment (`venv/`).

## Methodology Summary
1. Fetch IPR019888 sequences from UniProt/InterPro stream.
2. Filter sequences by expected domain length range (130-200 AA).
3. Perform CD-HIT representative clustering for tractable MSA input.
4. Build MSA with MAFFT and trim columns with trimAl (`-gt 0.5`).
5. Build an ML-like phylogeny with FastTree.
6. Convert tree topology to a hierarchical linkage representation.
7. Cut the hierarchy into monophyletic clades by cophenetic-distance threshold.
8. Identify clade LCAs and export cluster summary tables.

## Repository Structure
- `src/`: pipeline modules
  - `data_fetcher.py`: InterPro/UniProt sequence ingestion
  - `sequence_cluster.py`: length filtering + CD-HIT clustering
  - `msa_builder.py`: MAFFT alignment + trimAl trimming
  - `tree_builder.py`: FastTree tree construction
  - `tree_cluster.py`: topological clade clustering + LCA reporting
  - `visualization.py`: QC and clustering visual outputs
- `tests/`: pytest suite for all core modules
- `data/raw/`: source sequence inputs (gitignored)
- `data/interim/`: intermediate artifacts (gitignored)
- `data/processed/`: processed artifacts (gitignored)
- `results/`: vector graphics and tabular outputs organized by analysis
  - `results/sequence_filtering/`
  - `results/alignment_qc/`
  - `results/topological_clustering/`

## Results Organization Policy
Results are grouped by analysis purpose and never by phase number:
- `results/sequence_filtering/`: sequence-length QC outputs
- `results/alignment_qc/`: MSA quality outputs
- `results/topological_clustering/`: tree-clade assignments, LCA summaries, and dendrograms

## Reproducibility Notes
- Use root virtual environment commands, for example: `./venv/bin/python -m pytest -q`.
- Generate vector figures as SVG by default.
- `_archive_v1/` is excluded from active development and execution.
