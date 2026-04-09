# BADASPer Core Instructions

## Workflow Governance
- You must pause at the end of each BADASP phase and explicitly request user review and approval before moving to the next phase.
- You may execute terminal commands autonomously within the active phase.
- Do not start a new phase without explicit user go-ahead.

## Test-Driven Development
- For every new core pipeline module, write or update pytest tests before implementing code.
- Run the relevant tests and then the full test suite after implementation changes.

## Environment Rules
- Always run pipeline scripts and tests with the root virtual environment at `venv/`.
- Do not use environments or artifacts under `_archive_v1/`.

## Plot And Figure Rules
- Generate publication-ready vector outputs only.
- Default to SVG for standard QC plots.
- Use PDF only when dense tree/print constraints make PDF more appropriate.

## Results Directory Policy
- Organize `results/` into descriptive analysis-based subdirectories.
- Use names tied to analysis purpose (for example: `sequence_filtering`, `alignment_qc`, `topological_clustering`).
- Do not use phase-number directory naming.

## README Maintenance
- Maintain `README.md` continuously.
- Keep `README.md` current with: project purpose, current status, methodology, and directory structure.

## Methodological Rules
- Topological Hierarchy: The pipeline strictly uses a 3-level hierarchy (Groups, Families, Subfamilies) derived from dynamic tree cutting, not a flat structure.
- BADASP Scoring: Scoring uses the Bradley et al. (2019) subtractive formula, `Score = RC - (AC * p_ac)`.
- Pairwise Logic: Scores are calculated exclusively between nearest sister-clades within the same parent hierarchy, avoiding global squashing.
- SDP Definition: Specificity Determining Positions are defined by counting threshold-exceeding switches, not by raw score averaging.
