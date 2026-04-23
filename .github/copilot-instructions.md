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
- Reconciliation Logic: Any reconciliation run MUST use CD-HIT `.clstr` cluster expansion plus fuzzy overlap classification (5% overlap tolerance or <=2 overlapping species treated as Speciation) with metagenome/environmental/uncultured taxa filtered out; do not rely on strict binary ete3 defaults alone.

## Optional Pipeline Modules (Refactor Track)
- Alignment engine may be selected via CLI/module flag:
  - `mafft` (default)
  - `famsa` (FAMSA2-compatible execution path)
- Tree rooting mode may be selected via CLI/module flag:
  - `midpoint` (default)
  - `mad` (Minimal Ancestral Deviation rooting wrapper)
- Evolutionary reconciliation is an optional exploratory module:
  - Gene/species tree reconciliation with duplication/speciation labeling
  - Outputs should be written under `results/reconciliation/`

## The BADASP Scientific Pipeline Roadmap

### Phase 1: Architecture & Data Ingestion
- Scaffold the project directory (`src/`, `tests/`, `data/`, `docs/`, `results/`).
- Enforce `.gitignore` rules.
- Fetch and filter target sequences (e.g., via InterPro/UniProt APIs).

### Phase 2: Alignment & Phylogeny
- Align sequences using robust tools (e.g., MAFFT or via HMM alignment).
- Trim spurious gap columns using `trimAl`.
- Construct a Maximum-Likelihood phylogeny (e.g., using FastTree or RAxML).

### Phase 3: Topological Subfamily Clustering (The Deep Node Filter)
- **CRITICAL:** Do not use sequence identity clustering (like CD-HIT) for subfamily definition.
- Apply a topological, tree-based clustering approach (e.g., `scipy.cluster.hierarchy` on cophenetic distances, or `TreeCluster`) to cut the tree into monophyletic clades/subfamilies based strictly on evolutionary branch lengths.
- Identify the Last Common Ancestor (LCA) internal node for each defined computational clade.

### Phase 4: Ancestral Sequence Reconstruction (ASR)
- Run ASR (e.g., FastML or CodeML/PAML) to infer the amino acid sequences of the internal nodes on the tree.

### Phase 5: Restricted BADASP Scoring
- Implement the adapted BADASP formula: `Score = RC - (AC * p(AC))`.
  - `RC` (Recent Conservation): Conservation within the descendent computational clades.
  - `AC` (Ancestral Conservation): Substitution matrix score between the LCA of Clade A and the LCA of Clade B.
  - `p(AC)`: Posterior probability of the ancestral prediction.
- **CRITICAL:** Restrict this calculation to evaluate ONLY the divergent switches between the deep LCA nodes identified in Phase 3. Ignore shallow intra-clade mutational drift.
- Calculate the 95th percentile threshold strictly on these deep-node switches to isolate the true "bursts" of functional divergence (Specificity Determining Positions).

### Phase 6: Structural Mapping
- Fetch a representative PDB structure for the family (using `Bio.PDB.PDBList`).
- Map the 1D trimmed alignment columns (1-165) to the true 3D PDB residue numbering using profile-to-sequence alignment.
- Generate a PyMOL script (.pml) that colors SDPs on the 3D structure by their hierarchical level (Group, Family, Subfamily).
- Output: `.pml` script, mapped residue lookup table, and structure-aligned visualizations.

### Phase 7: Evolutionary & Physicochemical Analysis
- **Evolutionary Timeline (Age of Switches)**: Calculate the phylogenetic depth (distance from the root) for the LCA node of every switch event. Generate a timeline plot showing the accumulation/frequency of switches over evolutionary time (ancient vs. recent).
- **Structural Clustering**: Calculate 3D distances between top SDPs to prove if they form a spatial cluster (e.g., an active site pocket) compared to random residues.
- **Co-evolution Networks**: Calculate the correlation between switch positions across the phylogeny and plot a co-evolution network graph.
- **Physicochemical Trajectories**: Analyze the specific amino acid changes at the top SDPs to quantify massive biochemical shifts (Charge, Volume, Hydrophobicity transitions).

### Phase 7b: Advanced Structural & Taxonomic Synthesis
- Architectural Domain Mapping must be executed across all three hierarchy levels (Groups, Families, Subfamilies).
- Co-evolution Network Community Extraction must be performed from coevolution matrices and exported as residue-to-community assignments.
- Taxonomic/Species Distribution analysis must map top SDPs to major taxa and export clade-level distributions.

### Phase 9: Evolutionary Reconciliation (Exploratory)
- Add optional gene/species reconciliation workflow (separate from core grouping logic until validated).
- Use species-aware mapping from sequence headers (TaxID/species labels), generate species tree, and reconcile against the BADASP gene tree.
- Export duplication/speciation calls to `results/reconciliation/` plus an annotated vector tree visualization.
