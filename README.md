# BADASP Analysis for IPR019888

This repository implements the BADASP (Burst After Duplication with Ancestral Sequence Predictions) methodology to identify functional specificity determining sites in the AsnC-like transcription regulator family (IPR019888).

It follows standard bioinformatics data science practices and the approaches described in:
- *Evolution of protein kinase substrate recognition at the active site* (Beltrao and Bradley, 2019)
- *BADASP: predicting functional specificity in protein families using ancestral sequences* (Edwards and Shields, 2005)

## Directory Structure

- `data/raw/`: Raw protein sequences fetched from UniProt API.
- `data/interim/`: Generated sequence alignments and reconstructed phylogenetic trees via IQ-TREE ASR.
- `data/processed/`: Core BADASP algorithm scoring outputs.
- `notebooks/`: Jupyter notebooks encapsulating Exploratory Data Analysis and BADASP insights.
- `src/`: Core Python pipeline scripts.
- `results/`: Plotted distributions and specific high-scoring functionality nodes.

## Setup Instructions

Ensure that you have `Python 3.9+`, `mafft`, and `iqtree2` installed on your system.

1. Clone the repository and navigate into it.
2. Initialize virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Execution

You can step through the scripts in the following order:
1. `python src/data_fetcher.py`
2. `python src/eda.py`
3. `python src/msa_builder.py`
4. `python src/tree_builder.py`
5. `python src/asr_runner.py`
6. `python src/badasp_core.py`
7. `python src/visualization.py`

Alternatively, browse the Jupyter Notebooks inside `notebooks/` to interpret the gathered outputs.
