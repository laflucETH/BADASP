import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    text = """\
# Exploratory Data Analysis (EDA)
This notebook analyzes the initial sequence dataset (IPR019888) fetched from UniProt.
"""
    code = """\
import pandas as pd
from IPython.display import Image, display

# Display the sequence length distribution
display(Image(filename='../results/sequence_length_distribution.png'))

# Display the summary statistics
df_summary = pd.read_csv('../results/eda_summary.csv', index_col=0)
display(df_summary)
"""
    nb['cells'] = [nbf.v4.new_markdown_cell(text), nbf.v4.new_code_cell(code)]
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/01_EDA.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_badasp_notebook():
    nb = nbf.v4.new_notebook()
    
    text1 = """\
# BADASP Results: Functional Specificity Analysis
This notebook presents the results of the BADASP (Burst After Duplication with Ancestral Sequence Predictions) pipeline applied to the IPR019888 transcription regulator family.

### Global Distribution
"""
    code1 = """\
from IPython.display import Image, display
import pandas as pd

display(Image(filename='../results/score_distribution.png'))
"""
    text2 = """\
### High-Scoring Sites
The algorithm scores alignment sites based on their conservation within sister clades and their divergence between the clades (inferred via Ancestral Sequence Reconstruction).
"""
    code2 = """\
display(Image(filename='../results/max_score_per_site.png'))

top_events, = [pd.read_csv('../results/top_events.csv')]
display(top_events)
"""
    nb['cells'] = [
        nbf.v4.new_markdown_cell(text1), nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(text2), nbf.v4.new_code_cell(code2)
    ]
    with open('notebooks/02_BADASP_Results.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_eda_notebook()
    create_badasp_notebook()
    print("Notebooks generated successfully.")
