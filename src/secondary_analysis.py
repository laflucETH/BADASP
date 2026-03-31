"""
Replicates the secondary structural analyses from Beltrao & Bradley (2019):
1. Structural Clustering (Kolmogorov-Smirnov Test): Checks if highly switching
   residues are significantly closer to each other in 3D space (C-alpha distances)
   compared to background residue distances.
2. Functional Enrichment (Fisher's Exact Test): Checks if highly switching
   residues are disproportionately located within the DNA-contacting HTH domain.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, fisher_exact
import Bio.PDB
from Bio.PDB.PDBList import PDBList
import warnings

warnings.filterwarnings('ignore', category=Bio.PDB.PDBExceptions.PDBConstructionWarning)


def get_ca_coordinates(pdb_id, chain_id):
    """Downloads PDB and extracts C-alpha coordinates for the specified chain."""
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format="mmCif", pdir="data/raw", obsolete=False)
    
    parser = Bio.PDB.MMCIFParser()
    structure = parser.get_structure(pdb_id, pdb_file)
    
    ca_coords = {}
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if Bio.PDB.is_aa(residue) and 'CA' in residue:
                        res_id = residue.get_id()[1]
                        ca_coords[res_id] = residue['CA'].get_coord()
                break # Only process first matching chain in first model
        break
    
    return ca_coords

def run_secondary_analyses(scores_csv, mapping_csv, output_dir, pdb_id="2CG4", chain_id="A"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load domain structure vs functional switches
    mapping = pd.read_csv(mapping_csv) if os.path.exists(mapping_csv) else None
    
    if mapping is not None and 'PDB_Residue' in mapping.columns:
        domain_df = mapping.copy()
    else:
        # Fallback if no valid mapping exists
        df = pd.read_csv(scores_csv)
        threshold = df['BADASP_Score'].quantile(0.95)
        switches = df[df['BADASP_Score'] > threshold]
        domain_df = switches.groupby('Site').size().reset_index(name='Switch_Count').rename(columns={'Site':'PDB_Residue'})
        
    if 'Switch_Count' not in domain_df.columns:
        print("Required Switch_Count not found in mapping.")
        return
    
    if 'Is_Frequent' not in domain_df.columns:
        switch_counts = domain_df['Switch_Count']
        if len(switch_counts) > 0:
            freq_threshold = max(1.0, switch_counts.quantile(0.90))
        else:
            freq_threshold = 1.0
        domain_df['Is_Frequent'] = domain_df['Switch_Count'] >= freq_threshold
        
    frequent_residues = domain_df[domain_df['Is_Frequent'] == True]['PDB_Residue'].dropna().astype(int).tolist()
    
    # Generic residues are essentially all HTH residues (12-73) that are NOT frequently switching
    hth_range = set(range(12, 74))
    generic_residues = [r for r in hth_range if r not in frequent_residues]
    
    print(f"Identified {len(frequent_residues)} frequently switching residues (th={freq_threshold}) and {len(generic_residues)} generic residues.")
    
    # 2. Extract 3D Coordinates
    print(f"Fetching {pdb_id} coordinates...")
    ca_coords = get_ca_coordinates(pdb_id, chain_id)
    
    freq_coords = [ca_coords[r] for r in frequent_residues if r in ca_coords]
    generic_coords = [ca_coords[r] for r in generic_residues if r in ca_coords]
    
    if len(freq_coords) < 2 or len(generic_coords) < 2:
        print("Not enough coordinates matched to perform KS test.")
        return
        
    # Calculate pairwise distances
    freq_distances = []
    for i in range(len(freq_coords)):
        for j in range(i+1, len(freq_coords)):
            dist = np.linalg.norm(freq_coords[i] - freq_coords[j])
            freq_distances.append(dist)
            
    generic_distances = []
    for i in range(len(generic_coords)):
        for j in range(i+1, len(generic_coords)):
            dist = np.linalg.norm(generic_coords[i] - generic_coords[j])
            generic_distances.append(dist)
            
    # 3. Kolmogorov-Smirnov Test (Structural Clustering)
    ks_stat, p_val = ks_2samp(freq_distances, generic_distances)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    try:
        sns.kdeplot(freq_distances, fill=True, color='red', label='Freq Switches', warn_singular=False)
        sns.kdeplot(generic_distances, fill=True, color='blue', label='Generic', warn_singular=False)
    except Exception:
        pass
    plt.title(f"3D Clustering (KS-stat={ks_stat:.2f}, p={p_val:.2f})")
    plt.xlabel(r"Pairwise Distance ($\AA$)")
    plt.legend()
    
    # Calculate shortest distance to Recognition Helix (Residues 35-49 in 2CG4)
    rec_helix = list(range(35, 50))
    rec_coords = [ca_coords[r] for r in rec_helix if r in ca_coords]
    
    def min_dist_to_helix(coord):
        if not rec_coords: return 0
        return min(np.linalg.norm(coord - hc) for hc in rec_coords)
        
    freq_dist_to_dna = [min_dist_to_helix(c) for c in freq_coords]
    generic_dist_to_dna = [min_dist_to_helix(c) for c in generic_coords]
    
    ks_stat_dna, p_val_dna = ks_2samp(freq_dist_to_dna, generic_dist_to_dna)
    
    plt.subplot(1, 2, 2)
    try:
        sns.kdeplot(freq_dist_to_dna, fill=True, color='red', label='Freq Switches', warn_singular=False)
        sns.kdeplot(generic_dist_to_dna, fill=True, color='blue', label='Generic', warn_singular=False)
    except Exception:
        pass
    plt.title(f"Distance to DNA Helix (p={p_val_dna:.2f})")
    plt.xlabel(r"Min Distance to Helix 3 ($\AA$)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'structural_clustering_ks_test.png'), dpi=300)
    print(f"Saved structural clustering plots.")
    
    # 4. Fisher's Exact Test (Enrichment directly in Recognition Helix 35-49)
    # The AsnC HTH DNA-binding recognition helix is Helix C (residues 35-49)
    hth_start, hth_end = 35, 49
    
    a = len([r for r in frequent_residues if hth_start <= r <= hth_end])
    b = len([r for r in generic_residues if hth_start <= r <= hth_end])
    c = len([r for r in frequent_residues if not (hth_start <= r <= hth_end)])
    d = len([r for r in generic_residues if not (hth_start <= r <= hth_end)])
    
    contingency = [[a, b], [c, d]]
    odds_ratio, fisher_p = fisher_exact(contingency, alternative='greater')
    
    print("\n--- Fisher's Exact Test (Recognition Helix Enrichment) ---")
    print(f"Table: [[{a}, {b}], [{c}, {d}]]")
    print(f"Odds Ratio: {odds_ratio:.2f}, p-value: {fisher_p:.4f}")
    
    with open(os.path.join(output_dir, 'secondary_analysis_stats.txt'), 'w') as f:
        f.write("=== Beltrao/Bradley Replicative Secondary Analyses ===\n\n")
        f.write("1. Structural Clustering (KS Test on internal distances)\n")
        f.write(f"KS-statistic: {ks_stat:.4f}, P-value: {p_val:.2e}\n\n")
        
        f.write("2. Proximity to DNA Recognition Helix (KS Test)\n")
        f.write(f"KS-statistic: {ks_stat_dna:.4f}, P-value: {p_val_dna:.2e}\n\n")
            
        f.write("3. Functional Enrichment (Fisher's Exact Test on DNA-Binding Helix 35-49)\n")
        f.write(f"Frequent Switches in Rec Helix: {a}\n")
        f.write(f"Generic Residues in Rec Helix: {b}\n")
        f.write(f"Frequent Switches Outside: {c}\n")
        f.write(f"Generic Residues Outside: {d}\n")
        f.write(f"Odds Ratio: {odds_ratio:.2f}\n")
        f.write(f"P-value: {fisher_p:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default="results/badasp_scores_hth.csv")
    parser.add_argument("--mapping", type=str, default="results/hth/per_residue_scores.csv")
    parser.add_argument("--output", type=str, default="results/hth")
    args = parser.parse_args()
    
    run_secondary_analyses(args.scores, args.mapping, args.output)
