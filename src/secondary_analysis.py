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
    
    # Beltrao defined "frequently switching" as > 90th percentile of switch counts
    freq_threshold = domain_df['Switch_Count'].quantile(0.90)
    # Ensure minimum 1 switch
    if freq_threshold == 0: freq_threshold = 1 
    
    domain_df['Is_Frequent'] = domain_df['Switch_Count'] >= freq_threshold
    frequent_residues = domain_df[domain_df['Is_Frequent']]['PDB_Residue'].astype(int).tolist()
    generic_residues = domain_df[~domain_df['Is_Frequent']]['PDB_Residue'].astype(int).tolist()
    
    print(f"Identified {len(frequent_residues)} frequently switching residues (th={freq_threshold})")
    
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
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(freq_distances, fill=True, color='red', label='Frequently Switching Residues')
    sns.kdeplot(generic_distances, fill=True, color='blue', label='Generic Residues')
    plt.title(f"Structural Clustering of Switches (PDB: {pdb_id})\nKS-stat={ks_stat:.3f}, p={p_val:.2e}")
    plt.xlabel(r"Pairwise C-$\alpha$ Distance ($\AA$)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'structural_clustering_ks_test.png'), dpi=300)
    print(f"Saved structural clustering plot. KS p-value: {p_val:.2e}")
    
    # 4. Fisher's Exact Test (Enrichment in DNA-Contacting HTH Domain)
    # The AsnC HTH DNA-binding domain is located exactly at N-terminus (residues 12-73 in P0ACJ0/2CG4)
    hth_start, hth_end = 12, 73
    
    domain_df['Is_HTH'] = (domain_df['PDB_Residue'] >= hth_start) & (domain_df['PDB_Residue'] <= hth_end)
    
    # Contingency Table
    #                 | Frequent Switch | Non-Frequent Switch
    # ----------------|-----------------|---------------------
    # In HTH Domain   |       A         |          B
    # Outside HTH     |       C         |          D
    
    a = len(domain_df[(domain_df['Is_Frequent'] == True) & (domain_df['Is_HTH'] == True)])
    b = len(domain_df[(domain_df['Is_Frequent'] == False) & (domain_df['Is_HTH'] == True)])
    c = len(domain_df[(domain_df['Is_Frequent'] == True) & (domain_df['Is_HTH'] == False)])
    d = len(domain_df[(domain_df['Is_Frequent'] == False) & (domain_df['Is_HTH'] == False)])
    
    contingency = [[a, b], [c, d]]
    odds_ratio, fisher_p = fisher_exact(contingency, alternative='greater')
    
    print("\n--- Fisher's Exact Test (HTH Domain Enrichment) ---")
    print(f"Table: [[{a}, {b}], [{c}, {d}]]")
    print(f"Odds Ratio: {odds_ratio:.2f}, p-value: {fisher_p:.4f}")
    
    with open(os.path.join(output_dir, 'secondary_analysis_stats.txt'), 'w') as f:
        f.write("=== Beltrao/Bradley Replicative Secondary Analyses ===\n\n")
        f.write("1. Structural Clustering (KS Test)\n")
        f.write(f"KS-statistic: {ks_stat:.4f}\n")
        f.write(f"P-value: {p_val:.2e}\n")
        if p_val < 0.05:
            f.write("Conclusion: Frequently switching residues are significantly clustered in 3D space.\n\n")
        else:
            f.write("Conclusion: Switch residues do not significantly cluster differently than generic residues.\n\n")
            
        f.write("2. Functional Enrichment (Fisher's Exact Test on HTH residues 12-73)\n")
        f.write(f"Frequent Switches in HTH: {a}\n")
        f.write(f"Non-Frequent in HTH: {b}\n")
        f.write(f"Frequent Switches Outside: {c}\n")
        f.write(f"Non-Frequent Outside: {d}\n")
        f.write(f"Odds Ratio: {odds_ratio:.2f}\n")
        f.write(f"P-value: {fisher_p:.4f}\n")
        if fisher_p < 0.05:
            f.write("Conclusion: Switches are significantly enriched in the DNA-binding domain.\n")
        else:
            f.write("Conclusion: No significant enrichment in the DNA-binding domain.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, default="results/badasp_scores_hth.csv")
    parser.add_argument("--mapping", type=str, default="results/hth/per_residue_scores.csv")
    parser.add_argument("--output", type=str, default="results/hth")
    args = parser.parse_args()
    
    run_secondary_analyses(args.scores, args.mapping, args.output)
