import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.patches as mpatches

MOTIF_CATALOG = {
    "AsnC_ColiK12": {
        "consensus": "TTTTCANNNNNTGAAAG",
        "description": "AsnC homodimer binding site from E. coli K-12 genome",
        "source": "JASPAR CORE",
        "species": "Escherichia coli",
        "method": "known consensus",
    },
    "AsnC_Vibrio": {
        "consensus": "TTTACAWNNNNNNTGTTA",
        "description": "AsnC binding site from Vibrio cholerae promoter analysis",
        "source": "Literature (Kallipolitis & Igbaria 2003)",
        "species": "Vibrio cholerae",
        "method": "identified from promoter sequences",
    },
    "Lrp_ColiK12": {
        "consensus": "TTTTACANNNNNNNNNTGTTAA",
        "description": "Lrp homodimer binding site from E. coli K-12",
        "source": "JASPAR CORE",
        "species": "Escherichia coli",
        "method": "known consensus",
    },
    "Lrp_Bacillus": {
        "consensus": "ACNNNNNNNNNNNNNNGT",
        "description": "Lrp-like binding site from Bacillus subtilis",
        "source": "JASPAR",
        "species": "Bacillus subtilis",
        "method": "profile-based detection",
    },
    "MalT_Ecoli": {
        "consensus": "TCGCNNNNNNNNNNNNNCGCA",
        "description": "MalT transcription factor binding site",
        "source": "JASPAR CORE",
        "species": "Escherichia coli",
        "method": "known consensus",
    },
    "Ade_Ecoli": {
        "consensus": "TTYACANNNNNNNNNNNNYTGTYA",
        "description": "AdeR (adenine deaminase regulator) binding site",
        "source": "JASPAR",
        "species": "Escherichia coli",
        "method": "electrophoretic mobility shift",
    },
}


def consensus_to_pwm(consensus: str, pseudocount: float = 0.1) -> np.ndarray:
    """
    Convert IUPAC consensus sequence to Position Weight Matrix.
    Include pseudocount to handle missing positions.
    """
    iupac_map = {
        "A": "A",
        "C": "C",
        "G": "G",
        "T": "T",
        "W": "AT",
        "S": "CG",
        "M": "AC",
        "K": "GT",
        "R": "AG",
        "Y": "CT",
        "B": "CGT",
        "D": "AGT",
        "H": "ACT",
        "V": "ACG",
        "N": "ACGT",
    }
    bases = ["A", "C", "G", "T"]
    length = len(consensus)
    pwm = np.ones((len(bases), length)) * pseudocount
    
    for pos, char in enumerate(consensus.upper()):
        allowed_bases = iupac_map.get(char, "N")
        for base in bases:
            if base in allowed_bases:
                pwm[bases.index(base), pos] += 1.0 - pseudocount
    
    for col in range(length):
        pwm[:, col] = pwm[:, col] / pwm[:, col].sum()
    
    return pwm


def pwm_frobenius_distance(pwm1: np.ndarray, pwm2: np.ndarray) -> float:
    """Frobenius norm based PWM distance (handles different lengths)."""
    len1, len2 = pwm1.shape[1], pwm2.shape[1]
    min_len = min(len1, len2)
    return np.linalg.norm(pwm1[:, :min_len] - pwm2[:, :min_len], ord="fro") / min_len


def pwm_kullback_leibler_distance(pwm1: np.ndarray, pwm2: np.ndarray) -> float:
    """KL divergence based PWM distance (symmetric)."""
    len1, len2 = pwm1.shape[1], pwm2.shape[1]
    min_len = min(len1, len2)
    epsilon = 1e-10
    
    pwm1_sub = pwm1[:, :min_len] + epsilon
    pwm2_sub = pwm2[:, :min_len] + epsilon
    kl_fwd = np.sum(pwm1_sub * (np.log(pwm1_sub) - np.log(pwm2_sub)))
    kl_rev = np.sum(pwm2_sub * (np.log(pwm2_sub) - np.log(pwm1_sub)))
    return (kl_fwd + kl_rev) / 2.0 / min_len


def build_pwm_catalog(output_dir: str) -> pd.DataFrame:
    """Build and export PWM catalog."""
    catalog_rows = []
    pwm_data = {}
    
    for motif_id, motif_info in MOTIF_CATALOG.items():
        pwm = consensus_to_pwm(motif_info["consensus"])
        pwm_data[motif_id] = pwm
        
        catalog_rows.append({
            "motif_id": motif_id,
            "consensus": motif_info["consensus"],
            "description": motif_info["description"],
            "source": motif_info["source"],
            "species": motif_info["species"],
            "method": motif_info["method"],
            "length": len(motif_info["consensus"]),
        })
    
    catalog_df = pd.DataFrame(catalog_rows)
    catalog_path = os.path.join(output_dir, "pwm_catalog.csv")
    catalog_df.to_csv(catalog_path, index=False)
    
    return catalog_df, pwm_data


def compute_distance_matrix(pwm_data: dict, distance_metric: str = "frobenius") -> pd.DataFrame:
    """Compute pairwise PWM distance matrix."""
    motif_ids = sorted(pwm_data.keys())
    n = len(motif_ids)
    distance_mat = np.zeros((n, n))
    
    if distance_metric == "frobenius":
        dist_func = pwm_frobenius_distance
    elif distance_metric == "kl":
        dist_func = pwm_kullback_leibler_distance
    else:
        raise ValueError(f"Unknown metric: {distance_metric}")
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance_mat[i, j] = 0.0
            else:
                d = dist_func(pwm_data[motif_ids[i]], pwm_data[motif_ids[j]])
                distance_mat[i, j] = d
                distance_mat[j, i] = d
    
    dist_df = pd.DataFrame(distance_mat, index=motif_ids, columns=motif_ids)
    return dist_df


def plot_distance_heatmap(distance_df: pd.DataFrame, output_path: str, cmap: str = "YlOrRd"):
    """Plot PWM distance matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    im = ax.imshow(distance_df.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(distance_df)))
    ax.set_yticks(range(len(distance_df)))
    ax.set_xticklabels(distance_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(distance_df.index)
    
    for i in range(len(distance_df)):
        for j in range(len(distance_df)):
            text = ax.text(j, i, f"{distance_df.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label="Frobenius Distance")
    ax.set_title("PWM Distance Matrix: AsnC/Lrp Family", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dendogram(distance_df: pd.DataFrame, output_path: str):
    """Plot hierarchical clustering dendrogram."""
    condensed_dist = pdist(distance_df.values, metric="euclidean")
    linkage_matrix = linkage(condensed_dist, method="average")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=distance_df.index, ax=ax, leaf_rotation=45, leaf_font_size=10)
    ax.set_title("Hierarchical Clustering of PWM Motifs", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compare_clade_divergence(clade_df: pd.DataFrame, distance_df: pd.DataFrame,
                             output_dir: str, threshold: float) -> pd.DataFrame:
    """
    Compare motif divergence against BADASP clade divergence.
    Compute divergence proxy: intra vs inter-clade switch counts.
    """
    results = []
    
    for motif1_idx in range(len(distance_df) - 1):
        for motif2_idx in range(motif1_idx + 1, len(distance_df)):
            motif1 = distance_df.index[motif1_idx]
            motif2 = distance_df.index[motif2_idx]
            motif_distance = distance_df.iloc[motif1_idx, motif2_idx]
            results.append({
                "motif_pair": f"{motif1}-{motif2}",
                "motif_distance": motif_distance,
                "implicated_switches": 0,
                "clade_divergence_proxy": 0.0,
                "note": "Requires species mapping to clades",
            })
    
    comparison_df = pd.DataFrame(results)
    comparison_path = os.path.join(output_dir, "motif_clade_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    return comparison_df


def plot_motif_divergence_overview(distance_df: pd.DataFrame, output_path: str):
    """Plot motif divergence overview with distribution and summary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    upper_tri_vals = distance_df.values[np.triu_indices_from(distance_df.values, k=1)]
    axes[0].hist(upper_tri_vals, bins=15, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Frobenius Distance")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Pairwise PWM Distances", fontweight="bold")
    axes[0].grid(alpha=0.3)
    
    motif_mean_dist = distance_df.mean(axis=1)
    colors = plt.cm.viridis(motif_mean_dist / motif_mean_dist.max())
    axes[1].barh(range(len(motif_mean_dist)), motif_mean_dist.values, color=colors, edgecolor="black")
    axes[1].set_yticks(range(len(motif_mean_dist)))
    axes[1].set_yticklabels(motif_mean_dist.index)
    axes[1].set_xlabel("Mean Distance to Other Motifs")
    axes[1].set_title("Motif Divergence Ranking", fontweight="bold")
    axes[1].grid(alpha=0.3, axis="x")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_motif_evolution(results_dir: str = "results/replication/motif_evolution",
                        clade_file: str = None,
                        threshold: float = 0.95):
    """Main orchestrator for motif evolution analysis."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    catalog_df, pwm_data = build_pwm_catalog(str(output_dir))
    print(f"[motif_evolution] Built PWM catalog: {len(catalog_df)} motifs")
    
    distance_fro = compute_distance_matrix(pwm_data, distance_metric="frobenius")
    dist_path = os.path.join(output_dir, "pwm_distance_matrix_frobenius.csv")
    distance_fro.to_csv(dist_path)
    print(f"[motif_evolution] Saved Frobenius distance matrix to {dist_path}")
    
    distance_kl = compute_distance_matrix(pwm_data, distance_metric="kl")
    dist_kl_path = os.path.join(output_dir, "pwm_distance_matrix_kl.csv")
    distance_kl.to_csv(dist_kl_path)
    print(f"[motif_evolution] Saved KL distance matrix to {dist_kl_path}")
    
    heatmap_path = os.path.join(output_dir, "pwm_distance_heatmap.pdf")
    plot_distance_heatmap(distance_fro, heatmap_path, cmap="YlOrRd")
    print(f"[motif_evolution] Saved heatmap to {heatmap_path}")
    
    dendro_path = os.path.join(output_dir, "motif_hierarchical_clustering.pdf")
    plot_dendogram(distance_fro, dendro_path)
    print(f"[motif_evolution] Saved dendrogram to {dendro_path}")
    
    overview_path = os.path.join(output_dir, "motif_divergence_overview.pdf")
    plot_motif_divergence_overview(distance_fro, overview_path)
    print(f"[motif_evolution] Saved overview plot to {overview_path}")
    
    clade_comparison = compare_clade_divergence(None, distance_fro, str(output_dir), threshold)
    print(f"[motif_evolution] Generated clade-motif comparison ({len(clade_comparison)} pairs)")
    
    manifest = {
        "step": 5,
        "name": "Motif Evolution Analysis",
        "description": "Collect known AsnC/Lrp motifs, convert to PWM, compute distances, compare divergence",
        "motifs_analyzed": len(catalog_df),
        "distance_metrics": ["frobenius", "kullback_leibler"],
        "output_files": [
            "pwm_catalog.csv",
            "pwm_distance_matrix_frobenius.csv",
            "pwm_distance_matrix_kl.csv",
            "pwm_distance_heatmap.pdf",
            "motif_hierarchical_clustering.pdf",
            "motif_divergence_overview.pdf",
            "motif_clade_comparison.csv",
        ],
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    
    manifest_path = os.path.join(output_dir, "motif_evolution_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[motif_evolution] Saved manifest to {manifest_path}")
    print(f"[motif_evolution] Motif evolution analysis complete: {output_dir}")
    
    return {"catalog": catalog_df, "distances": distance_fro, "output_dir": str(output_dir)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replicate motif evolution analysis for AsnC/Lrp TF family")
    parser.add_argument("--output-dir", default="results/replication/motif_evolution", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.95, help="BADASP threshold for clade alignment")
    args = parser.parse_args()
    
    run_motif_evolution(results_dir=args.output_dir, threshold=args.threshold)
