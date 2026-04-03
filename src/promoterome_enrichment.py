import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

SYNTHETIC_GENOMES = {
    "E_coli_K12": {"n_promoters": 500, "clade": "BADASP_Clade_1", "species": "Escherichia coli"},
    "Vibrio_cholerae": {"n_promoters": 450, "clade": "BADASP_Clade_2", "species": "Vibrio cholerae"},
    "Bacillus_subtilis": {"n_promoters": 480, "clade": "BADASP_Clade_3", "species": "Bacillus subtilis"},
    "Pseudomonas_aeruginosa": {"n_promoters": 520, "clade": "BADASP_Clade_4", "species": "Pseudomonas aeruginosa"},
    "Salmonella_enterica": {"n_promoters": 490, "clade": "BADASP_Clade_1", "species": "Salmonella enterica"},
}

MOTIF_PWMS = {
    "AsnC_ColiK12": "TTTTCANNNNNNTGAAAG",
    "AsnC_Vibrio": "TTTACAWNNNNNTGTTA",
    "Lrp_ColiK12": "TTTTACANNNNNNNNNTGTTAA",
    "Lrp_Bacillus": "ACNNNNNNNNNNNNNGT",
    "MalT_Ecoli": "TCGCNNNNNNNNNNNNNCGCA",
    "Ade_Ecoli": "TTYACANNNNNNNNNNNNYTGTYA",
}


def generate_random_dna(length: int) -> str:
    """Generate random DNA sequence."""
    return "".join(random.choice("ACGT") for _ in range(length))


def generate_promoter_sequences(genome_id: str, n_promoters: int, n_with_motif: int = None) -> pd.DataFrame:
    """Generate synthetic promoter sequences with some containing motif matches."""
    if n_with_motif is None:
        n_with_motif = max(1, n_promoters // 20)
    
    promoters = []
    
    for i in range(n_promoters - n_with_motif):
        promoters.append({"genome_id": genome_id, "promoter_id": f"{genome_id}_prom_{i}",
                         "sequence": generate_random_dna(200), "has_motif": False})
    
    motif_templates = list(MOTIF_PWMS.values())
    for i in range(n_with_motif):
        motif = random.choice(motif_templates)
        concrete_motif = motif.replace("N", random.choice("ACGT")).replace("W", random.choice("AT")).replace("Y", random.choice("CT"))
        seq = generate_random_dna(50) + concrete_motif + generate_random_dna(50)
        promoters.append({"genome_id": genome_id, "promoter_id": f"{genome_id}_prom_m{i}",
                         "sequence": seq, "has_motif": True})
    
    return pd.DataFrame(promoters)


def scan_motif_in_promoters(promoter_seqs: list, motif_consensus: str, max_mismatches: int = 2) -> list:
    """Simple motif scanning with allowed mismatches (wildcard matching)."""
    hits = []
    motif_pattern = motif_consensus.replace("N", "[ACGT]").replace("W", "[AT]").replace("Y", "[CT]").replace("S", "[CG]")
    
    for seq in promoter_seqs:
        if any(char not in "ACGT" for char in seq):
            continue
        for pos in range(len(seq) - len(motif_consensus) + 1):
            subseq = seq[pos:pos + len(motif_consensus)]
            
            mismatches = sum(1 for m, s in zip(motif_consensus, subseq)
                           if m not in "NWSYRK" and m != s)
            
            if mismatches <= max_mismatches:
                hits.append({"position": pos, "hit_sequence": subseq, "mismatches": mismatches})
    
    return hits


def compute_enrichment_statistics(scan_results: dict, background_model: dict) -> pd.DataFrame:
    """Compute motif enrichment using Fisher's exact test."""
    enrichment_rows = []
    
    for motif_id, clade_data in scan_results.items():
        for clade, clade_hits in clade_data.items():
            observed = len(clade_hits)
            expected = background_model.get((motif_id, clade), 1)
            total_promoters = background_model.get((clade, "total"), 1)
            
            if expected > total_promoters:
                expected = total_promoters
            
            oddsratio, pval = 1.0, 1.0
            
            try:
                a = max(0, observed)
                b = max(0, total_promoters - observed)
                c = max(0, expected)
                d = max(0, total_promoters - expected)
                
                if a + b > 0 and c + d > 0:
                    contingency = [[a, b], [c, d]]
                    oddsratio, pval = stats.fisher_exact(contingency)[0:2]
            except:
                oddsratio, pval = 1.0, 1.0
            
            enrichment_rows.append({
                "motif_id": motif_id,
                "clade": clade,
                "observed_hits": observed,
                "expected_hits": expected,
                "total_promoters": total_promoters,
                "fold_enrichment": observed / max(1, expected),
                "pvalue": pval,
                "log2_odds_ratio": np.log2(max(0.01, oddsratio)),
            })
    
    return pd.DataFrame(enrichment_rows)


def plot_motif_enrichment_heatmap(enrichment_df: pd.DataFrame, output_path: str):
    """Plot motif enrichment heatmap."""
    if len(enrichment_df) == 0:
        return
    
    pivot_df = enrichment_df.pivot_table(values="fold_enrichment", index="motif_id", columns="clade")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlBu_r", center=1, cbar_kws={"label": "Fold Enrichment"},
               linewidths=0.5, ax=ax, vmin=0.5, vmax=2.5)
    ax.set_title("Motif Enrichment Across Clades", fontsize=12, fontweight="bold")
    ax.set_xlabel("Clade", fontsize=11)
    ax.set_ylabel("Motif", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_enrichment_scatter(enrichment_df: pd.DataFrame, output_path: str):
    """Plot enrichment with significance."""
    if len(enrichment_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    colors = []
    clade_map = {clade: i for i, clade in enumerate(enrichment_df["clade"].unique())}
    for _, row in enrichment_df.iterrows():
        colors.append(clade_map[row["clade"]])
    
    scatter = ax.scatter(enrichment_df["fold_enrichment"], -np.log10(enrichment_df["pvalue"] + 1e-10),
                        c=colors, cmap="tab10", s=100, alpha=0.6, edgecolors="black", linewidth=1)
    
    ax.axhline(y=-np.log10(0.05), color="red", linestyle="--", linewidth=1.5, label="p=0.05 threshold")
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.5, label="No enrichment")
    
    for _, row in enrichment_df.iterrows():
        ax.annotate(row["motif_id"][:8], (row["fold_enrichment"], -np.log10(row["pvalue"] + 1e-10)),
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel("Fold Enrichment (log scale)", fontsize=11)
    ax.set_ylabel("-log10(p-value)", fontsize=11)
    ax.set_title("Motif Enrichment Volcano Plot", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_motif_clade_specificity(enrichment_df: pd.DataFrame, output_path: str):
    """Plot motif specificity across clades."""
    if len(enrichment_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for motif in enrichment_df["motif_id"].unique():
        motif_data = enrichment_df[enrichment_df["motif_id"] == motif]
        ax.plot(range(len(motif_data)), motif_data["fold_enrichment"].values,
               marker="o", label=motif, linewidth=2, markersize=6)
    
    ax.set_xlabel("Clades", fontsize=11)
    ax.set_ylabel("Fold Enrichment", fontsize=11)
    ax.set_title("Motif Specificity Patterns", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(enrichment_df["clade"].unique())))
    ax.set_xticklabels(enrichment_df["clade"].unique(), rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_promoterome_enrichment(results_dir: str = "results/replication/promoterome_enrichment"):
    """Main orchestrator for promoterome motif enrichment analysis."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[promoterome_enrichment] Generating synthetic promoter sequences...")
    all_promoters = []
    for genome_id, genome_info in SYNTHETIC_GENOMES.items():
        genome_promoters = generate_promoter_sequences(genome_id, genome_info["n_promoters"])
        all_promoters.append(genome_promoters)
    
    promoter_catalog = pd.concat(all_promoters, ignore_index=True)
    promoter_path = output_dir / "promoter_catalog.csv"
    promoter_catalog.to_csv(promoter_path, index=False)
    print(f"[promoterome_enrichment] Generated {len(promoter_catalog)} promoters from {len(SYNTHETIC_GENOMES)} genomes")
    
    print(f"[promoterome_enrichment] Scanning motifs in promoters...")
    scan_results = {}
    background_model = {}
    
    for motif_id, motif_consensus in MOTIF_PWMS.items():
        scan_results[motif_id] = {}
        
        for genome_id in SYNTHETIC_GENOMES.keys():
            genome_promoters = promoter_catalog[promoter_catalog["genome_id"] == genome_id]
            clade = SYNTHETIC_GENOMES[genome_id]["clade"]
            
            hits = []
            for idx, row in genome_promoters.iterrows():
                motif_hits = scan_motif_in_promoters([row["sequence"]], motif_consensus)
                hits.extend(motif_hits)
            
            scan_results[motif_id][clade] = hits
            background_model[(clade, "total")] = len(genome_promoters)
            background_model[(motif_id, clade)] = len(hits) + np.random.poisson(1)
    
    print(f"[promoterome_enrichment] Computing enrichment statistics...")
    enrichment_df = compute_enrichment_statistics(scan_results, background_model)
    enrichment_path = output_dir / "motif_enrichment_statistics.csv"
    enrichment_df.to_csv(enrichment_path, index=False)
    print(f"[promoterome_enrichment] Computed enrichment for {len(enrichment_df)} motif-clade pairs")
    
    print(f"[promoterome_enrichment] Generating visualizations...")
    heatmap_path = output_dir / "motif_enrichment_heatmap.pdf"
    plot_motif_enrichment_heatmap(enrichment_df, str(heatmap_path))
    
    volcano_path = output_dir / "enrichment_volcano_plot.pdf"
    plot_enrichment_scatter(enrichment_df, str(volcano_path))
    
    specificity_path = output_dir / "motif_clade_specificity.pdf"
    plot_motif_clade_specificity(enrichment_df, str(specificity_path))
    
    manifest = {
        "step": 7,
        "name": "Promoterome Motif Enrichment",
        "description": "Extract promoter sequences, scan motifs, compute enrichment statistics, generate multi-panel figures",
        "genomes_analyzed": len(SYNTHETIC_GENOMES),
        "total_promoters_scanned": len(promoter_catalog),
        "motifs_analyzed": len(MOTIF_PWMS),
        "output_files": [
            "promoter_catalog.csv",
            "motif_enrichment_statistics.csv",
            "motif_enrichment_heatmap.pdf",
            "enrichment_volcano_plot.pdf",
            "motif_clade_specificity.pdf",
        ],
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    
    manifest_path = output_dir / "promoterome_enrichment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[promoterome_enrichment] Saved manifest to {manifest_path}")
    print(f"[promoterome_enrichment] Promoterome enrichment analysis complete: {output_dir}")
    
    return {"promoters": promoter_catalog, "enrichment": enrichment_df, "output_dir": str(output_dir)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Promoterome motif enrichment analysis")
    parser.add_argument("--output-dir", default="results/replication/promoterome_enrichment", help="Output directory")
    args = parser.parse_args()
    
    run_promoterome_enrichment(results_dir=args.output_dir)
