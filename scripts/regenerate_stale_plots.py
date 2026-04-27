from __future__ import annotations

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError
import seaborn as sns
from Bio import Phylo

from src.tree_cluster import tree_to_linkage
from src.visualization import (
    build_terminal_color_map,
    generate_duplication_tree_switch_plot,
    plot_gap_percentage_per_column,
    plot_duplication_badasp_distribution,
    plot_duplication_switch_counts,
    plot_sequence_length_distribution,
    plot_topological_dendrogram,
    plot_topological_tree_dendrogram,
)
from src.pdb_mapper import PDBMapper
from src.evolutionary_analysis import (
    _load_switch_events_from_duplications,
    _plot_architecture_boxplot,
    _plot_architecture_distribution,
    _plot_clustered_heatmap,
    count_switches_per_domain,
)


REF = Path("results/badasp_scoring/badasp_sdps_duplications.csv")
MAD_TREE = Path("results/topological_clustering/mad_rooted.tree")
ASSIGNMENTS = Path("results/topological_clustering/tree_cluster_assignments.csv")
TREE_CLUSTERS = Path("results/topological_clustering/tree_clusters.csv")
SCORES_DIR = Path("results/badasp_scoring")
EVO_DIR = Path("results/evolutionary_analysis")
STRUCT_DIR = Path("results/structural_mapping")

# Recorded in PIPELINE_STATE.md (linkage-coordinate dendrogram exports).
LINKAGE_THRESHOLDS = {
    "groups": 8.579924,
    "families": 6.929765,
    "subfamilies": 4.729553,
}


def stale_visuals(reference: Path) -> list[Path]:
    exts = {".svg", ".png", ".cxc"}
    ref_mtime = reference.stat().st_mtime
    stale: list[Path] = []
    for path in Path("results").rglob("*"):
        if path.is_file() and path.suffix.lower() in exts and path.stat().st_mtime < ref_mtime:
            stale.append(path)
    return sorted(stale)


def regenerate_phase3_plots() -> None:
    tree = Phylo.read(str(MAD_TREE), "newick")
    _, linkage_rows = tree_to_linkage(tree)

    # Canonical topological tree plots.
    plot_topological_tree_dendrogram(
        tree_path=MAD_TREE,
        output_svg=Path("results/topological_clustering/tree_dendrogram.svg"),
        title="Topological Clustering Dendrogram",
        line_color="#B0B0B0",
    )

    if ASSIGNMENTS.exists():
        for plural, singular in (("groups", "group"), ("families", "family"), ("subfamilies", "subfamily")):
            terminal_colors = build_terminal_color_map(ASSIGNMENTS, f"{singular}_id")
            plot_topological_tree_dendrogram(
                tree_path=MAD_TREE,
                output_svg=Path(f"results/topological_clustering/tree_dendrogram_{plural}.svg"),
                title=f"Topological Clustering Dendrogram ({plural.capitalize()})",
                line_color="#B0B0B0",
                terminal_colors=terminal_colors,
            )

    # Linkage-coordinate dendrograms with recorded cut thresholds.
    for level, threshold in LINKAGE_THRESHOLDS.items():
        plot_topological_dendrogram(
            linkage_matrix=linkage_rows,
            output_svg=Path(f"results/topological_clustering/linkage_dendrogram_{level}.svg"),
            color_threshold=threshold,
        )

    # Regenerate clade member count summary figure from current tree clusters.
    if TREE_CLUSTERS.exists():
        clusters_df = pd.read_csv(TREE_CLUSTERS)
        summary = clusters_df.groupby("level")["member_count"].describe()[["count", "mean", "max"]].reset_index()
        out = Path("results/topological_clustering/clade_member_count_statistics.svg")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        sns.barplot(data=summary, x="level", y="count", ax=axes[0], color="#4C78A8")
        sns.barplot(data=summary, x="level", y="mean", ax=axes[1], color="#F58518")
        sns.barplot(data=summary, x="level", y="max", ax=axes[2], color="#54A24B")
        axes[0].set_title("Cluster count")
        axes[1].set_title("Mean member count")
        axes[2].set_title("Max member count")
        for ax in axes:
            ax.set_xlabel("Level")
        fig.tight_layout()
        fig.savefig(out, format="svg")
        plt.close(fig)


def regenerate_badasp_plots() -> None:
    pairwise = SCORES_DIR / "raw_pairwise_duplications.csv"
    scores = SCORES_DIR / "badasp_sdps_duplications.csv"

    plot_duplication_badasp_distribution(
        raw_pairwise_path=pairwise,
        output_svg=SCORES_DIR / "badasp_score_distribution_duplications.svg",
    )
    plot_duplication_badasp_distribution(
        raw_pairwise_path=pairwise,
        output_svg=SCORES_DIR / "badasp_score_distribution.svg",
    )

    plot_duplication_switch_counts(
        raw_pairwise_path=pairwise,
        output_svg=SCORES_DIR / "switch_counts_duplications.svg",
    )

    generate_duplication_tree_switch_plot(
        rooted_tree_path=MAD_TREE,
        raw_pairwise_duplications=pairwise,
        output_svg=SCORES_DIR / "tree_switches_duplications.svg",
    )
    generate_duplication_tree_switch_plot(
        rooted_tree_path=MAD_TREE,
        raw_pairwise_duplications=pairwise,
        output_svg=SCORES_DIR / "dendrogram_switches_duplications.svg",
    )

    # Rebuild exploratory panel plots from duplication score table.
    sdf = pd.read_csv(scores)

    # plot_1_position_counts.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = pd.DataFrame({"analysis": ["duplications"], "positions": [int((sdf["switch_count"] > 0).sum())]})
    sns.barplot(data=counts, x="analysis", y="positions", ax=ax, color="#B24A2A")
    ax.set_title("Positions with switches > 0")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_1_position_counts.svg", format="svg")
    plt.close(fig)

    # plot_2_switch_statistics.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    stats = pd.DataFrame(
        {
            "analysis": ["duplications"],
            "mean_switch_count": [float(sdf["switch_count"].mean())],
            "max_switch_count": [float(sdf["switch_count"].max())],
        }
    )
    stats_m = stats.melt(id_vars="analysis", var_name="metric", value_name="value")
    sns.barplot(data=stats_m, x="analysis", y="value", hue="metric", ax=ax)
    ax.set_title("Switch statistics")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_2_switch_statistics.svg", format="svg")
    plt.close(fig)

    # plot_3_boxplot_switches.svg
    box_df = sdf[["switch_count"]].copy()
    box_df["analysis"] = "duplications"
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=box_df, x="analysis", y="switch_count", ax=ax, color="#B24A2A")
    ax.set_title("Switch-count distribution")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_3_boxplot_switches.svg", format="svg")
    plt.close(fig)

    # plot_4_top_positions_by_level.svg
    rows = []
    top = sdf.sort_values(["switch_count", "max_score"], ascending=[False, False]).head(10)
    for _, row in top.iterrows():
        rows.append({"analysis": "duplications", "position": int(row["position"]), "switch_count": int(row["switch_count"])})
    top_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=top_df, x="position", y="switch_count", hue="analysis", ax=ax)
    ax.set_title("Top positions by duplication analysis")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_4_top_positions_by_level.svg", format="svg")
    plt.close(fig)

    # plot_5_switches_vs_badasp.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=sdf, x="switch_count", y="badasp_score", ax=ax, s=18)
    ax.set_title("Switch count vs BADASP score")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_5_switches_vs_badasp.svg", format="svg")
    plt.close(fig)

    # plot_6_switch_count_distribution.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(box_df, x="switch_count", bins=30, ax=ax, color="#B24A2A")
    ax.set_title("Switch-count distribution")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_6_switch_count_distribution.svg", format="svg")
    plt.close(fig)

    # plot_7_percentage_active_positions.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    pct = pd.DataFrame(
        {
            "analysis": ["duplications"],
            "active_pct": [100.0 * float((sdf["switch_count"] > 0).sum()) / max(1, len(sdf))],
        }
    )
    sns.barplot(data=pct, x="analysis", y="active_pct", ax=ax, color="#B24A2A")
    ax.set_title("Percentage of active positions")
    ax.set_ylabel("Percent")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_7_percentage_active_positions.svg", format="svg")
    plt.close(fig)


def regenerate_structural_artifacts() -> None:
    mapper = PDBMapper(pdb_id="2cg4", pdb_file="data/raw/2cg4.pdb")
    duplication_scores = SCORES_DIR / "badasp_sdps_duplications.csv"
    if duplication_scores.exists():
        mapper.generate_chimerax_scripts(
            alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
            sdp_csv_duplications=duplication_scores,
            output_dir=STRUCT_DIR,
        )

    physico_csv = EVO_DIR / "physicochemical_shifts_for_mapping.csv"
    if physico_csv.exists() and physico_csv.stat().st_size > 0:
        try:
            physico_df = pd.read_csv(physico_csv)
        except EmptyDataError:
            physico_df = pd.DataFrame()
        if not physico_df.empty:
            mapper.generate_physicochemical_chimerax_script(
                alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
                physicochemical_csv=physico_csv,
                output_cxc=STRUCT_DIR / "highlight_physicochemistry.cxc",
            )


def regenerate_evolutionary_plots() -> None:
    domain_arch = pd.read_json("data/domain_architecture.json", typ="series").to_dict()

    level = "duplications"
    events = _load_switch_events_from_duplications(
        tree_path=MAD_TREE,
        raw_pairwise_path=SCORES_DIR / "raw_pairwise_duplications.csv",
    )
    domain_counts = count_switches_per_domain(events, domain_arch)
    _plot_architecture_distribution(
        domain_counts,
        domain_arch,
        EVO_DIR / f"architectural_distribution_{level}.svg",
        level,
        normalize=False,
    )
    _plot_architecture_distribution(
        domain_counts,
        domain_arch,
        EVO_DIR / f"architectural_distribution_{level}_normalized.svg",
        level,
        normalize=True,
    )

    level_scores = pd.read_csv(SCORES_DIR / "badasp_sdps_duplications.csv")
    _plot_architecture_boxplot(
        level_scores,
        domain_arch,
        EVO_DIR / f"architectural_boxplot_{level}.svg",
        level,
    )

    # Rebuild matrix heatmaps from current matrix CSVs.
    matrix_specs = [
        (EVO_DIR / "distance_matrix.csv", EVO_DIR / "sdp_distance_heatmap.svg", "Top SDP C-alpha Distance Heatmap", "mako_r", "Distance (Å)", True),
        (EVO_DIR / "distance_matrix_duplications.csv", EVO_DIR / "sdp_distance_heatmap_duplications.svg", "Top SDP C-alpha Distance Heatmap (Duplications)", "mako_r", "Distance (Å)", True),
        (EVO_DIR / "coevolution_matrix.csv", EVO_DIR / "coevolution_matrix.svg", "Co-evolution Matrix", "viridis", None, False),
        (EVO_DIR / "coevolution_matrix_duplications.csv", EVO_DIR / "coevolution_matrix_duplications.svg", "Co-evolution Matrix (Duplications)", "viridis", None, False),
    ]
    for csv_path, svg_path, title, cmap, cbar_label, is_distance in matrix_specs:
        if not csv_path.exists():
            continue
        matrix = pd.read_csv(csv_path, index_col=0)
        if matrix.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, "No matrix data available", ha="center", va="center")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(svg_path, format="svg")
            plt.close(fig)
        else:
            _plot_clustered_heatmap(
                matrix=matrix,
                output_svg=svg_path,
                title=title,
                cmap=cmap,
                cbar_label=cbar_label,
                is_distance_matrix=is_distance,
            )


def regenerate_misc_qc_and_scaling() -> None:
    plot_sequence_length_distribution(
        fasta_path=Path("data/raw/IPR019888.fasta"),
        output_svg=Path("results/sequence_filtering/raw_length_dist.svg"),
    )
    plot_gap_percentage_per_column(
        msa_path=Path("data/interim/IPR019888_trimmed.aln"),
        output_svg=Path("results/alignment_qc/msa_gap_profile.svg"),
    )

    scaling_csv = Path("results/iqtree_scaling.csv")
    if scaling_csv.exists():
        df = pd.read_csv(scaling_csv)
        x_col = next((c for c in ("n_sequences", "subset_size", "sample_size") if c in df.columns), None)
        y_col = next((c for c in ("runtime_seconds", "runtime_sec", "runtime", "runtime_s") if c in df.columns), None)
        if y_col is None:
            numeric_candidates = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]
            if numeric_candidates:
                y_col = numeric_candidates[0]
        if x_col and y_col:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df, x=x_col, y=y_col, marker="o", ax=ax)
            ax.set_title("IQ-TREE ASR Scaling")
            fig.tight_layout()
            fig.savefig("results/iqtree_scaling_plot.svg", format="svg")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.regplot(data=df, x=x_col, y=y_col, scatter=True, ci=None, ax=ax)
            ax.set_title("IQ-TREE ASR Scaling (Trend)")
            fig.tight_layout()
            fig.savefig("results/iqtree_scaling_plot_extrapolated.svg", format="svg")
            plt.close(fig)
        else:
            for out_name in ("results/iqtree_scaling_plot.svg", "results/iqtree_scaling_plot_extrapolated.svg"):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.axis("off")
                ax.text(0.5, 0.5, "Unable to infer scaling columns", ha="center", va="center")
                ax.set_title("IQ-TREE ASR Scaling")
                fig.tight_layout()
                fig.savefig(out_name, format="svg")
                plt.close(fig)


def main() -> None:
    if not REF.exists():
        raise FileNotFoundError(f"Reference file not found: {REF}")

    stale_before = stale_visuals(REF)
    print(f"Reference mtime: {datetime.fromtimestamp(REF.stat().st_mtime)}")
    print(f"Stale before regeneration: {len(stale_before)}")

    regenerate_phase3_plots()
    regenerate_badasp_plots()
    regenerate_evolutionary_plots()
    regenerate_structural_artifacts()
    regenerate_misc_qc_and_scaling()

    stale_after = stale_visuals(REF)
    print(f"Stale after regeneration: {len(stale_after)}")
    if stale_after:
        print("Remaining stale files:")
        for path in stale_after:
            print(path.as_posix())


if __name__ == "__main__":
    main()
