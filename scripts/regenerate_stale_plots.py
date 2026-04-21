from __future__ import annotations

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio import Phylo

from src.tree_cluster import tree_to_linkage
from src.visualization import (
    build_terminal_color_map,
    generate_dendrogram_switch_plots,
    plot_badasp_score_distribution,
    plot_gap_percentage_per_column,
    plot_hierarchical_badasp_distributions,
    plot_hierarchical_switch_counts,
    plot_individual_hierarchical_badasp_distributions,
    plot_sequence_length_distribution,
    plot_topological_dendrogram,
    plot_topological_tree_dendrogram,
)
from src.pdb_mapper import PDBMapper
from src.evolutionary_analysis import (
    _load_switch_events_for_level,
    _plot_architecture_boxplot,
    _plot_architecture_distribution,
    _plot_clustered_heatmap,
    count_switches_per_domain,
)


REF = Path("results/badasp_scoring/badasp_scores_groups.csv")
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
    gp = SCORES_DIR / "raw_pairwise_groups.csv"
    fp = SCORES_DIR / "raw_pairwise_families.csv"
    sp = SCORES_DIR / "raw_pairwise_subfamilies.csv"

    gs = SCORES_DIR / "badasp_scores_groups.csv"
    fs = SCORES_DIR / "badasp_scores_families.csv"
    ss = SCORES_DIR / "badasp_scores_subfamilies.csv"

    plot_individual_hierarchical_badasp_distributions(
        group_pairwise=gp,
        family_pairwise=fp,
        subfamily_pairwise=sp,
        output_group_svg=SCORES_DIR / "badasp_score_distribution_groups.svg",
        output_family_svg=SCORES_DIR / "badasp_score_distribution_families.svg",
        output_subfamily_svg=SCORES_DIR / "badasp_score_distribution_subfamilies.svg",
    )

    # Legacy single distribution file: use subfamily distribution as the default summary view.
    plot_badasp_score_distribution(
        raw_pairwise_path=sp,
        output_svg=SCORES_DIR / "badasp_score_distribution.svg",
        title="BADASP Score Distribution",
        color="#2CA02C",
    )

    plot_hierarchical_badasp_distributions(
        group_pairwise=gp,
        family_pairwise=fp,
        subfamily_pairwise=sp,
        output_svg=SCORES_DIR / "hierarchical_distributions.svg",
    )

    plot_hierarchical_switch_counts(
        group_scores=gs,
        family_scores=fs,
        subfamily_scores=ss,
        output_svg=SCORES_DIR / "hierarchical_switch_counts.svg",
    )

    generate_dendrogram_switch_plots(
        tree_path=MAD_TREE,
        assignments_path=ASSIGNMENTS,
        raw_pairwise_groups=gp,
        raw_pairwise_families=fp,
        raw_pairwise_subfamilies=sp,
        output_groups_svg=SCORES_DIR / "dendrogram_switches_groups.svg",
        output_families_svg=SCORES_DIR / "dendrogram_switches_families.svg",
        output_subfamilies_svg=SCORES_DIR / "dendrogram_switches_subfamilies.svg",
        group_threshold=LINKAGE_THRESHOLDS["groups"],
        family_threshold=LINKAGE_THRESHOLDS["families"],
        subfamily_threshold=LINKAGE_THRESHOLDS["subfamilies"],
        min_clade_size=5,
    )

    # Rebuild legacy exploratory panel plots from current score tables.
    gdf = pd.read_csv(gs)
    fdf = pd.read_csv(fs)
    sdf = pd.read_csv(ss)
    levels = [("Groups", gdf), ("Families", fdf), ("Subfamilies", sdf)]

    # plot_1_position_counts.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = pd.DataFrame({"level": [name for name, df in levels], "positions": [int((df["switch_count"] > 0).sum()) for name, df in levels]})
    sns.barplot(data=counts, x="level", y="positions", ax=ax, palette="Set2")
    ax.set_title("Positions with switches > 0")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_1_position_counts.svg", format="svg")
    plt.close(fig)

    # plot_2_switch_statistics.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    stats = pd.DataFrame(
        {
            "level": [name for name, _ in levels],
            "mean_switch_count": [float(df["switch_count"].mean()) for _, df in levels],
            "max_switch_count": [float(df["switch_count"].max()) for _, df in levels],
        }
    )
    stats_m = stats.melt(id_vars="level", var_name="metric", value_name="value")
    sns.barplot(data=stats_m, x="level", y="value", hue="metric", ax=ax)
    ax.set_title("Switch statistics")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_2_switch_statistics.svg", format="svg")
    plt.close(fig)

    # plot_3_boxplot_switches.svg
    box_df = []
    for name, df in levels:
        tmp = df[["switch_count"]].copy()
        tmp["level"] = name
        box_df.append(tmp)
    box_df = pd.concat(box_df, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=box_df, x="level", y="switch_count", ax=ax)
    ax.set_title("Switch-count distribution by level")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_3_boxplot_switches.svg", format="svg")
    plt.close(fig)

    # plot_4_top_positions_by_level.svg
    rows = []
    for name, df in levels:
        top = df.sort_values(["switch_count", "max_score"], ascending=[False, False]).head(10)
        for _, row in top.iterrows():
            rows.append({"level": name, "position": int(row["position"]), "switch_count": int(row["switch_count"])})
    top_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=top_df, x="position", y="switch_count", hue="level", ax=ax)
    ax.set_title("Top positions by level")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_4_top_positions_by_level.svg", format="svg")
    plt.close(fig)

    # plot_5_switches_vs_badasp.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=sdf, x="switch_count", y="badasp_score", ax=ax, s=18)
    ax.set_title("Switch count vs BADASP score (subfamilies)")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_5_switches_vs_badasp.svg", format="svg")
    plt.close(fig)

    # plot_6_switch_count_distribution.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(box_df, x="switch_count", hue="level", multiple="layer", bins=30, ax=ax)
    ax.set_title("Switch-count distribution")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_6_switch_count_distribution.svg", format="svg")
    plt.close(fig)

    # plot_7_percentage_active_positions.svg
    fig, ax = plt.subplots(figsize=(8, 4))
    pct = pd.DataFrame(
        {
            "level": [name for name, _ in levels],
            "active_pct": [100.0 * float((df["switch_count"] > 0).sum()) / max(1, len(df)) for _, df in levels],
        }
    )
    sns.barplot(data=pct, x="level", y="active_pct", ax=ax, palette="Set1")
    ax.set_title("Percentage of active positions")
    ax.set_ylabel("Percent")
    fig.tight_layout()
    fig.savefig(SCORES_DIR / "plot_7_percentage_active_positions.svg", format="svg")
    plt.close(fig)


def regenerate_structural_artifacts() -> None:
    mapper = PDBMapper(pdb_id="2cg4", pdb_file="data/raw/2cg4.pdb")
    mapper.generate_chimerax_scripts(
        alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
        sdp_csv_groups=SCORES_DIR / "badasp_scores_groups.csv",
        sdp_csv_families=SCORES_DIR / "badasp_scores_families.csv",
        sdp_csv_subfamilies=SCORES_DIR / "badasp_scores_subfamilies.csv",
        output_dir=STRUCT_DIR,
    )

    mapper.generate_physicochemical_chimerax_script(
        alignment_path=Path("data/interim/IPR019888_trimmed.aln"),
        physicochemical_csv=EVO_DIR / "physicochemical_shifts_for_mapping.csv",
        output_cxc=STRUCT_DIR / "highlight_physicochemistry.cxc",
    )


def regenerate_evolutionary_plots() -> None:
    domain_arch = pd.read_json("data/domain_architecture.json", typ="series").to_dict()

    # Rebuild per-level architecture plots and boxplots from current score/event tables.
    for level in ("groups", "families", "subfamilies"):
        events = _load_switch_events_for_level(
            tree_path=MAD_TREE,
            assignments_path=ASSIGNMENTS,
            raw_pairwise_path=SCORES_DIR / f"raw_pairwise_{level}.csv",
            level=level,
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

        level_scores = pd.read_csv(SCORES_DIR / f"badasp_scores_{level}.csv")
        _plot_architecture_boxplot(
            level_scores,
            domain_arch,
            EVO_DIR / f"architectural_boxplot_{level}.svg",
            level,
        )

    # Rebuild matrix heatmaps from current matrix CSVs.
    matrix_specs = [
        (EVO_DIR / "distance_matrix.csv", EVO_DIR / "sdp_distance_heatmap.svg", "Top SDP C-alpha Distance Heatmap", "mako_r", "Distance (Å)", True),
        (EVO_DIR / "distance_matrix_groups.csv", EVO_DIR / "sdp_distance_heatmap_groups.svg", "Top SDP C-alpha Distance Heatmap (Groups)", "mako_r", "Distance (Å)", True),
        (EVO_DIR / "distance_matrix_families.csv", EVO_DIR / "sdp_distance_heatmap_families.svg", "Top SDP C-alpha Distance Heatmap (Families)", "mako_r", "Distance (Å)", True),
        (EVO_DIR / "distance_matrix_subfamilies.csv", EVO_DIR / "sdp_distance_heatmap_subfamilies.svg", "Top SDP C-alpha Distance Heatmap (Subfamilies)", "mako_r", "Distance (Å)", True),
        (EVO_DIR / "coevolution_matrix.csv", EVO_DIR / "coevolution_matrix.svg", "Co-evolution Matrix", "viridis", None, False),
        (EVO_DIR / "coevolution_matrix_groups.csv", EVO_DIR / "coevolution_matrix_groups.svg", "Co-evolution Matrix (Groups)", "viridis", None, False),
        (EVO_DIR / "coevolution_matrix_families.csv", EVO_DIR / "coevolution_matrix_families.svg", "Co-evolution Matrix (Families)", "viridis", None, False),
        (EVO_DIR / "coevolution_matrix_subfamilies.csv", EVO_DIR / "coevolution_matrix_subfamilies.svg", "Co-evolution Matrix (Subfamilies)", "viridis", None, False),
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
