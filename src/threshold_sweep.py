import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from Bio import Phylo


def _parse_thresholds(raw_thresholds):
    return [float(value) for value in raw_thresholds.split(",") if value.strip()]


def _format_threshold_label(threshold):
    return str(int(round(threshold * 100)))


def assign_clades_for_threshold(tree, scores, threshold, prefix="BADASP_Clade_"):
    score_cutoff = scores["BADASP_Score"].quantile(threshold)
    switch_nodes = set(scores.loc[scores["BADASP_Score"] >= score_cutoff, "Node"])

    clade_counter = [1]
    leaf_assignments = []

    def traverse(node, current_clade_id):
        assigned_clade = current_clade_id
        if node.name in switch_nodes:
            clade_counter[0] += 1
            assigned_clade = clade_counter[0]

        if node.is_terminal():
            leaf_assignments.append(
                {
                    "Sequence_ID": node.name,
                    "Functional_Clade_ID": f"{prefix}{assigned_clade}",
                }
            )
            return

        for child in node.clades:
            traverse(child, assigned_clade)

    traverse(tree.root, 1)

    assignments = pd.DataFrame(leaf_assignments)
    clade_sizes = assignments["Functional_Clade_ID"].value_counts()

    summary = {
        "threshold": threshold,
        "score_cutoff": score_cutoff,
        "switch_nodes": len(switch_nodes),
        "clades": clade_counter[0],
        "tips": len(assignments),
        "root_tips": int((assignments["Functional_Clade_ID"] == f"{prefix}1").sum()),
        "nonroot_tips": int((assignments["Functional_Clade_ID"] != f"{prefix}1").sum()),
        "nonroot_tip_fraction": float((assignments["Functional_Clade_ID"] != f"{prefix}1").mean()),
        "multi_tip_clades": int((clade_sizes >= 2).sum()),
        "singleton_clades": int((clade_sizes == 1).sum()),
        "largest_clade_size": int(clade_sizes.iloc[0]),
        "largest_clade_fraction": float(clade_sizes.iloc[0] / len(assignments)),
        "median_clade_size": float(clade_sizes.median()),
    }

    return assignments, summary


def _derive_output_stem(scores_file, output_stem=None):
    if output_stem:
        return output_stem

    stem = Path(scores_file).stem
    if stem.startswith("badasp_scores_"):
        return stem.replace("badasp_scores_", "")
    if stem.startswith("badasp_scores"):
        return stem.replace("badasp_scores", "analysis").strip("_") or "analysis"
    return stem or "analysis"


def run_threshold_sweep(tree_file, scores_file, output_dir, thresholds, prefix="BADASP_Clade_", output_stem=None):
    tree = Phylo.read(tree_file, "newick")
    scores = pd.read_csv(scores_file)
    output_stem = _derive_output_stem(scores_file, output_stem)

    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    for threshold in thresholds:
        assignments, summary = assign_clades_for_threshold(tree, scores, threshold, prefix=prefix)
        threshold_label = _format_threshold_label(threshold)
        output_csv = Path(output_dir) / f"{output_stem}_clades_{threshold_label}.csv"
        assignments.to_csv(output_csv, index=False)
        summary["mapping_csv"] = str(output_csv)
        summaries.append(summary)
        print(
            f"Threshold {threshold:.2f}: {summary['clades']} clades, "
            f"{summary['nonroot_tip_fraction']:.3f} non-root tip coverage, "
            f"saved {output_csv.name}"
        )

    summary_df = pd.DataFrame(summaries).sort_values("threshold")
    summary_csv = Path(output_dir) / f"{output_stem}_threshold_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(summary_df["threshold"], summary_df["clades"], marker="o", color="#1f77b4", label="Clades")
    ax1.set_xlabel("BADASP quantile threshold")
    ax1.set_ylabel("Number of clades", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(
        summary_df["threshold"],
        summary_df["nonroot_tip_fraction"],
        marker="s",
        color="#d62728",
        label="Non-root tip fraction",
    )
    ax2.set_ylabel("Fraction of tips beyond root clade", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.tight_layout()
    plot_path = Path(output_dir) / f"{output_stem}_threshold_sweep.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved sweep summary to {summary_csv}")
    print(f"Saved sweep plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep BADASP percentile thresholds and summarize clade coverage.")
    parser.add_argument("--tree", required=True, help="Newick tree with internal node labels")
    parser.add_argument("--scores", required=True, help="BADASP score CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for sweep outputs")
    parser.add_argument("--thresholds", default="0.80,0.85,0.90,0.95", help="Comma-separated quantile thresholds")
    parser.add_argument("--prefix", default="BADASP_Clade_", help="Prefix for clade IDs")
    parser.add_argument("--output-stem", default=None, help="Stem used for generated output filenames")

    args = parser.parse_args()
    run_threshold_sweep(
        args.tree,
        args.scores,
        args.output_dir,
        _parse_thresholds(args.thresholds),
        prefix=args.prefix,
        output_stem=args.output_stem,
    )