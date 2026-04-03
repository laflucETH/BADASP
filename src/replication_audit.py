import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_CHECKPOINTS = [
    {
        "method": "Core BADASP scoring (RC*AC*p(AC))",
        "paper": "Edwards_Shields_2005",
        "track": "full",
        "required_files": [
            "results/full/badasp_scores.csv",
            "src/badasp_core.py",
        ],
    },
    {
        "method": "Core BADASP scoring (RC*AC*p(AC))",
        "paper": "Edwards_Shields_2005",
        "track": "pf13404",
        "required_files": [
            "results/pf13404/badasp_scores_pf13404.csv",
            "src/badasp_core.py",
        ],
    },
    {
        "method": "Threshold sweep (80/85/90/95)",
        "paper": "Bradley_Beltrao_2019",
        "track": "full",
        "required_files": [
            "results/full/full_threshold_sweep_summary.csv",
            "results/full/full_clades_80.csv",
            "results/full/full_clades_85.csv",
            "results/full/full_clades_90.csv",
            "results/full/full_clades_95.csv",
            "src/threshold_sweep.py",
        ],
    },
    {
        "method": "Threshold sweep (80/85/90/95)",
        "paper": "Bradley_Beltrao_2019",
        "track": "pf13404",
        "required_files": [
            "results/pf13404/pf13404_threshold_sweep_summary.csv",
            "results/pf13404/pf13404_clades_80.csv",
            "results/pf13404/pf13404_clades_85.csv",
            "results/pf13404/pf13404_clades_90.csv",
            "results/pf13404/pf13404_clades_95.csv",
            "src/threshold_sweep.py",
        ],
    },
    {
        "method": "Functional clade tree visualization",
        "paper": "Bradley_Beltrao_2019",
        "track": "full",
        "required_files": [
            "results/full/full_functional_clades_tree_80.pdf",
            "results/full/full_functional_clades_tree_85.pdf",
            "results/full/full_functional_clades_tree_90.pdf",
            "results/full/full_functional_clades_tree_95.pdf",
            "src/plot_clades.py",
        ],
    },
    {
        "method": "Functional clade tree visualization",
        "paper": "Bradley_Beltrao_2019",
        "track": "pf13404",
        "required_files": [
            "results/pf13404/pf13404_functional_clades_tree_80.pdf",
            "results/pf13404/pf13404_functional_clades_tree_85.pdf",
            "results/pf13404/pf13404_functional_clades_tree_90.pdf",
            "results/pf13404/pf13404_functional_clades_tree_95.pdf",
            "src/plot_clades.py",
        ],
    },
    {
        "method": "Structural clustering + enrichment tests",
        "paper": "Bradley_Beltrao_2019",
        "track": "full",
        "required_files": [
            "results/full/secondary_analysis_stats.txt",
            "results/full/structural_clustering_ks_test.png",
            "src/secondary_analysis.py",
        ],
    },
    {
        "method": "Structural mapping outputs",
        "paper": "Bradley_Beltrao_2019",
        "track": "full",
        "required_files": [
            "results/full/mapped_switches_publication.cxc",
            "results/full/badasp_switch_counts.png",
            "src/structural_mapping.py",
        ],
    },
    {
        "method": "Structural mapping outputs",
        "paper": "Bradley_Beltrao_2019",
        "track": "pf13404",
        "required_files": [
            "results/pf13404/mapped_switches_publication.cxc",
            "results/pf13404/badasp_switch_counts.png",
            "src/structural_mapping.py",
        ],
    },
]


TRACK_MANIFESTS = {
    "full": {
        "track": "full",
        "source_studies": [
            "Edwards and Shields 2005",
            "Bradley and Beltrao 2019",
        ],
        "inputs": {
            "tree": "data/interim/asr_output/IPR019888_aligned.fasta.treefile",
            "alignment": "data/interim/IPR019888_aligned.fasta",
            "state": "data/interim/asr_output/IPR019888_aligned.fasta.state",
            "scores": "results/full/badasp_scores.csv",
        },
        "thresholds": [0.80, 0.85, 0.90, 0.95],
        "results_dir": "results/full",
    },
    "pf13404": {
        "track": "pf13404",
        "source_studies": [
            "Edwards and Shields 2005",
            "Bradley and Beltrao 2019",
        ],
        "inputs": {
            "tree": "data/interim/asr_pf13404/IPR019888_pf13404_aligned.fasta.treefile",
            "alignment": "data/interim/IPR019888_pf13404_aligned.fasta",
            "state": "data/interim/asr_pf13404/IPR019888_pf13404_aligned.fasta.state",
            "scores": "results/pf13404/badasp_scores_pf13404.csv",
        },
        "thresholds": [0.80, 0.85, 0.90, 0.95],
        "results_dir": "results/pf13404",
    },
}


def _evaluate_checkpoint(checkpoint):
    required_files = checkpoint["required_files"]
    missing = [path for path in required_files if not Path(path).exists()]
    total = len(required_files)
    present = total - len(missing)
    fraction = present / total if total else 0.0
    return {
        **checkpoint,
        "present_files": present,
        "required_count": total,
        "completion_fraction": fraction,
        "status": "done" if not missing else "partial",
        "missing_files": ";".join(missing),
    }


def _write_markdown_summary(df, output_path):
    lines = [
        "# Replication Method Fidelity Matrix",
        "",
        "| Paper | Method | Track | Status | Present/Required |",
        "|---|---|---|---|---|",
    ]
    for _, row in df.sort_values(["paper", "method", "track"]).iterrows():
        lines.append(
            f"| {row['paper']} | {row['method']} | {row['track']} | {row['status']} | {row['present_files']}/{row['required_count']} |"
        )

    missing_rows = df[df["missing_files"].astype(bool)]
    if not missing_rows.empty:
        lines.extend(["", "## Missing Artifacts", ""])
        for _, row in missing_rows.iterrows():
            lines.append(f"- {row['paper']} / {row['method']} / {row['track']}: {row['missing_files']}")

    output_path.write_text("\n".join(lines))


def _plot_completion(df, output_path):
    grouped = (
        df.groupby(["paper", "track"], as_index=False)["completion_fraction"]
        .mean()
        .sort_values(["paper", "track"])
    )
    grouped["label"] = grouped["paper"] + "\n" + grouped["track"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grouped["label"], grouped["completion_fraction"], color="#2c7fb8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average completion fraction")
    ax.set_title("Replication Coverage by Paper and Track")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _write_track_indexes(df, output_dir):
    output_dir = Path(output_dir)
    for track in sorted(df["track"].unique()):
        track_dir = output_dir / track
        track_dir.mkdir(parents=True, exist_ok=True)

        track_rows = df[df["track"] == track].copy()
        index_rows = []
        for _, row in track_rows.iterrows():
            for artifact in row["required_files"]:
                index_rows.append(
                    {
                        "track": track,
                        "paper": row["paper"],
                        "method": row["method"],
                        "artifact_path": artifact,
                        "exists": Path(artifact).exists(),
                    }
                )

        index_df = pd.DataFrame(index_rows)
        csv_path = track_dir / "artifact_index.csv"
        md_path = track_dir / "artifact_index.md"
        index_df.to_csv(csv_path, index=False)

        lines = [
            f"# Artifact Index ({track})",
            "",
            "| Paper | Method | Artifact | Exists |",
            "|---|---|---|---|",
        ]
        for _, idx_row in index_df.iterrows():
            exists_label = "yes" if idx_row["exists"] else "no"
            lines.append(
                f"| {idx_row['paper']} | {idx_row['method']} | {idx_row['artifact_path']} | {exists_label} |"
            )
        md_path.write_text("\n".join(lines))

        print(f"Saved {track} artifact index to {csv_path}")


def run_audit(output_dir):
    output_dir = Path(output_dir)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    for track_name, manifest in TRACK_MANIFESTS.items():
        (manifests_dir / f"{track_name}_run_manifest.json").write_text(json.dumps(manifest, indent=2))

    rows = [_evaluate_checkpoint(cp) for cp in METHOD_CHECKPOINTS]
    df = pd.DataFrame(rows)

    csv_path = output_dir / "method_fidelity_matrix.csv"
    md_path = output_dir / "method_fidelity_matrix.md"
    plot_path = output_dir / "replication_coverage.svg"

    df.to_csv(csv_path, index=False)
    _write_markdown_summary(df, md_path)
    _plot_completion(df, plot_path)
    _write_track_indexes(df, output_dir)

    print(f"Saved matrix CSV to {csv_path}")
    print(f"Saved matrix Markdown to {md_path}")
    print(f"Saved coverage plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a replication method-fidelity matrix and run manifests.")
    parser.add_argument("--output-dir", default="results/replication", help="Output directory for replication audit artifacts")
    args = parser.parse_args()
    run_audit(args.output_dir)
