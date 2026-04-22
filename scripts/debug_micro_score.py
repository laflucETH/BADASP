#!/usr/bin/env python3
"""Micro-debug: trace pair filtering without expensive tree ops."""

from pathlib import Path
import pandas as pd
from src.badasp_core import load_reconciliation_events

print("=" * 80)
print("MICRO-DEBUG: Inspecting Reconciliation Filter")
print("=" * 80)

# Load reconciliation events
recon = load_reconciliation_events(Path("results/reconciliation/duplication_nodes.csv"))
print(f"\nReconciliation events: {len(recon)}")

# Count event types
event_types = {}
for node, event_type in recon.items():
    event_types[event_type] = event_types.get(event_type, 0) + 1
print(f"Event type distribution: {event_types}")

# Load assignments to see what LCA labels are used
assign = pd.read_csv("results/topological_clustering/tree_cluster_assignments.csv")
print(f"\nAssignments: {assign.shape}")

# Check what LCA labels exist for groups
group_lca_col = "group_lca_node"
group_id_col = "group_id"

# Get unique group LCA labels
group_lca_labels = assign.groupby(group_id_col)[group_lca_col].first()
print(f"\nGroup clusters: {len(group_lca_labels)}")
print(f"Group LCA labels (first 5):")
for cid, lbl in group_lca_labels.head().items():
    print(f"  Cluster {cid}: {lbl}")

# Check if these labels appear in reconciliation
print(f"\nReconciliation coverage for Group LCA labels:")
found = 0
missing = 0
non_dup = 0
for cid, lbl in group_lca_labels.items():
    lbl_str = str(lbl).strip()
    if lbl_str in recon:
        event = recon[lbl_str]
        if event == "Duplication":
            found += 1
        else:
            non_dup += 1
            print(f"  {lbl_str} -> {event} (NOT duplication!)")
    else:
        print(f"  {lbl_str} -> MISSING from reconciliation")
        missing += 1

print(f"\nSummary: {found} Duplication, {non_dup} non-Duplication, {missing} missing")
print("=" * 80)
