import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from Bio import SeqIO


_OX_PATTERN = re.compile(r"\bOX=(\d+)\b")
_OS_PATTERN = re.compile(r"\bOS=([^=]+?)(?=\s[A-Z]{2}=|$)")


def extract_taxon_from_header(header: str) -> Optional[str]:
    ox_match = _OX_PATTERN.search(header)
    if ox_match:
        return ox_match.group(1)

    os_match = _OS_PATTERN.search(header)
    if os_match:
        return os_match.group(1).strip()

    return None


def load_alignment_taxa(alignment_path: Path) -> Dict[str, str]:
    taxa: Dict[str, str] = {}
    for record in SeqIO.parse(str(alignment_path), "fasta"):
        taxon = extract_taxon_from_header(record.description)
        if taxon:
            taxa[record.id] = taxon
    return taxa


def _resolve_leaf_taxids(leaf_taxa: Dict[str, str], ncbi: Any) -> Dict[str, int]:
    resolved: Dict[str, int] = {}

    unresolved_names: List[str] = []
    leaf_to_name: Dict[str, str] = {}
    for leaf, taxon in leaf_taxa.items():
        if taxon.isdigit():
            resolved[leaf] = int(taxon)
        else:
            unresolved_names.append(taxon)
            leaf_to_name[leaf] = taxon

    if unresolved_names:
        translation = ncbi.get_name_translator(sorted(set(unresolved_names)))
        for leaf, species_name in leaf_to_name.items():
            taxids = translation.get(species_name, [])
            if taxids:
                resolved[leaf] = int(taxids[0])

    return resolved


def classify_evolutionary_events(gene_tree: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    events = gene_tree.get_descendant_evol_events()

    for idx, event in enumerate(events, start=1):
        if hasattr(event, "node") and getattr(event, "node", None) is not None and getattr(event.node, "name", ""):
            node_name = event.node.name
        elif hasattr(event, "in_seqs") and getattr(event, "in_seqs", None):
            ancestor = gene_tree.get_common_ancestor(event.in_seqs)
            node_name = getattr(ancestor, "name", "") or f"Event_{idx}"
        else:
            node_name = f"Event_{idx}"

        event_type = "Duplication" if getattr(event, "etype", "") == "D" else "Speciation"
        rows.append({"node_name": str(node_name), "event_type": event_type})

    return rows


def run_reconciliation(
    gene_tree_path: Path,
    alignment_path: Path,
    output_csv: Path,
    ncbi: Optional[Any] = None,
    phylo_tree_factory: Optional[Callable[[str], Any]] = None,
) -> Dict[str, int]:
    if ncbi is None:
        from ete3 import NCBITaxa

        ncbi = NCBITaxa()

    if phylo_tree_factory is None:
        from ete3 import PhyloTree

        phylo_tree_factory = PhyloTree

    leaf_taxa = load_alignment_taxa(alignment_path)
    leaf_taxids = _resolve_leaf_taxids(leaf_taxa, ncbi)

    unique_taxids = sorted(set(leaf_taxids.values()))
    if unique_taxids:
        ncbi.get_topology(unique_taxids)

    gene_tree = phylo_tree_factory(str(gene_tree_path))
    for leaf in gene_tree.iter_leaves():
        taxid = leaf_taxids.get(leaf.name)
        if taxid is not None:
            leaf.species = str(taxid)

    rows = classify_evolutionary_events(gene_tree)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_name", "event_type"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["event_type"] for row in rows)
    return {"Duplication": counts.get("Duplication", 0), "Speciation": counts.get("Speciation", 0)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gene/species reconciliation and classify duplication/speciation nodes.")
    parser.add_argument("--gene-tree", default="results/topological_clustering/mad_rooted.tree")
    parser.add_argument("--alignment", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--output", default="results/reconciliation/duplication_nodes.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = run_reconciliation(
        gene_tree_path=Path(args.gene_tree),
        alignment_path=Path(args.alignment),
        output_csv=Path(args.output),
    )
    print(f"Found {counts['Duplication']} Duplications and {counts['Speciation']} Speciation events.")


if __name__ == "__main__":
    main()
