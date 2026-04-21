from __future__ import annotations

from pathlib import Path

import pandas as pd


LEVELS = ["groups", "families", "subfamilies"]
BASE = Path("results/badasp_scoring")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def audit_level(level: str) -> None:
    raw_path = BASE / f"raw_pairwise_{level}.csv"
    score_path = BASE / f"badasp_scores_{level}.csv"

    raw_df = _load_csv(raw_path)
    score_df = _load_csv(score_path)

    unique_pairs = int(raw_df["pair"].nunique()) if not raw_df.empty and "pair" in raw_df.columns else 0
    threshold = float(raw_df["score"].quantile(0.95)) if not raw_df.empty and "score" in raw_df.columns else 0.0
    max_abs_switch = int(score_df["switch_count"].abs().max()) if not score_df.empty and "switch_count" in score_df.columns else 0

    if score_df.empty or "switch_count" not in score_df.columns:
        surviving_top_sdp = 0
    else:
        top_switch = int(score_df["switch_count"].max())
        surviving_top_sdp = int((score_df["switch_count"] == top_switch).sum())

    active_positions = int((score_df["switch_count"] > 0).sum()) if "switch_count" in score_df.columns else 0

    print(f"[{level.upper()}]")
    print(f"  unique sister pairs compared: {unique_pairs}")
    print(f"  95th percentile BADASP threshold: {threshold:.6f}")
    print(f"  maximum absolute switch_count observed: {max_abs_switch}")
    print(f"  positions with switch_count > 0: {active_positions}")
    print(f"  positions surviving final Top SDP filter: {surviving_top_sdp}")
    print()


def main() -> None:
    for level in LEVELS:
        audit_level(level)


if __name__ == "__main__":
    main()
