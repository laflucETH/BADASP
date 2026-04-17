import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def resolve_fasttree_executable() -> str:
    if shutil.which("FastTreeMP") is not None:
        return "FastTreeMP"
    if shutil.which("FastTree") is not None:
        print(
            "WARNING: FastTreeMP not found on PATH. Falling back to single-threaded FastTree.",
            file=sys.stderr,
        )
        return "FastTree"
    raise FileNotFoundError("Neither FastTreeMP nor FastTree was found on PATH.")


def _build_fasttree_env(executable: str, threads: Optional[int] = None) -> dict:
    env = os.environ.copy()
    if executable == "FastTreeMP":
        cpu_count = os.cpu_count() or 1
        resolved_threads = cpu_count if threads is None else max(1, int(threads))
        env["OMP_NUM_THREADS"] = str(resolved_threads)
        env["OMP_DYNAMIC"] = "FALSE"
        env["OMP_PROC_BIND"] = "TRUE"
        print(
            f"Using FastTreeMP with OMP_NUM_THREADS={resolved_threads}",
            file=sys.stderr,
        )
    return env


def build_fasttree(trimmed_alignment: Path, tree_output: Path, threads: Optional[int] = None) -> Path:
    tree_output.parent.mkdir(parents=True, exist_ok=True)

    executable = resolve_fasttree_executable()
    env = _build_fasttree_env(executable=executable, threads=threads)
    cmd = [executable, str(trimmed_alignment)]
    with tree_output.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, env=env)

    return tree_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build maximum-likelihood tree using FastTree.")
    parser.add_argument("--input", default="data/interim/IPR019888_trimmed.aln")
    parser.add_argument("--output", default="data/interim/IPR019888.tree")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    tree_path = build_fasttree(
        trimmed_alignment=Path(args.input),
        tree_output=Path(args.output),
        threads=args.threads,
    )
    print(f"Tree written to: {tree_path}")


if __name__ == "__main__":
    main()
