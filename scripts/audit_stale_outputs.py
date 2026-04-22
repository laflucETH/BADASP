import os
import glob
import time

ref_files = [
    "results/badasp_scoring/badasp_scores_groups.csv",
    "results/badasp_scoring/badasp_scores_families.csv",
    "results/badasp_scoring/badasp_scores_subfamilies.csv",
]
ref_mtimes = [os.path.getmtime(f) for f in ref_files if os.path.exists(f)]
ref_mtime = max(ref_mtimes) if ref_mtimes else 0

stale_files = []
for ext in ["*.csv", "*.svg"]:
    for f in glob.glob(f"results/evolutionary_analysis/{ext}"):
        if os.path.getmtime(f) < ref_mtime:
            stale_files.append(f)

print("Reference files:")
for f in ref_files:
    print(f"  - {f}")
print(f"Reference mtime (latest score file): {int(ref_mtime)}\n")

print(f"Stale files: {len(stale_files)}")
for f in stale_files:
    print(f"  - {f} | producer=src/evolutionary_analysis.py")

if not stale_files:
    print("Scripts to run: none")
else:
    print("Scripts to run:\n  - src/evolutionary_analysis.py")
