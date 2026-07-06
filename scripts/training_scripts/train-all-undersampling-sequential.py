"""
Run all undersampling training scripts sequentially.

Scripts are run one at a time, allowing each to use all available CPU cores
via n_jobs=-1 without contention.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "train-rus.py",
    "train-cnn.py",
    "train-tomek.py",
    "train-oss.py",
    "train-enn.py",
    "train-renn.py",
    "train-allknn.py",
    "train-ncr.py",
    "train-nm1.py",
    "train-nm2.py",
]

SCRIPTS_DIR = Path(__file__).resolve().parent / "train-us"

for i, script in enumerate(SCRIPTS, 1):
    print(f"[{i}/{len(SCRIPTS)}] Running {script} ...")
    result = subprocess.run([sys.executable, SCRIPTS_DIR / script])
    if result.returncode != 0:
        print(f"FAILED: {script} — stopping.")
        sys.exit(result.returncode)
    print(f"Done: {script}\n")

print("All scripts completed successfully.")
