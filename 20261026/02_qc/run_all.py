"""Run the 20261026 QC steps in order, then execute the three notebooks.

    python run_all.py              # everything (~20 min; step 1 is ~10 of it)
    python run_all.py --from 2     # skip read assignment, reuse q01 outputs
    python run_all.py --no-notebooks

Env: karl_seq_analysis (spoa is called from the NGSpeciesID env by deep_consensus).
Each step writes outputs/q0N_*; a step only needs the steps above it to have run once.
"""
import argparse
import subprocess
import sys
import time

import config  # noqa: F401  (sets sys.path)
import q01_assign_reads, q02_crosstalk, q03_populations, q04_well_tables, q05_plate_reader

STEPS = [(1, "per-read assignment", q01_assign_reads),
         (2, "threshold + demux cross-talk", q02_crosstalk),
         (3, "consensus identity + unknown populations", q03_populations),
         (4, "well / strain / foreign tables", q04_well_tables),
         (5, "plate reader", q05_plate_reader)]
NOTEBOOKS = ["01_strain_identity_contamination.ipynb", "02_technical_replicates.ipynb",
             "03_24h_predicts_5d.ipynb"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--no-notebooks", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    for n, title, mod in STEPS:
        if n < a.start:
            continue
        print("\n" + "#" * 78 + f"\n# {n}. {title}\n" + "#" * 78, flush=True)
        mod.main()
    if not a.no_notebooks:
        for nb in NOTEBOOKS:
            print(f"\nexecuting {nb}", flush=True)
            subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                            "--execute", "--inplace", "--ExecutePreprocessor.timeout=1800",
                            str(config.BASE / nb)], check=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
