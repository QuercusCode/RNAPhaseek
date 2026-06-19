"""Path bootstrap: run any pipeline script FROM THE PROJECT ROOT, e.g.
   python scripts/training/run_v6_production.py
This puts the project root (for `Functions`, `Data/`, `model/`) and every script
subfolder (for inter-script imports like `from run_v5_final import ...`) on sys.path."""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "scripts/data_prep", "scripts/training", "scripts/generation",
             "scripts/analysis", "scripts/reporting", "scripts/legacy"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
