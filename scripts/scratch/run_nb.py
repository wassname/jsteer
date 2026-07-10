"""Execute a notebook via nbclient (bypasses nbconvert's broken global config). (Claude)

    uv run python scripts/scratch/run_nb.py nbs/word_steering.ipynb /tmp/out.ipynb

Runs with cwd = the notebook's dir so its `sys.path.insert("..")` / `../artifacts`
relative paths resolve. Writes the executed copy (with outputs) to the out path.
"""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

nb_path = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
os.chdir(nb_path.parent)

nb = nbformat.read(nb_path, as_version=4)
NotebookClient(nb, timeout=2400, kernel_name="python3").execute()
nbformat.write(nb, out_path)
print(f"EXECUTED_OK -> {out_path}")
