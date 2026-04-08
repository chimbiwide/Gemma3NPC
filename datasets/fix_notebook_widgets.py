import json
import sys

if len(sys.argv) != 2:
    print("Usage: python fix_notebook_widgets.py <notebook.ipynb>")
    sys.exit(1)

path = sys.argv[1]

with open(path, "r") as f:
    nb = json.load(f)

if "widgets" in nb.get("metadata", {}):
    nb["metadata"].pop("widgets")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Removed metadata.widgets from {path}")
else:
    print(f"No metadata.widgets found in {path}, nothing to do.")
