"""Build the course notebooks from jupytext percent-format sources.

Usage
-----
    python build.py                 # convert every src_nb/*.py -> notebooks/*.ipynb (clean, no outputs)
    python build.py 03 --execute    # convert + execute notebook 03 (for testing), report errors
    python build.py --execute       # execute all

The sources are plain Python files with `# %% [markdown]` / `# %%` cell markers so
they diff nicely in git.  The generated .ipynb files in notebooks/ are what students open
in Google Colab.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import jupytext
import nbformat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

REPO = "AxelRolov/chemoinformatics_tutorials"
BRANCH = "main"


def colab_badge(nb_name: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/notebooks/{nb_name}"
    return (
        f'<a href="{url}" target="_parent">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )


def assign_stable_ids(nb) -> None:
    """Give every cell a deterministic id.

    nbformat >= 4.5 stores a per-cell "id" and generates a *random* one on every write, which would make
    `build.py` non-reproducible: rebuilding an unchanged source produced a diff in every cell, so the
    "notebooks are in sync with src_nb" check in CI could never pass.

    The id is derived from the cell's own content, so editing one cell does not renumber the others
    (an index-based scheme would rewrite every id below an inserted cell). Identical cells - two empty
    "YOUR CODE HERE" cells, say - get a numeric suffix to keep ids unique, as nbformat requires.
    """
    seen: dict[str, int] = {}
    for cell in nb.cells:
        digest = hashlib.sha256(
            (cell.cell_type + "\x00" + "".join(cell.source)).encode("utf-8")
        ).hexdigest()[:12]
        n = seen.get(digest, 0)
        seen[digest] = n + 1
        cell["id"] = digest if n == 0 else f"{digest}-{n}"


def convert(src: Path) -> Path:
    nb = jupytext.read(src)
    nb_name = src.with_suffix(".ipynb").name
    # Inject the Colab badge as the very first line of the first markdown cell
    first = nb.cells[0]
    if first.cell_type == "markdown" and "colab-badge" not in first.source:
        first.source = colab_badge(nb_name) + "\n\n" + first.source
    # Clean outputs / execution counts and set metadata
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
        c.metadata.pop("jupyter", None)
        c.metadata.pop("lines_to_next_cell", None)
    assign_stable_ids(nb)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True, "name": nb_name},
        "accelerator": "GPU" if src.stem[:2] in {"06", "07", "10", "11"} else "None",
    }
    nb.metadata.pop("jupytext", None)
    nb.nbformat, nb.nbformat_minor = 4, 5
    out = OUT / nb_name
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


HEADLESS_PRELUDE = """\
# Injected by build.py --execute. No front-end is attached, so ipywidgets cannot be interacted with, and a
# matplotlib figure drawn inside an interact() callback makes nbclient wait for its timeout. Replace interact
# by a function that calls the callback once with mid-range arguments, so the cell runs and its code is tested.
import ipywidgets as _ipw

def _headless_interact(f=None, **kw):
    def run(func):
        args = {}
        for k, v in kw.items():
            if isinstance(v, tuple) and len(v) >= 2:
                lo, hi = v[0], v[1]
                args[k] = lo + (hi - lo) // 2 if isinstance(lo, int) and isinstance(hi, int) else (lo + hi) / 2
            elif isinstance(v, (list, dict)):
                args[k] = list(v)[0]
            else:
                args[k] = v
        func(**args)
        return func
    return run(f) if f is not None else run

_ipw.interact = _headless_interact
"""


def execute(nb_path: Path, timeout: int = 1800) -> tuple[bool, str]:
    """Execute a notebook copy with nbclient; return (ok, message). Never writes outputs to repo."""
    from nbclient import NotebookClient

    nb = nbformat.read(nb_path, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(HEADLESS_PRELUDE))   # see HEADLESS_PRELUDE
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
        allow_errors=True,
    )
    t0 = time.time()
    client.execute()
    errors = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append(f"cell {i - 1}: {o.get('ename')}: {str(o.get('evalue'))[:300]}")   # i-1: skip the prelude
    executed = nb_path.parent / f".executed_{nb_path.name}"
    nbformat.write(nb, executed)  # kept locally for inspection (git-ignored)
    msg = f"{nb_path.name}: {len(errors)} error(s) in {time.time() - t0:.0f}s"
    if errors:
        msg += "\n  " + "\n  ".join(errors)
    return not errors, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="prefixes of notebooks to build (e.g. 00 03)")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    srcs = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))
    if args.which:
        srcs = [s for s in srcs if any(s.name.startswith(w) for w in args.which)]
    ok_all = True
    for s in srcs:
        out = convert(s)
        print(f"built {out.relative_to(ROOT)}")
        if args.execute:
            ok, msg = execute(out)
            ok_all &= ok
            print(msg)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
