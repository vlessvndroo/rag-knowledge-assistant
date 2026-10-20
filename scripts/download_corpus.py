"""Download a subset of the public n8n documentation as the RAG corpus.

The docs are pinned to a commit so results (and evals) are reproducible.
Files are not committed to this repo: n8n-docs is MIT + Commons Clause,
so the corpus is fetched from the source instead of redistributed.

Usage: python scripts/download_corpus.py
"""

import json
import re
import urllib.request
from pathlib import Path

REPO = "n8n-io/n8n-docs"
COMMIT = "026e74274fbc50c321496f38aa1c3138e71c9703"
OUT_DIR = Path("data/raw/n8n")

INCLUDE = re.compile(r"^docs/(get-started|build/(flow-logic|understand-workflows|work-with-data))/.*\.md$")
# Function-reference tables and index/billing pages retrieve poorly and add noise.
EXCLUDE = re.compile(r"SUMMARY\.md$|expression-reference/|learning-paths\.md$|use-gateway-credits\.md$")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def main() -> None:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    paths = [
        t["path"] for t in tree["tree"]
        if t["type"] == "blob" and INCLUDE.match(t["path"]) and not EXCLUDE.search(t["path"])
    ]
    for path in paths:
        target = OUT_DIR / Path(path).relative_to("docs")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"))
    print(f"Downloaded {len(paths)} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
