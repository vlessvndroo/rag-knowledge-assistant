"""Turn GitBook-flavored markdown (n8n docs) into clean text plus metadata."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_GITBOOK_TAG = re.compile(r"^\s*\{%.*?%\}\s*$", re.MULTILINE)  # {% hint %}, {% include %}, {% tabs %}...
_HTML_ANCHOR = re.compile(r'\s*<a href="#[^"]*" id="[^"]*"></a>')
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class Document:
    source: str
    title: str
    url: str | None
    text: str


def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    meta: dict = {}
    if m := _FRONT_MATTER.match(raw):
        meta = yaml.safe_load(m.group(1)) or {}
        raw = raw[m.end():]

    text = _GITBOOK_TAG.sub("", raw)
    text = _HTML_ANCHOR.sub("", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)  # keep link text, drop URLs: they add tokens, not meaning
    text = _BLANK_LINES.sub("\n\n", text).strip()

    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = meta.get("nodeTitle") or (title_match.group(1) if title_match else path.stem)
    return Document(source=path.as_posix(), title=title, url=meta.get("url"), text=text)
