"""Turn GitBook-flavored markdown (n8n docs) into clean text plus metadata."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_GITBOOK_TAG = re.compile(r"^\s*\{%.*?%\}\s*$", re.MULTILINE)  # {% hint %}, {% include %}, {% tabs %}...
_HTML_ANCHOR = re.compile(r'\s*<a href="#[^"]*" id="[^"]*"></a>')
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_IMG = re.compile(r"<img\b[^>]*>")
# Strip only known HTML tags (keeping their inner text): a generic <...> pattern would
# also eat placeholders written in prose, such as "<unix timestamp>".
_HTML_TAG = re.compile(
    r"</?(?:a|b|br|code|div|em|figcaption|figure|i|li|ol|p|span|strong|table|tbody|td|th|thead|tr|ul)\b[^>]*>"
)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BLANK_LINES = re.compile(r"\n\s*\n\s*\n+")

CORPUS_DIR = Path("data/raw/n8n")
DOCS_BASE_URL = "https://docs.n8n.io"


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
    text = _HTML_IMG.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _LINK.sub(r"\1", text)  # keep link text, drop URLs: they add tokens, not meaning
    text = _BLANK_LINES.sub("\n\n", text).strip()

    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = meta.get("nodeTitle") or (title_match.group(1) if title_match else path.stem)
    url = meta.get("url") or _derive_url(path)
    return Document(source=path.as_posix(), title=title, url=url, text=text)


def _derive_url(path: Path) -> str | None:
    # Section index pages (README.md) carry no url in their front matter;
    # their public URL mirrors the folder path, as for every other page.
    try:
        rel = path.resolve().relative_to(CORPUS_DIR.resolve())
    except ValueError:
        return None
    parts = rel.with_suffix("").parts
    if parts[-1] == "README":
        parts = parts[:-1]
    return f"{DOCS_BASE_URL}/{'/'.join(parts)}"
