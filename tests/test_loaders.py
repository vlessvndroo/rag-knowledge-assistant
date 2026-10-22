from rag.loaders import load_markdown

SAMPLE = """---
nodeTitle: Handle errors gracefully
url: 'https://docs.n8n.io/build/flow-logic/handle-errors-gracefully'
---

# Error handling <a href="#error-handling" id="error-handling"></a>

Set an [error workflow](../workflows.md) in settings.

{% hint style="info" %}
Check your executions.
{% endhint %}

{% include "https://app.gitbook.com/s/abc/~/reusable/xyz/" %}

![diagram](img.png)

<figure>
<img src="../.gitbook/assets/merge.png" alt="">
<figcaption>Merging data from two nodes</figcaption>
</figure>

Pass a <unix timestamp> value.
"""


def test_load_markdown_cleans_gitbook_syntax(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(SAMPLE, encoding="utf-8")
    doc = load_markdown(f)

    assert doc.title == "Handle errors gracefully"
    assert doc.url == "https://docs.n8n.io/build/flow-logic/handle-errors-gracefully"
    assert doc.text == (
        "# Error handling\n\n"
        "Set an error workflow in settings.\n\n"
        "Check your executions.\n\n"
        "Merging data from two nodes\n\n"
        "Pass a <unix timestamp> value."
    )


def test_index_page_url_is_derived_from_path(monkeypatch, tmp_path):
    import rag.loaders as loaders

    monkeypatch.setattr(loaders, "CORPUS_DIR", tmp_path)
    page = tmp_path / "build" / "work-with-data" / "README.md"
    page.parent.mkdir(parents=True)
    page.write_text("# Work with data\n\nIntro.", encoding="utf-8")

    assert loaders.load_markdown(page).url == "https://docs.n8n.io/build/work-with-data"
