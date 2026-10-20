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
        "Check your executions."
    )
