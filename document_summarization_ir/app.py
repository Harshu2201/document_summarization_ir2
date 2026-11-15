"""Dark-themed Flask Web Interface for the Document Summarizer.

Styled to match the Digital Forensic Image Analysis Tool web UI.
Run with:
    python app.py
Then open http://127.0.0.1:5001 in Chrome.
"""

from pathlib import Path
import tempfile

from flask import Flask, render_template_string, request

from document_summarizer import (
    SUPPORTED_EXTENSIONS,
    SummarizationError,
    summarize_file,
    summarize_text,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Document Summarization Tool</title>
<style>
    * { box-sizing: border-box; }
    body { font-family: Segoe UI, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }
    .container { max-width: 1200px; margin: auto; }
    .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 14px; }
    h1 { margin: 0; color: #38bdf8; font-size: 26px; }
    .badge-top { background: #1e293b; color: #7dd3fc; border: 1px solid #334155; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: bold; }
    .panel { background: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px; }
    .panel h3 { margin-top: 0; color: #7dd3fc; font-size: 16px; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 16px; }
    label { display: block; font-size: 13px; color: #94a3b8; margin-bottom: 6px; font-weight: 600; }
    input[type="file"], select, textarea {
        width: 100%;
        background: #0f172a;
        color: #e2e8f0;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: inherit;
        font-size: 14px;
    }
    textarea { min-height: 190px; resize: vertical; line-height: 1.6; }
    input[type="file"]:focus, select:focus, textarea:focus { outline: none; border-color: #38bdf8; }
    .or-divider { text-align: center; color: #64748b; font-size: 12px; font-weight: bold; letter-spacing: 0.1em; margin: 14px 0; text-transform: uppercase; }
    button { background: #0284c7; color: white; border: none; padding: 11px 22px; border-radius: 6px; font-weight: bold; font-size: 14px; cursor: pointer; margin-top: 14px; }
    button:hover { background: #0369a1; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
    .card { background: #1e293b; padding: 16px; border-radius: 10px; border: 1px solid #334155; }
    .card span { display: block; font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .card strong { display: block; margin-top: 6px; font-size: 20px; color: #4ade80; }
    .text-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        max-height: 380px;
        overflow-y: auto;
        line-height: 1.75;
        font-size: 14px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .summary-highlight { border-left: 4px solid #38bdf8; }
    .caption { font-size: 12px; color: #94a3b8; margin-top: 8px; }
    .alert { background: #7f1d1d; color: #fecaca; border: 1px solid #991b1b; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px; }
    @media (max-width: 760px) {
        .grid-2 { grid-template-columns: 1fr; }
        .metrics { grid-template-columns: repeat(2, 1fr); }
    }
</style>
</head>
<body>
<div class="container">
    <div class="topbar">
        <h1>Document Summarization Tool</h1>
        <span class="badge-top">IR &amp; NLP Text Engine</span>
    </div>

    <form class="panel" method="POST" enctype="multipart/form-data">
        <h3>Input Document or Raw Text</h3>
        <div class="grid-2">
            <div>
                <label for="file">Upload Document ({{ supported_exts }})</label>
                <input id="file" type="file" name="file">
            </div>
            <div>
                <label for="ratio">Summary Compression Ratio</label>
                <select id="ratio" name="ratio">
                    <option value="0.2" {% if ratio == "0.2" %}selected{% endif %}>20% — Very Concise</option>
                    <option value="0.3" {% if ratio == "0.3" %}selected{% endif %}>30% — Balanced (Recommended)</option>
                    <option value="0.4" {% if ratio == "0.4" %}selected{% endif %}>40% — Detailed</option>
                    <option value="0.5" {% if ratio == "0.5" %}selected{% endif %}>50% — Half Length</option>
                </select>
            </div>
        </div>

        <div class="or-divider">— OR PASTE RAW TEXT DIRECTLY —</div>

        <div>
            <label for="text">Document Text</label>
            <textarea id="text" name="text" placeholder="Paste your article, report, or document text here...">{{ input_text }}</textarea>
        </div>

        <button type="submit">Generate Summary</button>
    </form>

    {% if error %}
    <div class="alert">{{ error }}</div>
    {% endif %}

    {% if summary %}
    <div class="metrics">
        <div class="card"><span>Source</span><strong>{{ source_label }}</strong></div>
        <div class="card"><span>Target Ratio</span><strong>{{ (ratio | float * 100) | int }}%</strong></div>
        <div class="card"><span>Original Words</span><strong>{{ orig_words }}</strong></div>
        <div class="card"><span>Summary Words</span><strong>{{ summary_words }} ({{ reduction }}% shorter)</strong></div>
    </div>

    <div class="grid-2">
        <div class="panel">
            <h3>Extracted Summary</h3>
            <div class="text-box summary-highlight">{{ summary }}</div>
            <p class="caption">Key sentences ranked and extracted from the source content.</p>
        </div>
        <div class="panel">
            <h3>Original Source Preview</h3>
            <div class="text-box">{{ preview_text }}</div>
            <p class="caption">Source material analyzed by the summarizer.</p>
        </div>
    </div>
    {% endif %}
</div>
</body>
</html>"""


def _extract_summary_text(result) -> tuple[str, str]:
    """Handle either string or structured dictionary/object returns from document_summarizer."""
    if isinstance(result, str):
        return result, ""
    if isinstance(result, dict):
        summary_str = result.get("summary") or result.get("summary_text") or str(result)
        orig_str = result.get("original_text") or result.get("text") or ""
        return str(summary_str), str(orig_str)
    summary_str = getattr(result, "summary", str(result))
    orig_str = getattr(result, "original_text", "")
    return str(summary_str), str(orig_str)


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    preview_text = ""
    input_text = ""
    error = ""
    source_label = "Raw Text"
    ratio = "0.3"
    orig_words = 0
    summary_words = 0
    reduction = 0

    supported_exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))

    if request.method == "POST":
        ratio = request.form.get("ratio", "0.3")
        input_text = request.form.get("text", "").strip()
        uploaded = request.files.get("file")

        try:
            ratio_val = float(ratio)

            if uploaded and uploaded.filename:
                safe_name = Path(uploaded.filename).name
                source_label = safe_name
                with tempfile.TemporaryDirectory() as tmp_dir:
                    temp_path = Path(tmp_dir) / safe_name
                    uploaded.save(temp_path)
                    raw_result = summarize_file(temp_path, ratio=ratio_val)
                    summary, extracted_orig = _extract_summary_text(raw_result)
                    if extracted_orig:
                        preview_text = extracted_orig
                    elif temp_path.suffix.lower() in {".txt", ".md", ".csv"}:
                        preview_text = temp_path.read_text(encoding="utf-8", errors="replace")
                    else:
                        preview_text = f"[Uploaded binary document: {safe_name}]"
            elif input_text:
                source_label = "Pasted Text"
                raw_result = summarize_text(input_text, ratio=ratio_val)
                summary, _ = _extract_summary_text(raw_result)
                preview_text = input_text
            else:
                error = "Please upload a supported document or paste some text first."

            if summary:
                summary_words = len(summary.split())
                orig_words = len(preview_text.split()) if not preview_text.startswith("[Uploaded binary") else 0
                if orig_words > 0:
                    reduction = max(0, round((1 - (summary_words / orig_words)) * 100))
                else:
                    orig_words = summary_words
                    reduction = 0

        except SummarizationError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Summarization failed: {exc}"

    return render_template_string(
        HTML,
        supported_exts=supported_exts,
        summary=summary,
        preview_text=preview_text,
        input_text=input_text,
        error=error,
        source_label=source_label,
        ratio=ratio,
        orig_words=orig_words,
        summary_words=summary_words,
        reduction=reduction,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Document Summarization Web Server Running!")
    print("Open in Chrome: http://127.0.0.1:5001")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5001, debug=False)
