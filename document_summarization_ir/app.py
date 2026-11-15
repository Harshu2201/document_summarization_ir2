"""Flask web interface for the IR Document Summarization System.

Features:
- Paste text directly into the browser.
- Upload TXT, PDF, or DOCX documents.
- Select a summary ratio between 30% and 40%.
- Reuses the same TF-IDF summarization core used by the CLI.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
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

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IR Summary | Document Summarizer</title>
<style>
    :root {
        color-scheme: light;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #172033;
        background: #f3f6fb;
        font-synthesis: none;
        text-rendering: optimizeLegibility;
        --ink: #172033;
        --muted: #64748b;
        --line: #dbe3ef;
        --accent: #3156d3;
        --accent-dark: #2343ad;
    }
    * { box-sizing: border-box; }
    body {
        min-width: 320px;
        min-height: 100vh;
        margin: 0;
        background:
            radial-gradient(ellipse at 12% 0%, rgba(80, 119, 238, .13), transparent 34rem),
            #f3f6fb;
    }
    .wrap { width: min(100% - 40px, 1120px); margin: 0 auto; padding-bottom: 48px; }
    .topbar {
        display: flex;
        min-height: 82px;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        border-bottom: 1px solid rgba(203, 214, 231, .75);
    }
    .brand { display: inline-flex; align-items: center; gap: 11px; color: #172b55; font-size: .96rem; font-weight: 800; text-decoration: none; }
    .brand-mark {
        display: grid;
        width: 38px;
        height: 38px;
        place-items: center;
        border-radius: 12px;
        background: #3156d3;
        box-shadow: 0 6px 14px rgba(49, 86, 211, .22);
        color: #fff;
        font-size: .86rem;
        letter-spacing: -.04em;
    }
    .top-link { color: #52627e; font-size: .9rem; font-weight: 650; text-decoration: none; }
    .top-link:hover { color: var(--accent); }
    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(210px, .55fr);
        align-items: end;
        gap: 36px;
        padding: 68px 0 38px;
    }
    .hero-copy { max-width: 700px; }
    .eyebrow {
        margin: 0 0 10px;
        color: var(--accent);
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    h1 { margin: 0; color: #102044; font-size: clamp(2rem, 5vw, 3.25rem); letter-spacing: -.045em; line-height: 1.08; }
    h2 { margin: 0 0 12px; color: #102044; font-size: 1.05rem; }
    .sub { max-width: 670px; margin: 14px 0 0; color: var(--muted); font-size: 1.05rem; line-height: 1.7; }
    .hero-note {
        padding: 18px 20px;
        border: 1px solid #dfe7f7;
        border-radius: 16px;
        background: rgba(255, 255, 255, .76);
        color: #455675;
        font-size: .9rem;
        line-height: 1.65;
    }
    .hero-note strong { display: block; margin-bottom: 4px; color: #1d3158; }
    .card {
        margin-bottom: 20px;
        padding: clamp(22px, 4vw, 36px);
        border: 1px solid rgba(219, 227, 239, .9);
        border-radius: 20px;
        background: rgba(255, 255, 255, .96);
        box-shadow: 0 18px 50px rgba(25, 45, 85, .08);
    }
    .form-card { padding: clamp(24px, 4vw, 40px); }
    .section-heading { margin-bottom: 24px; }
    .section-heading h2 { margin-bottom: 6px; font-size: 1.25rem; }
    .section-heading p { margin: 0; color: var(--muted); font-size: .92rem; line-height: 1.6; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .field { min-width: 0; }
    label { display: block; margin-bottom: 9px; color: #233454; font-size: .92rem; font-weight: 700; }
    textarea, input[type=file], select {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: #fff;
        color: var(--ink);
        font: inherit;
        transition: border-color .18s ease, box-shadow .18s ease;
    }
    textarea { min-height: 250px; margin-top: 20px; padding: 16px; line-height: 1.65; resize: vertical; }
    input[type=file], select { min-height: 50px; padding: 12px; }
    input[type=file] { border-style: dashed; background: #fbfcff; }
    input[type=file]::file-selector-button {
        margin-right: 12px;
        padding: 8px 11px;
        border: 0;
        border-radius: 8px;
        background: #edf2ff;
        color: #2747b5;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
    }
    .paste-label { margin-top: 24px; }
    textarea:focus, input[type=file]:focus-visible, select:focus-visible, button:focus-visible {
        outline: 3px solid rgba(49, 86, 211, .2);
        outline-offset: 2px;
        border-color: var(--accent);
    }
    .hint { margin: 9px 0 0; color: var(--muted); font-size: .82rem; line-height: 1.5; }
    .divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 23px 0 0;
        color: #8793a8;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .divider::before, .divider::after { height: 1px; flex: 1; background: #e8edf5; content: ""; }
    button {
        display: inline-flex;
        min-height: 50px;
        align-items: center;
        justify-content: center;
        margin-top: 16px;
        padding: 0 22px;
        border: 0;
        border-radius: 12px;
        background: var(--accent);
        box-shadow: 0 8px 18px rgba(49, 86, 211, .2);
        color: #fff;
        font: inherit;
        font-weight: 750;
        cursor: pointer;
        transition: background .18s ease, transform .18s ease;
    }
    button:hover { background: var(--accent-dark); transform: translateY(-1px); }
    .button-arrow { margin-left: 10px; font-size: 1.1rem; }
    .trust-row { display: flex; flex-wrap: wrap; gap: 10px 20px; margin: 18px 0 0; color: #64748b; font-size: .8rem; }
    .trust-row span::before { margin-right: 7px; color: #17865c; content: "✓"; font-weight: 800; }
    .steps { padding: 26px clamp(22px, 4vw, 36px); }
    .steps h2 { margin-bottom: 20px; }
    .step-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
    .step { display: flex; align-items: flex-start; gap: 12px; }
    .step-number {
        display: grid;
        width: 32px;
        height: 32px;
        flex: 0 0 32px;
        place-items: center;
        border-radius: 10px;
        background: #edf2ff;
        color: #3156d3;
        font-size: .82rem;
        font-weight: 800;
    }
    .step strong { display: block; margin: 2px 0 4px; color: #263757; font-size: .88rem; }
    .step p { margin: 0; color: var(--muted); font-size: .8rem; line-height: 1.55; }
    .result {
        padding: 22px;
        border: 1px solid #dce5f6;
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        background: #f7f9fe;
        color: #233454;
        font-size: 1rem;
        line-height: 1.85;
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
    .error { border-color: #fecdca; background: #fff6f5; color: #912018; }
    .error p { margin: 0; line-height: 1.6; }
    .meta { margin: -2px 0 16px; color: var(--muted); font-size: .88rem; overflow-wrap: anywhere; }
    .result-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .result-badge { padding: 6px 10px; border-radius: 999px; background: #edf8f2; color: #18764f; font-size: .76rem; font-weight: 750; white-space: nowrap; }
    footer { padding: 8px 0 0; color: #8793a8; font-size: .78rem; text-align: center; }
    @media (max-width: 700px) {
        .wrap { width: min(100% - 28px, 1120px); }
        .topbar { min-height: 68px; }
        .hero { grid-template-columns: 1fr; gap: 20px; padding: 46px 0 28px; }
        .hero-note { max-width: 420px; }
        .grid { grid-template-columns: 1fr; gap: 18px; }
        textarea { min-height: 210px; }
        button { width: 100%; }
        .step-grid { grid-template-columns: 1fr; gap: 17px; }
        .result-heading { flex-direction: column; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
    }
</style>
</head>
<body>
<main class="wrap">
    <nav class="topbar" aria-label="Main navigation">
        <a class="brand" href="/" aria-label="IR Summary home">
            <span class="brand-mark" aria-hidden="true">IR</span>
            <span>IR Summary</span>
        </a>
        <a class="top-link" href="#how-it-works">How it works</a>
    </nav>
    <header class="hero">
        <div class="hero-copy">
            <p class="eyebrow">Information Retrieval · Extractive NLP</p>
            <h1>Make long documents easier to read.</h1>
            <p class="sub">Get a clear, concise summary by selecting the most relevant sentences from your document. Every sentence comes from your original text.</p>
        </div>
        <aside class="hero-note">
            <strong>Simple. Source-based. Offline-friendly.</strong>
            TF-IDF and TextRank find important ideas without generating new content or calling an AI service.
        </aside>
    </header>
    <section class="card form-card" aria-labelledby="input-heading">
        <div class="section-heading">
            <h2 id="input-heading">Start with your document</h2>
            <p>Upload a file or paste text below. Choose how concise you want the summary to be.</p>
        </div>
        <form method="post" enctype="multipart/form-data">
            <div class="grid">
                <div class="field">
                    <label for="document">Upload a document</label>
                    <input id="document" type="file" name="document" accept=".txt,.pdf,.docx">
                    <p class="hint">TXT, PDF, or DOCX · Maximum size: 20 MB</p>
                </div>
                <div class="field">
                    <label for="ratio">Summary length</label>
                    <select id="ratio" name="ratio">
                        <option value="0.30" {% if ratio == 0.30 %}selected{% endif %}>30%</option>
                        <option value="0.35" {% if ratio == 0.35 %}selected{% endif %}>35% (Recommended)</option>
                        <option value="0.40" {% if ratio == 0.40 %}selected{% endif %}>40%</option>
                    </select>
                    <p class="hint">Choose how many of the source sentences to keep.</p>
                </div>
            </div>

            <div class="divider" aria-hidden="true">or paste text</div>
            <label class="paste-label" for="text">Document text</label>
            <textarea id="text" name="text" placeholder="Paste a document here to get started...">{{ text }}</textarea>
            <button type="submit">Generate summary <span class="button-arrow" aria-hidden="true">→</span></button>
            <p class="trust-row"><span>Original sentences</span><span>No external AI service</span><span>TXT, PDF, DOCX</span></p>
        </form>
    </section>

    {% if error %}
    <section class="card error" role="alert"><p><strong>Unable to summarize:</strong> {{ error }}</p></section>
    {% endif %}

    {% if summary %}
    <section class="card" aria-live="polite" aria-labelledby="summary-heading">
        <div class="result-heading">
            <div>
                <h2 id="summary-heading">Your summary</h2>
                <p class="meta">Source: {{ source_name }} · Requested summary length: {{ ratio_percent }}%</p>
            </div>
            <span class="result-badge">Extractive summary</span>
        </div>
        <div class="result">{{ summary }}</div>
    </section>
    {% endif %}

    <section id="how-it-works" class="card steps" aria-labelledby="steps-heading">
        <h2 id="steps-heading">How it works</h2>
        <div class="step-grid">
            <div class="step">
                <span class="step-number">01</span>
                <div><strong>Add your document</strong><p>Upload a supported file or paste in your text.</p></div>
            </div>
            <div class="step">
                <span class="step-number">02</span>
                <div><strong>Find key sentences</strong><p>TF-IDF and TextRank score related sentences.</p></div>
            </div>
            <div class="step">
                <span class="step-number">03</span>
                <div><strong>Read the summary</strong><p>Important sentences are returned in source order.</p></div>
            </div>
        </div>
    </section>
    <footer>Built for an Information Retrieval mini-project · Summaries use the source document's wording.</footer>
</main>
</body>
</html>
"""


def allowed_upload(filename: str) -> bool:
    return bool(filename) and Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    summary = ""
    error = ""
    source = "Pasted text"
    ratio = 0.35

    if request.method == "POST":
        try:
            ratio = float(request.form.get("ratio", "0.35"))
            text = request.form.get("text", "")
            uploaded = request.files.get("document")

            if uploaded and uploaded.filename:
                if not allowed_upload(uploaded.filename):
                    raise SummarizationError(
                        "Unsupported upload. Please choose a TXT, PDF, or DOCX file."
                    )

                suffix = Path(uploaded.filename).suffix.lower()
                # The temporary folder removes the upload after summarization.
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir) / f"document{suffix}"
                    uploaded.save(temp_path)
                    _, summary = summarize_file(str(temp_path), ratio=ratio)
                    source = uploaded.filename
            elif text.strip():
                summary = summarize_text(text, ratio=ratio)
            else:
                raise SummarizationError("Upload a TXT/PDF/DOCX file or paste text first.")

        except (SummarizationError, ValueError) as exc:
            error = str(exc)

    return render_template_string(
        HTML,
        text=text,
        summary=summary,
        error=error,
        source_name=source,
        ratio=ratio,
        ratio_percent=int(ratio * 100),
    )


@app.errorhandler(413)
def too_large(_):
    return render_template_string(
        HTML,
        text="",
        summary="",
        error="The uploaded file is larger than the 20 MB limit.",
        source_name="",
        ratio=0.35,
        ratio_percent=35,
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False)
