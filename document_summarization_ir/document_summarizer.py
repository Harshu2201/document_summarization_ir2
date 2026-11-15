"""
Document Summarization System
--------------------------------
Final-year Computer Engineering mini-project demonstrating Information
Retrieval (IR) based extractive document summarization.

Core idea:
1. Read a TXT, PDF, or DOCX document.
2. Split the document into sentences.
3. Represent each sentence using TF-IDF.
4. Compare sentence TF-IDF vectors and rank connected sentences with TextRank.
5. Select relevant, non-redundant sentences and return about 35% in source order.

This is extractive summarization: the program does not generate new text.
It selects important sentences directly from the source document.
"""

from pathlib import Path
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}
TARGET_RATIO = 0.35


class SummarizationError(Exception):
    """Custom exception for expected summarization/input errors."""


def read_text_file(file_path: str) -> str:
    """Read a UTF-8 plain-text file.

    UTF-8 is used because it supports normal English text as well as many
    international characters. errors='replace' prevents a single malformed
    byte from crashing the complete prototype.
    """
    path = Path(file_path)

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SummarizationError(f"Could not read text file: {exc}") from exc


def read_pdf_file(file_path: str) -> str:
    """Extract text from every page of a PDF using pypdf.

    PDF files store layout information rather than clean paragraphs, so the
    extracted text may contain extra whitespace. That is acceptable for this
    educational prototype because the IR stage only needs usable text.
    """
    if PdfReader is None:
        raise SummarizationError(
            "PDF support requires pypdf. Install dependencies with: "
            "pip install -r requirements.txt"
        )

    try:
        reader = PdfReader(file_path)
        pages = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        return "\n".join(pages)

    except Exception as exc:
        raise SummarizationError(f"Could not read PDF file: {exc}") from exc


def read_docx_file(file_path: str) -> str:
    """Extract readable paragraph text from a Microsoft Word .docx file.

    The DOCX format is a ZIP/XML document rather than a plain-text file, so
    python-docx is used to read paragraph content. Tables are also included
    when they contain text, which makes the upload feature more useful for
    reports and notes.
    """
    if Document is None:
        raise SummarizationError(
            "DOCX support requires python-docx. Install dependencies with: "
            "pip install -r requirements.txt"
        )

    try:
        document = Document(file_path)
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(". ".join(cells))

        return "\n".join(parts)
    except Exception as exc:
        raise SummarizationError(f"Could not read DOCX file: {exc}") from exc


def load_document(file_path: str) -> str:
    """Load a supported document based on its file extension.

    This function is the input layer of the system. Keeping file handling
    separate from summarization makes the IR algorithm reusable.
    """
    path = Path(file_path)

    if not path.exists():
        raise SummarizationError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise SummarizationError(f"Path is not a file: {file_path}")

    extension = path.suffix.lower()

    if extension == ".txt":
        text = read_text_file(file_path)
    elif extension == ".pdf":
        text = read_pdf_file(file_path)
    elif extension == ".docx":
        text = read_docx_file(file_path)
    else:
        raise SummarizationError(
            f"Unsupported format '{extension}'. Supported formats: .txt, .pdf, .docx"
        )

    text = clean_text(text)

    if not text:
        raise SummarizationError("The document is empty or contains no readable text.")

    return text


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving sentence and paragraph breaks.

    We do not aggressively remove punctuation because '.', '?' and '!' help
    the simple sentence tokenizer identify sentence boundaries.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> List[List[str]]:
    """Split normalized text into paragraphs and sentences."""
    paragraphs = []
    for paragraph in clean_text(text).split("\n\n"):
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", paragraph
            )
            if sentence.strip()
        ]
        if sentences:
            paragraphs.append(sentences)
    return paragraphs


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight rule-based tokenizer.

    NLTK's Punkt tokenizer is excellent, but it normally needs an additional
    downloaded model. For a small offline student project, a regular
    expression keeps installation simple and avoids external downloads.

    This tokenizer is intentionally simple; abbreviations such as 'Dr.'
    can produce imperfect boundaries.
    """
    return [
        sentence
        for paragraph in _split_paragraphs(text)
        for sentence in paragraph
    ]


def _build_sentence_graph(matrix) -> List[List[Tuple[int, float]]]:
    """Build a sparse graph connecting each sentence to its closest peers."""
    sentence_count = matrix.shape[0]
    neighbors: List[List[Tuple[int, float]]] = [
        [] for _ in range(sentence_count)
    ]
    neighbor_limit = min(10, sentence_count - 1)

    for start in range(0, sentence_count, 32):
        stop = min(start + 32, sentence_count)
        # Compare small batches so long documents do not need one huge matrix.
        similarities = cosine_similarity(matrix[start:stop], matrix)

        for offset, row in enumerate(similarities):
            sentence_index = start + offset
            candidates = [
                (index, float(score))
                for index, score in enumerate(row)
                if index != sentence_index and score > 0
            ]
            neighbors[sentence_index] = sorted(
                candidates, key=lambda item: item[1], reverse=True
            )[:neighbor_limit]

    return neighbors


def _text_rank(neighbors: List[List[Tuple[int, float]]]) -> List[float]:
    """Calculate TextRank scores over a weighted sentence-similarity graph."""
    sentence_count = len(neighbors)
    if sentence_count == 0:
        return []

    weights = [sum(weight for _, weight in row) for row in neighbors]
    scores = [1 / sentence_count] * sentence_count

    for _ in range(100):
        # Sentences with no matching terms still keep a share of the rank score.
        dangling_score = sum(
            scores[index]
            for index, total_weight in enumerate(weights)
            if total_weight == 0
        )
        updated = [
            0.15 / sentence_count + 0.85 * dangling_score / sentence_count
            for _ in range(sentence_count)
        ]

        for index, row in enumerate(neighbors):
            if weights[index] == 0:
                continue
            contribution = 0.85 * scores[index] / weights[index]
            for neighbor_index, weight in row:
                updated[neighbor_index] += contribution * weight

        difference = sum(
            abs(updated[index] - scores[index])
            for index in range(sentence_count)
        )
        scores = updated
        if difference < 1e-6:
            break

    return scores


def rank_sentences(sentences: List[str]) -> List[Tuple[int, float]]:
    """Rank sentences by how strongly they connect to the document's ideas.

    TF-IDF vectors are connected by cosine similarity, and TextRank-style
    propagation gives higher scores to sentences connected to other relevant
    sentences. Similarity is calculated in small batches to avoid building a
    full dense sentence-by-sentence matrix.

    Returns:
        [(original_sentence_index, score), ...] sorted by descending score.
    """
    if len(sentences) < 2:
        return [(0, 1.0)] if sentences else []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'-]*\b",
        sublinear_tf=True,
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Happens when the document contains only stop words/punctuation.
        return [(index, 0.0) for index in range(len(sentences))]

    neighbors = _build_sentence_graph(matrix)
    scores = _text_rank(neighbors)
    return sorted(
        enumerate(scores), key=lambda item: item[1], reverse=True
    )


def summarize_text(text: str, ratio: float = TARGET_RATIO) -> str:
    """Generate an extractive summary from raw text.

    The default ratio is 35%, which satisfies the project's requested
    30-40% concise-summary range.

    For very short documents, at least one sentence is returned. For a
    two-sentence document, one sentence is normally enough; for larger
    documents, approximately 35% of the sentences are selected.
    """
    _validate_ratio(ratio)
    if not text or not text.strip():
        raise SummarizationError("Cannot summarize empty text.")

    paragraphs = _split_paragraphs(text)
    sentences = [
        sentence for paragraph in paragraphs for sentence in paragraph
    ]

    if not sentences:
        raise SummarizationError("No sentences could be identified.")

    if len(sentences) == 1:
        return sentences[0]

    # Round to the nearest sentence while guaranteeing at least one.
    summary_count = max(1, round(len(sentences) * ratio))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'-]*\b",
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        matrix = None

    if matrix is None:
        ranked = [(index, 0.0) for index in range(len(sentences))]
        neighbors = [[] for _ in sentences]
    else:
        neighbors = _build_sentence_graph(matrix)
        ranked = list(enumerate(_text_rank(neighbors)))

    relevance = {index: score for index, score in ranked}
    highest_score = max(relevance.values(), default=0.0)
    if highest_score:
        relevance = {
            index: score / highest_score for index, score in relevance.items()
        }

    selected: List[int] = []
    remaining = set(range(len(sentences)))

    while remaining and len(selected) < summary_count:
        best_index = -1
        best_score = float("-inf")
        for index in remaining:
            position_bonus = 0.15 / (index + 1) ** 0.5
            coherence_bonus = (
                0.08
                if selected
                and min(abs(index - selected_index) for selected_index in selected) <= 2
                else 0.0
            )
            base_score = (
                0.85 * relevance[index] + position_bonus + coherence_bonus
            )
            neighbor_scores = dict(neighbors[index])
            # Avoid selecting several sentences that express the same idea.
            redundancy = max(
                (neighbor_scores.get(selected_index, 0.0)
                 for selected_index in selected),
                default=0.0,
            )
            selection_score = base_score - 0.25 * redundancy
            if selection_score > best_score:
                best_index = index
                best_score = selection_score
        selected.append(best_index)
        remaining.remove(best_index)

    selected_indices = set(selected)
    summary_paragraphs = []
    sentence_index = 0
    # Keep the source order; ranking only decides which sentences are included.
    for paragraph in paragraphs:
        selected_paragraph = []
        for sentence in paragraph:
            if sentence_index in selected_indices:
                selected_paragraph.append(sentence)
            sentence_index += 1
        if selected_paragraph:
            summary_paragraphs.append(" ".join(selected_paragraph))

    return "\n\n".join(summary_paragraphs)


def summarize_file(file_path: str, ratio: float = TARGET_RATIO) -> Tuple[str, str]:
    """Load and summarize a supported file.

    Returns:
        (cleaned_original_text, summary)
    """
    _validate_ratio(ratio)
    text = load_document(file_path)

    # Very short input is not useful for aggressive summarization.
    # Returning the original text is more meaningful than deleting content.
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return text, text

    return text, summarize_text(text, ratio=ratio)


def _validate_ratio(ratio: float) -> None:
    if not 0.30 <= ratio <= 0.40:
        raise ValueError("Summary ratio must be between 0.30 and 0.40.")


def format_result(
    file_path: str, summary: str, ratio: float = TARGET_RATIO
) -> str:
    """Create a readable command-line result."""
    _validate_ratio(ratio)
    return (
        "\n" + "=" * 70 + "\n"
        "DOCUMENT SUMMARIZATION RESULT\n"
        + "=" * 70 + "\n"
        f"Input : {file_path}\n"
        f"Summary ratio : {ratio:.0%}\n\n"
        "SUMMARY\n"
        + "-" * 70 + "\n"
        + summary
        + "\n"
        + "=" * 70
    )


if __name__ == "__main__":
    # Small standalone test so the core module can be executed directly.
    demo = (
        "Information Retrieval is the process of finding useful information "
        "from a large collection of data. Search engines are a common example "
        "of information retrieval systems. TF-IDF is a classical technique "
        "used to estimate how important a term is in a document. It gives "
        "higher weight to terms that are important in one document but not "
        "common across all documents. Extractive summarization can use these "
        "weights to rank sentences. The highest-ranked sentences can then be "
        "combined into a shorter summary while keeping the original wording."
    )
    print(summarize_text(demo))
