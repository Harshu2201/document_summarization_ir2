import io
from pathlib import Path
import tempfile
import unittest

from docx import Document

from app import app
from document_summarizer import (
    SummarizationError,
    format_result,
    load_document,
    split_sentences,
    summarize_file,
    summarize_text,
)


class SummarizationTests(unittest.TestCase):
    def test_summary_is_extractive_ordered_and_uses_requested_ratio(self):
        sentences = [
            "Information retrieval ranks documents for useful search results.",
            "A search engine compares a query with indexed documents.",
            "TF-IDF gives rare terms greater importance in a document.",
            "Sentence vectors represent the terms found in each sentence.",
            "Cosine similarity measures the distance between two vectors.",
            "TextRank uses sentence connections to estimate importance.",
            "The ranking step identifies central sentences in the document.",
            "Extractive summarization selects sentences from the source.",
            "Selected sentences are restored to their original order.",
            "The final summary preserves wording from the source document.",
        ]

        summary = summarize_text(" ".join(sentences), ratio=0.30)
        selected = split_sentences(summary)

        self.assertEqual(len(selected), 3)
        self.assertTrue(all(sentence in sentences for sentence in selected))
        self.assertEqual(
            selected,
            sorted(selected, key=sentences.index),
        )

    def test_rejects_invalid_ratio_for_short_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "short.txt"
            path.write_text("First sentence. Second sentence.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "between 0.30 and 0.40"):
                summarize_file(str(path), ratio=0.8)

    def test_single_sentence_summary_is_returned_unchanged(self):
        text = "TF-IDF supports extractive document summarization."
        self.assertEqual(summarize_text(text), text)

    def test_empty_and_unsupported_documents_report_errors(self):
        with self.assertRaises(SummarizationError):
            summarize_text("  ")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.csv"
            path.write_text("not a supported document", encoding="utf-8")
            with self.assertRaisesRegex(SummarizationError, "Unsupported format"):
                load_document(str(path))

    def test_docx_text_is_loaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.docx"
            document = Document()
            document.add_paragraph("TF-IDF ranks source sentences.")
            document.save(path)

            self.assertEqual(load_document(str(path)), "TF-IDF ranks source sentences.")

    def test_cli_result_reports_the_selected_ratio(self):
        result = format_result("report.txt", "Selected source sentence.", ratio=0.4)
        self.assertIn("Summary ratio : 40%", result)


class WebAppTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_home_page_has_accessible_form_and_responsive_styles(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="document"', response.data)
        self.assertIn(b'for="document"', response.data)
        self.assertIn(b"@media (max-width: 700px)", response.data)
        self.assertIn(b"prefers-reduced-motion", response.data)
        self.assertIn(b"How it works", response.data)
        self.assertIn(b"Make long documents easier to read.", response.data)

    def test_pasted_text_returns_an_extractive_summary(self):
        text = (
            "Information retrieval organizes search results. "
            "TF-IDF measures term importance. "
            "TextRank ranks related sentences. "
            "Extractive summaries retain original sentences. "
            "The output restores the source order."
        )
        response = self.client.post(
            "/",
            data={"text": text, "ratio": "0.40"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Requested summary length: 40%", response.data)
        self.assertIn(b"Your summary", response.data)
        self.assertNotIn(b"Unable to summarize", response.data)

    def test_invalid_ratio_is_shown_as_an_error(self):
        response = self.client.post(
            "/",
            data={"text": "One sentence. Another sentence.", "ratio": "0.8"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"between 0.30 and 0.40", response.data)

    def test_uploads_docx_documents(self):
        document = Document()
        document.add_paragraph(
            "Information retrieval ranks documents. "
            "TF-IDF weighs terms. TextRank ranks sentences. "
            "Extractive summaries select source content."
        )
        content = io.BytesIO()
        document.save(content)
        content.seek(0)

        response = self.client.post(
            "/",
            data={
                "ratio": "0.35",
                "document": (content, "report.docx"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Source: report.docx", response.data)
        self.assertIn(b"Your summary", response.data)

    def test_uploaded_filename_is_html_escaped(self):
        response = self.client.post(
            "/",
            data={
                "ratio": "0.35",
                "document": (io.BytesIO(b"Some plain text."), "<script>.txt"),
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"<script>", response.data)


if __name__ == "__main__":
    unittest.main()
