import pytest
import os
import fitz
from libbydbot.brain.ingest import FileSystemIngester, PDFPipeline, PDFDocument, TextSplitter, ChunkInfo
from sqlmodel import Session, select


def test_initialization_default():
    ingester = FileSystemIngester(path="tests/test_corpus", file_type="pdf")
    assert ingester.path == "tests/test_corpus"
    assert ingester.file_type == "pdf"
    assert ingester.engine is not None


def test_ingest():
    ingester = FileSystemIngester(path="tests/test_corpus", file_type="pdf")
    ingester.ingest()

    with Session(ingester.engine) as session:
        statement = select(PDFDocument)
        results = session.exec(statement).all()
        assert len(results) == 1
        for r in results:
            assert isinstance(r.text, dict)


def test_pdf_pipeline():
    pdf_path = "tests/test_corpus"
    pipeline = PDFPipeline(path=str(pdf_path))
    documents = list(pipeline)
    assert len(documents) == 1
    text, metadata = documents[0]
    assert isinstance(text, dict)
    assert isinstance(metadata, dict)


def test_pdf_pipeline_with_chunking():
    pipeline = PDFPipeline(path="tests/test_corpus", chunk_size=200, chunk_overlap=20)
    documents = list(pipeline)
    assert len(documents) == 1
    chunks, metadata = documents[0]
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, ChunkInfo)
        assert isinstance(chunk.text, str)
        assert len(chunk.text) > 0
        assert isinstance(chunk.page_number, int)
        assert isinstance(chunk.chunk_index, int)


def test_pdf_pipeline_chunking_preserves_page_numbers():
    pipeline = PDFPipeline(path="tests/test_corpus", chunk_size=200, chunk_overlap=20)
    documents = list(pipeline)
    chunks, _ = documents[0]
    for chunk in chunks:
        assert chunk.page_number >= 0


class TestTextSplitter:
    def test_empty_text(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []

    def test_short_text_single_chunk(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        result = splitter.split_text("Hello world")
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_respects_chunk_size(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "word " * 200
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 50 + 2

    def test_splits_by_paragraph_first(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        para1 = "A" * 80
        para2 = "B" * 80
        text = para1 + "\n\n" + para2
        chunks = splitter.split_text(text)
        assert len(chunks) == 2
        assert "A" in chunks[0]
        assert "B" in chunks[1]

    def test_hard_split_for_very_long_word(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=0)
        text = "A" * 200
        chunks = splitter.split_text(text)
        assert len(chunks) >= 4
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_overlap_applied(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "word " * 100
        chunks = splitter.split_text(text)
        if len(chunks) > 1:
            for chunk in chunks:
                assert len(chunk) <= splitter.chunk_size

    def test_max_chunk_chars_cap(self):
        splitter = TextSplitter(chunk_size=999999, chunk_overlap=0)
        assert splitter.chunk_size == TextSplitter.MAX_CHUNK_CHARS

    def test_overlap_capped_to_half_chunk_size(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=999)
        assert splitter.chunk_overlap == 50

    def test_semantic_splitting_by_sentence(self):
        splitter = TextSplitter(chunk_size=60, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 60 + 5

    def test_all_chunks_within_max_chars(self):
        splitter = TextSplitter(chunk_size=800, chunk_overlap=100)
        long_text = "\n\n".join(["Paragraph " + "x" * 200 for _ in range(50)])
        chunks = splitter.split_text(long_text)
        for chunk in chunks:
            assert len(chunk) <= 800


# ---------------------------------------------------------------------------
# Multi-format ingestion (MarkItDown)
# ---------------------------------------------------------------------------
import shutil

from libbydbot.brain.ingest import (
    DocumentPipeline,
    SUPPORTED_EXTENSIONS,
    extract_document,
    extract_full_text,
    is_supported_file,
    list_supported_files,
)


@pytest.fixture
def mixed_corpus(tmp_path):
    """A directory with one document per interesting format."""
    from pptx import Presentation

    shutil.copy("tests/test_corpus/sample.pdf", tmp_path / "sample.pdf")
    (tmp_path / "page.html").write_text(
        "<html><head><title>Vector Borne Diseases</title></head>"
        "<body><h1>Dengue</h1><p>Transmitted by Aedes mosquitoes.</p>"
        "<script>var x = 1;</script></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("Plain text notes about malaria.", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Heading\n\nSome *markdown* body.", encoding="utf-8")
    (tmp_path / "data.csv").write_text("city,cases\nRio,10\nRecife,20\n", encoding="utf-8")

    prs = Presentation()
    for title in ("First slide", "Second slide"):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = f"Body of {title.lower()}"
    prs.save(tmp_path / "deck.pptx")

    (tmp_path / "binary.exe").write_bytes(b"\x00\x01")
    return tmp_path


def test_is_supported_file():
    assert is_supported_file("a.PDF")
    assert is_supported_file("dir/report.docx")
    assert is_supported_file("page.html")
    assert not is_supported_file("program.exe")
    assert not is_supported_file("no_extension")
    assert ".pptx" in SUPPORTED_EXTENSIONS


def test_list_supported_files_skips_unknown(mixed_corpus):
    names = {os.path.basename(f) for f in list_supported_files(str(mixed_corpus))}
    assert names == {"sample.pdf", "page.html", "notes.txt", "readme.md", "data.csv", "deck.pptx"}


def test_extract_html_uses_title_and_drops_scripts(mixed_corpus):
    pages, meta = extract_document(str(mixed_corpus / "page.html"))
    assert meta["title"] == "Vector Borne Diseases"
    assert meta["source_format"] == "html"
    assert meta["source_name"] == "page"
    assert "Dengue" in pages[0] and "Aedes" in pages[0]
    assert "var x" not in pages[0]


def test_extract_pptx_splits_slides_into_pages(mixed_corpus):
    pages, meta = extract_document(str(mixed_corpus / "deck.pptx"))
    assert sorted(pages) == [0, 1]
    assert "First slide" in pages[0]
    assert "Second slide" in pages[1]
    assert "Slide number" not in pages[0]


def test_extract_pdf_keeps_pymupdf_pages(mixed_corpus):
    pages, meta = extract_document(str(mixed_corpus / "sample.pdf"))
    assert meta["source_format"] == "pdf"
    assert len(pages) >= 1


def test_extract_unsupported_raises(mixed_corpus):
    with pytest.raises(ValueError):
        extract_document(str(mixed_corpus / "binary.exe"))


def test_extract_full_text_csv(mixed_corpus):
    text, meta = extract_full_text(str(mixed_corpus / "data.csv"))
    assert "Recife" in text
    assert meta["source_format"] == "csv"


def test_document_pipeline_handles_all_formats(mixed_corpus):
    pipeline = DocumentPipeline(str(mixed_corpus), chunk_size=200, chunk_overlap=20)
    docs = {meta["source_format"]: chunks for chunks, meta in pipeline}
    assert set(docs) == {"pdf", "html", "txt", "md", "csv", "pptx"}
    for chunks in docs.values():
        assert chunks and all(isinstance(c, ChunkInfo) and c.text for c in chunks)
    assert {c.page_number for c in docs["pptx"]} == {0, 1}


def test_pdf_pipeline_ignores_other_formats(mixed_corpus):
    docs = list(PDFPipeline(str(mixed_corpus)))
    assert len(docs) == 1
    assert docs[0][1]["source_format"] == "pdf"


def test_filesystem_ingester_all_formats(mixed_corpus):
    ingester = FileSystemIngester(path=str(mixed_corpus), file_type="all", chunk_size=200, chunk_overlap=20)
    ingester.ingest()
    with Session(ingester.engine) as session:
        assert len(session.exec(select(PDFDocument)).all()) == 6


def test_filesystem_ingester_single_extension(mixed_corpus):
    ingester = FileSystemIngester(path=str(mixed_corpus), file_type="html")
    ingester.ingest()
    with Session(ingester.engine) as session:
        assert len(session.exec(select(PDFDocument)).all()) == 1
