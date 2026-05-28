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
        assert len(results) == 2
        for r in results:
            assert isinstance(r.text, dict)


def test_pdf_pipeline():
    pdf_path = "tests/test_corpus"
    pipeline = PDFPipeline(path=str(pdf_path))
    documents = list(pipeline)
    assert len(documents) == 2
    text, metadata = documents[0]
    assert isinstance(text, dict)
    assert isinstance(metadata, dict)


def test_pdf_pipeline_with_chunking():
    pipeline = PDFPipeline(path="tests/test_corpus", chunk_size=200, chunk_overlap=20)
    documents = list(pipeline)
    assert len(documents) == 2
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
