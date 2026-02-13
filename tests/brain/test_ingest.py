import pytest
import os
import fitz
from libbydbot.brain.ingest import FileSystemIngester, PDFPipeline, PDFDocument
from sqlmodel import Session, select


def test_initialization_default():
    ingester = FileSystemIngester(path="tests/test_corpus", file_type="pdf")
    assert ingester.path == "tests/test_corpus"
    assert ingester.file_type == "pdf"
    assert ingester.engine is not None

# def test_initialization_custom():
#     ingester = FileSystemIngester(path="test_corpus", file_type="txt", dburl="sqlite:///custom.db")
#     assert ingester.path == "test_path"
#     assert ingester.file_type == "txt"
#     assert ingester.engine.url.database == "custom.db"

def test_ingest():
    ingester = FileSystemIngester(path="tests/test_corpus", file_type="pdf")
    ingester.ingest()

    with Session(ingester.engine) as session:
        statement = select(PDFDocument)
        results = session.exec(statement).all()
        assert len(results) == 1
        assert isinstance(results[0].text, dict)
        assert isinstance( results[0].text['0'], str)

def test_pdf_pipeline():
    pdf_path = "tests/test_corpus"
    pipeline = PDFPipeline(path=str(pdf_path))
    documents = list(pipeline)
    assert len(documents) == 1
    text, metadata = documents[0]
    assert isinstance(text, dict)
    assert isinstance(metadata, dict)