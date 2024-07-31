"""
Code for ingesting documents from multiple sources such as the file system or web urls.
"""

import os
import fitz
from fitz import EmptyFileError
from glob import glob
from sqlmodel import SQLModel, Field, create_engine, Session, select, Column, JSON

class PDFDocument(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    text: dict = Field(default_factory=dict, sa_column=Column(JSON)) # JSON indexed by page number
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))
class FileSystemIngester:
    """
    Ingests documents from the file system into a database.

    Attributes:
        path (str): Path to the directory containing the documents.
        file_type (str): Type of files to ingest (default is 'pdf').
        engine (Engine): SQLAlchemy engine connected to the database.
        pipeline (PDFPipeline | None): Pipeline for processing PDF files.
    """
    def __init__(self, path: str, file_type: str = 'pdf', dburl: str = "sqlite:///:memory:"):
        self.path = path
        self.file_type = file_type
        self._setup_db(dburl)
        if file_type == 'pdf':
            self.pipeline = PDFPipeline(path)
        else:
            self.pipeline = None

    def _setup_db(self, dburl: str)->None:
        """
        Setup the database
        :param dburl: database url
        :return:
        """
        self.engine = create_engine(dburl)
        SQLModel.metadata.create_all(self.engine)
    def ingest(self):
        """
        Ingest documents from the file system
        :return: list of documents
        """
        with Session(self.engine) as session:
            for text, metadata in self.pipeline:
                doc = PDFDocument(text=text, meta=metadata)
                session.add(doc)
                session.commit()
                session.refresh(doc)




class PDFPipeline:
    """
    Pipeline for processing PDF files in a directory.

    Attributes:
        path (str): Path to the directory containing the PDF files.
    """
    def __init__(self, path: str):
        self.path = path

    def __iter__(self):
        """
        Iterates over the PDF files in the directory, extracting text and metadata.

        Yields:
            tuple: A tuple containing the text (dict of pages) and metadata (dict) of each PDF file.
        """
        for d in glob(os.path.join(self.path, '*.pdf')):
            try:
                doc = fitz.open(d)
                metadata = doc.metadata
            except EmptyFileError:
                continue
            n = doc.name
            text = {}
            for page_number, page in enumerate(doc):
                p_text = page.get_text()
                if not p_text:
                    continue
                text[page_number] = p_text

            yield text, metadata