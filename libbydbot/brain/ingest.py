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
    def __init__(self, path: str, file_type: str = 'pdf', dburl: str = "sqlite:///:memory:", chunk_size: int = 800, chunk_overlap: int = 100):
        self.path = path
        self.file_type = file_type
        self._setup_db(dburl)
        if file_type == 'pdf':
            self.pipeline = PDFPipeline(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
                # If text is a list of chunks, we'll store it as a dict indexed by index for compatibility
                if isinstance(text, list):
                    text_dict = {str(i): chunk for i, chunk in enumerate(text)}
                else:
                    text_dict = text
                
                doc = PDFDocument(text=text_dict, meta=metadata)
                session.add(doc)
                session.commit()
                session.refresh(doc)




class TextSplitter:
    """
    Splits text into smaller chunks with overlap.
    """
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        chunks = []
        if not text:
            return chunks
        
        # Simple recursive-ish split by double newlines, then newlines, then spaces
        # For simplicity in this initial step, we'll do a basic character-based split with overlap
        # but trying to avoid breaking words if possible.
        
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            
            if end < text_len:
                # Try to find a whitespace to split at
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= text_len:
                break
                
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            # Ensure we progress
            if start <= (end - self.chunk_size // 2):
                 start = end - self.chunk_overlap
            
            # Simple check to avoid infinite loop if overlap is too large
            if start >= end:
                start = end
                
        return chunks


class PDFPipeline:
    """
    Pipeline for processing PDF files in a directory.

    Attributes:
        path (str): Path to the directory containing the PDF files.
        splitter (TextSplitter | None): Optional splitter to chunk the text.
    """
    def __init__(self, path: str, chunk_size: int = 0, chunk_overlap: int = 0):
        self.path = path
        if chunk_size > 0:
            self.splitter = TextSplitter(chunk_size, chunk_overlap)
        else:
            self.splitter = None

    def __iter__(self):
        """
        Iterates over the PDF files in the directory, extracting text and metadata.

        Yields:
            tuple: A tuple containing the text (dict or list of chunks) and metadata (dict) of each PDF file.
        """
        for d in glob(os.path.join(self.path, '*.pdf')):
            try:
                doc = fitz.open(d)
                metadata = doc.metadata
            except (EmptyFileError, Exception) as e:
                logger.error(f"Error opening {d}: {e}")
                continue
            
            n = doc.name
            if self.splitter:
                # If splitting is enabled, we'll yield chunks
                all_text = ""
                for page in doc:
                    all_text += page.get_text() + "\n"
                
                chunks = self.splitter.split_text(all_text)
                yield chunks, metadata
            else:
                # Legacy behavior: dict indexed by page number
                text = {}
                for page_number, page in enumerate(doc):
                    p_text = page.get_text()
                    if not p_text:
                        continue
                    text[page_number] = p_text

                yield text, metadata