"""
Code for ingesting documents from multiple sources such as the file system or web urls.
"""

import os

import fitz
import loguru
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
    def __init__(self, path: str, file_type: str = 'pdf', dburl: str = "sqlite:///:memory:", chunk_size: int = 500, chunk_overlap: int = 80):
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
                if isinstance(text, list) and text and isinstance(text[0], ChunkInfo):
                    text_dict = {str(ci.page_number * 10000 + ci.chunk_index): ci.text for ci in text}
                elif isinstance(text, list):
                    text_dict = {str(i): chunk for i, chunk in enumerate(text)}
                else:
                    text_dict = text
                
                doc = PDFDocument(text=text_dict, meta=metadata)
                session.add(doc)
                session.commit()
                session.refresh(doc)




class TextSplitter:
    """
    Recursive text splitter that respects semantic boundaries.

    Tries splitting by paragraphs (``\\n\\n``), then lines (``\\n``),
    then sentences (``. ``), and finally by spaces — always keeping
    every chunk within *chunk_size* characters.
    """

    _SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]
    MAX_CHUNK_CHARS: int = 8000

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = min(chunk_size, self.MAX_CHUNK_CHARS)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        return self._recursive_split(text, self._SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]

        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)

        chunks: list[str] = []
        current = ""

        for i, part in enumerate(parts):
            candidate = (current + sep + part) if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if len(current) <= self.chunk_size:
                        chunks.append(current.strip())
                    else:
                        chunks.extend(self._recursive_split(current, remaining_seps))
                    current = ""

                if len(part) <= self.chunk_size:
                    current = part
                else:
                    chunks.extend(self._recursive_split(part, remaining_seps))

        if current and current.strip():
            if len(current) <= self.chunk_size:
                chunks.append(current.strip())
            else:
                chunks.extend(self._recursive_split(current, remaining_seps))

        return self._apply_overlap(chunks)

    def _hard_split(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped: list[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                tail = chunks[i - 1][-self.chunk_overlap:]
                chunk = tail + chunk
            chunk = chunk.strip()
            if len(chunk) > self.chunk_size:
                chunk = chunk[:self.chunk_size]
            if chunk:
                overlapped.append(chunk)
        return overlapped


logger = loguru.logger


class ChunkInfo:
    """Metadata for a single chunk produced by the pipeline."""

    __slots__ = ("text", "page_number", "chunk_index")

    def __init__(self, text: str, page_number: int, chunk_index: int):
        self.text = text
        self.page_number = page_number
        self.chunk_index = chunk_index


class PDFPipeline:
    """
    Pipeline for processing PDF files in a directory.

    When *chunk_size* > 0 the pipeline processes each page **individually**
    and yields a list of :class:`ChunkInfo` objects instead of a raw dict.
    This avoids loading the full document text into memory and preserves
    real page numbers for every chunk.

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
        for d in glob(os.path.join(self.path, '*.pdf')):
            try:
                doc = fitz.open(d)
                metadata = doc.metadata
            except (EmptyFileError, Exception) as e:
                logger.error(f"Error opening {d}: {e}")
                continue

            if self.splitter:
                chunks: list[ChunkInfo] = []
                for page_number, page in enumerate(doc):
                    page_text = page.get_text()
                    if not page_text or not page_text.strip():
                        continue
                    page_chunks = self.splitter.split_text(page_text)
                    for ci, chunk_text in enumerate(page_chunks):
                        chunks.append(
                            ChunkInfo(chunk_text, page_number, ci)
                        )
                yield chunks, metadata
            else:
                text: dict[int, str] = {}
                for page_number, page in enumerate(doc):
                    p_text = page.get_text()
                    if not p_text:
                        continue
                    text[page_number] = p_text
                yield text, metadata