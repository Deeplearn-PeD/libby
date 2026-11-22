"""
Code for ingesting documents from multiple sources such as the file system or web urls.
"""

import os
import re

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
        file_type (str): Type of files to ingest: an extension such as 'pdf'
            or 'docx' (default 'pdf'), or 'all' for every supported format.
        engine (Engine): SQLAlchemy engine connected to the database.
        pipeline (DocumentPipeline): Pipeline for processing the files.
    """
    def __init__(self, path: str, file_type: str = 'pdf', dburl: str = "sqlite:///:memory:", chunk_size: int = 500, chunk_overlap: int = 80):
        self.path = path
        self.file_type = file_type
        self._setup_db(dburl)
        if file_type == 'pdf':
            self.pipeline = PDFPipeline(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif file_type in ('all', '*'):
            self.pipeline = DocumentPipeline(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            ext = file_type if file_type.startswith('.') else f'.{file_type}'
            self.pipeline = DocumentPipeline(
                path, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                extensions=frozenset({ext.lower()}),
            )

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


# File extensions handled natively by PyMuPDF (page-accurate extraction).
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

# File extensions converted to Markdown by Microsoft's MarkItDown.
MARKITDOWN_EXTENSIONS: frozenset[str] = frozenset({
    # plain text & markup
    ".txt", ".text", ".md", ".markdown", ".rst",
    ".html", ".htm", ".xhtml",
    # office documents
    ".docx", ".pptx", ".xlsx", ".xls", ".msg",
    # structured data / misc
    ".csv", ".json", ".jsonl", ".xml", ".rss", ".atom", ".ipynb", ".epub",
})

SUPPORTED_EXTENSIONS: frozenset[str] = PDF_EXTENSIONS | MARKITDOWN_EXTENSIONS

# MarkItDown marks PowerPoint slide boundaries with this comment; we use it to
# map each slide to a "page" so chunks keep a meaningful location.
_SLIDE_MARKER = re.compile(r"^<!-- Slide number: (\d+) -->\s*$", re.MULTILINE)

_markitdown = None


def is_supported_file(path: str) -> bool:
    """Return True when *path* has an extension Libby can ingest."""
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def _get_markitdown():
    """Lazily build a shared MarkItDown converter (plugins disabled)."""
    global _markitdown
    if _markitdown is None:
        try:
            from markitdown import MarkItDown
        except ImportError as e:  # pragma: no cover - dependency is declared
            raise ImportError(
                "The 'markitdown' package is required to ingest non-PDF documents. "
                "Install it with: uv add 'markitdown[docx,pptx,xlsx,xls,outlook]'"
            ) from e
        _markitdown = MarkItDown(enable_plugins=False)
    return _markitdown


def _split_slides(markdown: str) -> dict[int, str]:
    """Split MarkItDown PPTX output into {slide_index: text} (0-based)."""
    pages: dict[int, str] = {}
    matches = list(_SLIDE_MARKER.finditer(markdown))
    if not matches:
        return {0: markdown}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        pages[int(m.group(1)) - 1] = markdown[m.end():end].strip()
    return pages


def extract_document(path: str) -> tuple[dict[int, str], dict]:
    """
    Extract the text of a single document, organised by page.

    PDFs are read with PyMuPDF so real page numbers are preserved. Every other
    supported format is converted to Markdown with MarkItDown; PowerPoint
    decks are split per slide, everything else is returned as page 0.

    :param path: path to the document
    :return: (pages, metadata) where pages maps page number -> text and
        metadata always contains ``source_name``, ``source_path`` and
        ``source_format`` (plus ``title`` when the document declares one)
    :raises ValueError: if the file extension is not supported
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext or path}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    pages: dict[int, str] = {}
    if ext in PDF_EXTENSIONS:
        doc = fitz.open(path)
        metadata = dict(doc.metadata)
        for page_number, page in enumerate(doc):
            p_text = page.get_text()
            if p_text:
                pages[page_number] = p_text
    else:
        result = _get_markitdown().convert(path)
        markdown = result.markdown or ""
        metadata = {"title": result.title or ""}
        pages = _split_slides(markdown) if ext == ".pptx" else {0: markdown}

    # Expose the source filename so callers can fall back to it when the
    # document's title metadata is empty (PyMuPDF returns '' for unset
    # titles, which defeats dict.get(key, default)).
    metadata.setdefault("source_name", os.path.splitext(os.path.basename(path))[0])
    metadata.setdefault("source_path", path)
    metadata["source_format"] = ext.lstrip(".")
    return pages, metadata


def extract_full_text(path: str) -> tuple[str, dict]:
    """Extract a document's complete text (pages joined by newlines)."""
    pages, metadata = extract_document(path)
    text = "\n".join(pages[k] for k in sorted(pages))
    return text, metadata


def list_supported_files(path: str, extensions: frozenset[str] | None = None) -> list[str]:
    """
    List the ingestible files directly inside *path* (non-recursive), sorted.
    If *path* is itself a file, it is returned when its extension is accepted.
    """
    exts = extensions if extensions is not None else SUPPORTED_EXTENSIONS
    if os.path.isfile(path):
        candidates = [path]
    else:
        candidates = glob(os.path.join(path, "*"))
    return sorted(
        f for f in candidates
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts
    )


class DocumentPipeline:
    """
    Pipeline for processing every supported document in a directory.

    PDFs are extracted with PyMuPDF; other formats (Word, PowerPoint, Excel,
    HTML, plain text, Markdown, CSV, JSON, EPUB, Outlook messages, ...) are
    converted to Markdown with MarkItDown.

    When *chunk_size* > 0 the pipeline processes each page **individually**
    and yields a list of :class:`ChunkInfo` objects instead of a raw dict.
    This avoids loading the full document text into memory and preserves
    real page numbers for every chunk.

    Attributes:
        path (str): Path to the directory containing the documents.
        extensions (frozenset[str]): File extensions to pick up.
        splitter (TextSplitter | None): Optional splitter to chunk the text.
    """

    def __init__(
        self,
        path: str,
        chunk_size: int = 0,
        chunk_overlap: int = 0,
        extensions: frozenset[str] | None = None,
    ):
        self.path = path
        self.extensions = extensions if extensions is not None else SUPPORTED_EXTENSIONS
        if chunk_size > 0:
            self.splitter = TextSplitter(chunk_size, chunk_overlap)
        else:
            self.splitter = None

    def __iter__(self):
        for d in list_supported_files(self.path, self.extensions):
            try:
                pages, metadata = extract_document(d)
            except (EmptyFileError, Exception) as e:
                logger.error(f"Error opening {d}: {e}")
                continue

            if self.splitter:
                chunks: list[ChunkInfo] = []
                for page_number in sorted(pages):
                    page_text = pages[page_number]
                    if not page_text or not page_text.strip():
                        continue
                    page_chunks = self.splitter.split_text(page_text)
                    for ci, chunk_text in enumerate(page_chunks):
                        chunks.append(ChunkInfo(chunk_text, page_number, ci))
                yield chunks, metadata
            else:
                yield pages, metadata


class PDFPipeline(DocumentPipeline):
    """
    Pipeline restricted to PDF files (kept for backward compatibility).
    See :class:`DocumentPipeline` for the multi-format version.
    """

    def __init__(self, path: str, chunk_size: int = 0, chunk_overlap: int = 0):
        super().__init__(path, chunk_size, chunk_overlap, extensions=PDF_EXTENSIONS)
