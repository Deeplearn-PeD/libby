# import chromadb
import os
from glob import glob
from hashlib import sha256
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

import dotenv
import fitz
import loguru
import ollama
from google import genai
from google.genai import types
from fitz import EmptyFileError
from pgvector.sqlalchemy import Vector
import sqlite_vec
from sqlalchemy import (
    event,
    Column,
    Integer,
    String,
    Sequence,
    text,
    create_engine,
    select,
    Table,
    insert,
    func,
)
from sqlalchemy.exc import IntegrityError, NoSuchModuleError
from sqlalchemy.orm import DeclarativeBase, Session
import duckdb
from duckdb import array_type

dotenv.load_dotenv()
logger = loguru.logger


# engine = create_engine(os.getenv("PGURL"))
# with Session(engine) as session:
#     session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))


# create a class to store the embeddings
class Base(DeclarativeBase):
    pass


class Embedding(Base):
    # To use with Postgresql.
    __tablename__ = "embedding"
    __table_args__ = {"extend_existing": True}
    # id_seq = Sequence("id_seq", metadata=Base.metadata)
    id = Column(Integer, autoincrement=True, primary_key=True)
    collection_name = Column(String)
    doc_name = Column(String)
    page_number = Column(Integer)
    doc_hash = Column(String, unique=True)
    document = Column(String)
    embedding = Column(Vector(1024))


class DocEmbedder:
    def __init__(
        self,
        col_name,
        dburl: str = "",
        embedding_model: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.dburl = dburl if dburl else os.getenv("PGURL")
        if embedding_model is None:
            from libbydbot.settings import Settings

            settings = Settings()
            embedding_model = settings.default_embedding_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._connection = None

        # Configure Google AI if using Gemini model
        if "gemini" in embedding_model.lower():
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.client = ollama.Client(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )

        if self.dburl.startswith("sqlite"):
            self.table_name = "embedding_sqlite"
            self._create_sqlite_table(self.connection.cursor())
            self.engine = None
            self.embedding = None
        elif self.dburl.startswith("duckdb"):
            self.table_name = "embedding_duckdb"
            self.engine = None  # We'll use self._connection
            self.embedding = None
            # The connection property will handle _init_duckdb
            _ = self.connection
        else:
            # PostgreSQL
            try:
                logger.info(f"Connecting to database with dburl: {self.dburl}")
                self.engine = create_engine(self.dburl)
            except NoSuchModuleError as exc:
                logger.error(f"Invalid dburl string passed to DocEmbedder: \n{exc}")
                self.engine = create_engine("sqlite:///data/embedding.db")
            if self.dburl.lower().startswith("postgres"):
                self.embedding = Embedding

        self.collection_name = col_name

        # Check if tables exist and create them only if they don't exist
        if self._should_create_tables():
            if self.dburl.startswith("sqlite"):
                self._create_sqlite_table(self.connection.cursor())
            elif not self.dburl.startswith("duckdb"):
                # PostgreSQL
                Base.metadata.create_all(
                    self.engine,
                    tables=[Base.metadata.sorted_tables[0]],
                    checkfirst=True,
                )

    def _init_duckdb(self):
        """
        Initialize DuckDB: install vss extension, create table if not exists,
        and create HNSW index.
        """
        dimension = self._get_embedding_dimension()
        db_path = self.dburl.split(":///")[1] if ":///" in self.dburl else ":memory:"
        if db_path == ":memory:":
            conn = duckdb.connect(":memory:")
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = duckdb.connect(db_path)

        # Install and load vss extension
        conn.sql("INSTALL vss;")
        conn.sql("LOAD vss;")

        # Check if table exists
        result = conn.sql(
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.table_name}';"
        ).fetchall()
        # result is a list of tuples; we check if count > 0
        if result[0][0] == 0:
            # Create sequence for auto-incrementing ID
            conn.sql(f"CREATE SEQUENCE IF NOT EXISTS {self.table_name}_seq;")
            # Create table
            create_sql = f"""
            CREATE TABLE {self.table_name} (
                id INTEGER PRIMARY KEY DEFAULT nextval('{self.table_name}_seq'),
                collection_name TEXT,
                doc_name TEXT,
                page_number INTEGER,
                doc_hash TEXT UNIQUE,
                document TEXT,
                embedding FLOAT[{dimension}]
            );
            """
            conn.sql(create_sql)
            logger.info("Created DuckDB embedding table with vector support.")

            # Create FTS index
            conn.sql("INSTALL fts;")
            conn.sql("LOAD fts;")
            conn.sql(f"PRAGMA create_fts_index('{self.table_name}', 'id', 'document');")
            logger.info("Created DuckDB FTS index for hybrid search.")

            if ":memory:" in self.dburl:
                # Create HNSW index.
                conn.sql(f"""
                    CREATE INDEX embedding_duckdb_index 
                    ON {self.table_name} USING HNSW(embedding) 
                    WITH (metric='cosine');
                """)
                logger.info("Created HNSW index on embedding column.")
        return conn

    def _should_create_tables(self):
        """
        Check if the appropriate embedding table exists in the database
        """
        try:
            if self.dburl.startswith("sqlite"):
                with self.connection as conn:
                    cursor = conn.cursor()
                    result = cursor.execute(
                        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.table_name}'"
                    ).fetchone()
                return result is None
            elif self.dburl.startswith("duckdb"):
                conn = self.connection
                result = conn.sql(
                    f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.table_name}';"
                ).fetchall()
                # result is a list of tuples; we check if count == 0
                return result[0][0] == 0
            else:
                # PostgreSQL
                with Session(self.engine) as session:
                    result = session.execute(
                        text(
                            "SELECT tablename FROM pg_tables WHERE tablename='embedding';"
                        )
                    ).fetchone()
                    return result is None
        except Exception as e:
            logger.warning(
                f"Error checking if tables exist: {e}. Will attempt to create tables."
            )
            return True

    @property
    def connection(self):
        """
        Get database connection. For SQLite, returns the native connection.
        For DuckDB, returns the duckdb connection object.
        For PostgreSQL, returns the SQLAlchemy engine.
        """
        if self._connection:
            return self._connection

        if self.dburl.startswith("sqlite"):
            self._connection = self._get_sqlite_connection()
            return self._connection
        elif self.dburl.startswith("duckdb"):
            self._connection = self._init_duckdb()
            return self._connection
        else:  # PostgreSQL
            return self.engine

    def _get_sqlite_connection(self):
        dbpath = self.dburl.split(":///")[1] if ":///" in self.dburl else ":memory:"
        # Handle in-memory databases
        if dbpath == ":memory:":
            connection = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            connection = sqlite3.connect(dbpath, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection

    @property
    def embeddings_list(self):
        """Return list of available embedding table names."""
        if self.dburl.startswith("duckdb"):
            return ["embedding_duckdb"]
        elif self.dburl.startswith("sqlite"):
            return ["embedding_sqlite"]
        else:
            return ["embedding"]

    def _create_sqlite_table(self, cursor):
        """
        Create the embedding table in SQLite database using native SQL
        """
        dimension = self._get_embedding_dimension()

        # Create the table with all necessary columns
        create_table_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING vec0(
            collection_name TEXT,
            doc_name TEXT,
            page_number INTEGER,
            doc_hash TEXT,
            document TEXT,
            embedding float[{dimension}]
        );
        """
        # Create FTS5 table for keyword search
        create_fts_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}_fts USING fts5(
            document,
            doc_hash UNINDEXED,
            content='{self.table_name}',
            content_rowid='rowid'
        );
        """
        try:
            cursor.execute(create_table_sql)
            cursor.execute(create_fts_sql)
            logger.info(
                "Created SQLite embedding table and FTS5 table for hybrid search."
            )
        except Exception as e:
            logger.error(f"Failed to create SQLite tables: {e}")
            raise
        # conn.close()

    def _get_embedding_dimension(self):
        """
        Get the embedding dimension based on the model
        """
        if self.embedding_model == "gemini-embedding-001":
            return 1024  # 1536
        else:  # mxbai-embed-large and other Ollama models
            return 1024

    def _generate_embedding(self, text: str):
        """
        Generate embedding using the specified model
        """
        if self.embedding_model == "gemini-embedding-001":
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self._get_embedding_dimension(),
                    task_type="retrieval_document",
                ),
            )
            return result.embeddings[0].values
        else:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response["embedding"]

    def _check_vector_exists(self):
        """
        Check if the vector extension exists in the database
        """
        if self.dburl.startswith("postgresql"):
            with Session(self.engine) as session:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                session.commit()
        elif self.dburl.startswith("duckdb"):
            # Already handled in _setup_duckdb
            pass
        elif self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT sqlite_version(), vec_version()")
                    sqlite_version, vec_version = cursor.fetchone()
                    logger.info(
                        f"SQLite version: {sqlite_version}, vector extension version: {vec_version}"
                    )
                except Exception as e:
                    logger.error(f"Error checking vector extension: {e}")

    def _check_existing(self, hash: str):
        """
        Check if a document with this hash already exists in the database
        :param hash: SHA256 hash of the document
        :return:
        """
        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                if self._should_create_tables():
                    self._create_sqlite_table(cursor)
                q = f"SELECT doc_hash FROM {self.table_name} WHERE doc_hash=?"
                result = cursor.execute(q, (hash,)).fetchone()
            return result is not None
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            result = conn.sql(
                f"SELECT id FROM {self.table_name} WHERE doc_hash = ?", params=[hash]
            ).fetchall()
            return len(result) > 0
        else:
            # PostgreSQL
            with Session(self.engine) as session:
                statement = select(self.embedding).where(
                    self.embedding.doc_hash == hash
                )
                result = session.execute(statement).all()
            return len(result) > 0

    def embed_text(self, doctext: str, docname: str, page_number: int):
        """
        Embed a page of a document.
        :param doctext: page of a document
        :param docname: name of the document
        :param page_number: page number
        :return:
        """
        document_hash = sha256(doctext.encode()).hexdigest()
        if self._check_existing(document_hash):
            logger.info(
                f"Document {docname} page {page_number} already exists in the database, skipping."
            )
            return
        doctext = doctext.replace("\x00", "\ufffd")
        embedding = self._generate_embedding(doctext)

        if self.dburl.startswith("sqlite"):
            # Use native SQLite operations
            import struct

            embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

            with self.connection as conn:
                cursor = conn.cursor()
                if self._should_create_tables():
                    self._create_sqlite_table(cursor)
                try:
                    logger.info(
                        f"Inserting into {self.table_name}: {docname} page {page_number}"
                    )
                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_name} (collection_name, doc_name, page_number, doc_hash, document, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            self.collection_name,
                            docname,
                            page_number,
                            document_hash,
                            doctext,
                            embedding_bytes,
                        ),
                    )

                    # Update FTS table
                    cursor.execute(
                        f"INSERT INTO {self.table_name}_fts(rowid, document, doc_hash) VALUES (?, ?, ?)",
                        (cursor.lastrowid, doctext, document_hash),
                    )

                    conn.commit()
                except sqlite3.IntegrityError as e:
                    logger.warning(
                        f"Document {docname} page {page_number} already exists in the database: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error: {e} generated when attempting to embed the following text: \n{doctext}"
                    )
        elif self.dburl.startswith("duckdb"):
            # Use direct SQL for DuckDB
            dimension = self._get_embedding_dimension()
            # Convert embedding list to a string representation for DuckDB array literal
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            conn = self.connection
            try:
                # Use conn.sql() for direct DuckDB SQL execution
                conn.sql(
                    f"""
                    INSERT INTO {self.table_name} (doc_hash, doc_name, collection_name, page_number, document, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    params=[
                        document_hash,
                        docname,
                        self.collection_name,
                        page_number,
                        doctext,
                        embedding_str,
                    ],
                )
            except Exception as e:
                logger.warning(
                    f"Document {docname} page {page_number} may already exist or error: {e}"
                )
        else:
            # PostgreSQL
            with Session(self.engine) as session:
                doc_vector = self.embedding(
                    doc_hash=document_hash,
                    doc_name=docname,
                    collection_name=self.collection_name,
                    page_number=page_number,
                    document=doctext,
                    embedding=embedding,
                )
                try:
                    session.add(doc_vector)
                    session.commit()
                except IntegrityError as e:
                    session.rollback()
                    logger.warning(
                        f"Document {docname} page {page_number} already exists in the database: {e}"
                    )
                except ValueError as e:
                    logger.error(
                        f"Error: {e} generated when attempting to embed the following text: \n{doctext}"
                    )
                    session.rollback()

    def embed_path(self, corpus_path: str):
        """
        Embed all documents in a path using chunking.
        :param corpus_path:  path to a folder containing PDFs
        :return:
        """
        from libbydbot.brain.ingest import PDFPipeline

        pipeline = PDFPipeline(
            corpus_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        for text, metadata in pipeline:
            docname = metadata.get("title", "Unknown")
            if isinstance(text, list):
                # Text is already chunked
                for i, chunk in enumerate(text):
                    self.embed_text(chunk, docname, i)
            else:
                # Legacy page-based dict
                for page_number, page_text in text.items():
                    self.embed_text(page_text, docname, page_number)

    def retrieve_docs(self, query: str, collection: str = "", num_docs: int = 5) -> str:
        """
        Retrieve documents based on a query.
        :param query: query string
        :param collection: collection name
        :param num_docs: number of documents to retrieve
        :return: all documents as a string
        """
        query_embedding = self._generate_embedding(query)

        if self.dburl.startswith("sqlite"):
            # Use native SQLite operations
            import struct

            query_embedding_bytes = struct.pack(
                f"{len(query_embedding)}f", *query_embedding
            )
            with self.connection as conn:
                cursor = conn.cursor()

                # Vector Search
                if collection:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, document, distance FROM {self.table_name} WHERE collection_name = ? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                        (collection, query_embedding_bytes, num_docs * 2),
                    ).fetchall()
                else:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, document, distance FROM {self.table_name} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                        (query_embedding_bytes, num_docs * 2),
                    ).fetchall()

                # Keyword Search (FTS5)
                fts_results = cursor.execute(
                    f"SELECT doc_hash, document, rank FROM {self.table_name}_fts WHERE document MATCH ? ORDER BY rank LIMIT ?",
                    (query, num_docs * 2),
                ).fetchall()

                # Hybrid Search combining results using RRF (Reciprocal Rank Fusion)
                k = 60
                scores = {}
                docs = {}

                for rank, (d_hash, doc, dist) in enumerate(vector_results):
                    scores[d_hash] = scores.get(d_hash, 0) + 1.0 / (k + rank + 1)
                    docs[d_hash] = doc

                for rank, (d_hash, doc, rank_score) in enumerate(fts_results):
                    scores[d_hash] = scores.get(d_hash, 0) + 1.0 / (k + rank + 1)
                    docs[d_hash] = doc

                # Sort by score and take top num_docs
                sorted_ids = sorted(
                    scores.keys(), key=lambda x: scores[x], reverse=True
                )[:num_docs]
                pages = [docs[d_hash] for d_hash in sorted_ids]
        elif self.dburl.startswith("duckdb"):
            # Use direct SQL for DuckDB
            dimension = self._get_embedding_dimension()
            embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            conn = self.connection

            # Vector Search
            vector_results = conn.sql(f"""
                SELECT id, document, array_cosine_similarity(embedding, {embedding_str}::FLOAT[{dimension}]) as similarity
                FROM {self.table_name} 
                {f"WHERE collection_name = '{collection}'" if collection else ""}
                ORDER BY similarity DESC
                LIMIT {num_docs * 2}
            """).fetchall()

            # Keyword Search (FTS)
            try:
                conn.sql("LOAD fts;")
                fts_results = conn.sql(
                    f"""
                    SELECT id, document, fts_main_{self.table_name}.match_bm25(id, ?) as score
                    FROM {self.table_name}
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT {num_docs * 2}
                """,
                    params=[query],
                ).fetchall()
            except Exception as e:
                logger.warning(f"FTS search failed: {e}")
                fts_results = []

            # Hybrid Search merging
            k = 60
            scores = {}
            docs = {}

            for rank, (d_id, doc, sim) in enumerate(vector_results):
                scores[d_id] = scores.get(d_id, 0) + 1.0 / (k + rank + 1)
                docs[d_id] = doc

            for rank, (d_id, doc, fts_score) in enumerate(fts_results):
                scores[d_id] = scores.get(d_id, 0) + 1.0 / (k + rank + 1)
                docs[d_id] = doc

            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
                :num_docs
            ]
            pages = [docs[d_id] for d_id in sorted_ids]
        else:
            # PostgreSQL
            dimension = self._get_embedding_dimension()
            with Session(self.engine) as session:
                if collection:
                    statement = (
                        select(self.embedding.document)
                        .where(self.embedding.collection_name == collection)
                        .order_by(self.embedding.embedding.l2_distance(query_embedding))
                        .limit(num_docs)
                    )
                else:
                    statement = (
                        select(self.embedding.document)
                        .order_by(self.embedding.embedding.l2_distance(query_embedding))
                        .limit(num_docs)
                    )
                pages = session.scalars(statement)

        data = "\n".join(pages)
        return data

    def retrieve_docs_with_metadata(
        self, query: str, collection: str = "", num_docs: int = 5
    ) -> list[dict]:
        """
        Retrieve documents with metadata based on a query.
        :param query: query string
        :param collection: collection name
        :param num_docs: number of documents to retrieve
        :return: list of dicts with doc_name, page_number, content, and score
        """
        query_embedding = self._generate_embedding(query)
        results = []

        if self.dburl.startswith("sqlite"):
            import struct

            query_embedding_bytes = struct.pack(
                f"{len(query_embedding)}f", *query_embedding
            )
            with self.connection as conn:
                cursor = conn.cursor()

                # Vector Search with metadata
                if collection:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, doc_name, page_number, document, distance FROM {self.table_name} WHERE collection_name = ? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                        (collection, query_embedding_bytes, num_docs * 2),
                    ).fetchall()
                else:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, doc_name, page_number, document, distance FROM {self.table_name} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                        (query_embedding_bytes, num_docs * 2),
                    ).fetchall()

                # Keyword Search (FTS5) with join to get metadata
                fts_results = cursor.execute(
                    f"""
                    SELECT f.doc_hash, t.doc_name, t.page_number, f.document, f.rank 
                    FROM {self.table_name}_fts f 
                    JOIN {self.table_name} t ON f.doc_hash = t.doc_hash
                    WHERE f.document MATCH ? ORDER BY f.rank LIMIT ?
                    """,
                    (query, num_docs * 2),
                ).fetchall()

                # Hybrid Search combining results using RRF (Reciprocal Rank Fusion)
                k = 60
                scores = {}
                docs = {}

                for rank, (d_hash, doc_name, page_num, doc, dist) in enumerate(
                    vector_results
                ):
                    scores[d_hash] = scores.get(d_hash, 0) + 1.0 / (k + rank + 1)
                    docs[d_hash] = {
                        "doc_name": doc_name,
                        "page_number": page_num,
                        "content": doc,
                    }

                for rank, (d_hash, doc_name, page_num, doc, rank_score) in enumerate(
                    fts_results
                ):
                    scores[d_hash] = scores.get(d_hash, 0) + 1.0 / (k + rank + 1)
                    docs[d_hash] = {
                        "doc_name": doc_name,
                        "page_number": page_num,
                        "content": doc,
                    }

                sorted_ids = sorted(
                    scores.keys(), key=lambda x: scores[x], reverse=True
                )[:num_docs]
                for d_hash in sorted_ids:
                    results.append(
                        {
                            "doc_name": docs[d_hash]["doc_name"],
                            "page_number": docs[d_hash]["page_number"],
                            "content": docs[d_hash]["content"],
                            "score": scores[d_hash],
                        }
                    )

        elif self.dburl.startswith("duckdb"):
            dimension = self._get_embedding_dimension()
            embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            conn = self.connection

            # Vector Search with metadata
            vector_results = conn.sql(f"""
                SELECT id, doc_name, page_number, document, array_cosine_similarity(embedding, {embedding_str}::FLOAT[{dimension}]) as similarity
                FROM {self.table_name} 
                {f"WHERE collection_name = '{collection}'" if collection else ""}
                ORDER BY similarity DESC
                LIMIT {num_docs * 2}
            """).fetchall()

            # Keyword Search (FTS) with metadata
            try:
                conn.sql("LOAD fts;")
                fts_results = conn.sql(
                    f"""
                    SELECT id, doc_name, page_number, document, fts_main_{self.table_name}.match_bm25(id, ?) as score
                    FROM {self.table_name}
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT {num_docs * 2}
                """,
                    params=[query],
                ).fetchall()
            except Exception as e:
                logger.warning(f"FTS search failed: {e}")
                fts_results = []

            # Hybrid Search merging
            k = 60
            scores = {}
            docs = {}

            for rank, (d_id, doc_name, page_num, doc, sim) in enumerate(vector_results):
                scores[d_id] = scores.get(d_id, 0) + 1.0 / (k + rank + 1)
                docs[d_id] = {
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "content": doc,
                }

            for rank, (d_id, doc_name, page_num, doc, fts_score) in enumerate(
                fts_results
            ):
                scores[d_id] = scores.get(d_id, 0) + 1.0 / (k + rank + 1)
                docs[d_id] = {
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "content": doc,
                }

            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
                :num_docs
            ]
            for d_id in sorted_ids:
                results.append(
                    {
                        "doc_name": docs[d_id]["doc_name"],
                        "page_number": docs[d_id]["page_number"],
                        "content": docs[d_id]["content"],
                        "score": scores[d_id],
                    }
                )
        else:
            # PostgreSQL
            with Session(self.engine) as session:
                if collection:
                    statement = (
                        select(
                            self.embedding.doc_name,
                            self.embedding.page_number,
                            self.embedding.document,
                        )
                        .where(self.embedding.collection_name == collection)
                        .order_by(self.embedding.embedding.l2_distance(query_embedding))
                        .limit(num_docs)
                    )
                else:
                    statement = (
                        select(
                            self.embedding.doc_name,
                            self.embedding.page_number,
                            self.embedding.document,
                        )
                        .order_by(self.embedding.embedding.l2_distance(query_embedding))
                        .limit(num_docs)
                    )
                rows = session.execute(statement).fetchall()
                for rank, (doc_name, page_num, doc) in enumerate(rows):
                    results.append(
                        {
                            "doc_name": doc_name,
                            "page_number": page_num,
                            "content": doc,
                            "score": 1.0
                            / (rank + 1),  # Simple inverse rank for PostgreSQL
                        }
                    )

        return results

    def get_embedded_documents(self):
        """
        Get a list of all embedded documents.
        :return: List of tuples (doc_name, collection_name) for all embedded documents
        """
        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                self._create_sqlite_table(cursor)
                result = cursor.execute(
                    f"SELECT DISTINCT doc_name, collection_name FROM {self.table_name}"
                ).fetchall()
            return [(row[0], row[1]) for row in result]
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            result = conn.sql(
                f"SELECT DISTINCT doc_name, collection_name FROM {self.table_name}"
            ).fetchall()
            return [(row[0], row[1]) for row in result]
        else:
            # PostgreSQL
            with Session(self.engine) as session:
                statement = select(
                    self.embedding.doc_name, self.embedding.collection_name
                ).distinct()
                result = session.execute(statement).fetchall()
                return [(row[0], row[1]) for row in result]

    def __del__(self):
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
        if hasattr(self, "engine") and self.engine:
            try:
                self.engine.dispose()
            except:
                pass
