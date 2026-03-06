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
    # id = Column(Integer, id_seq, server_default=id_seq.next_value(), primary_key=True)
    collection_name = Column(String)
    doc_name = Column(String)
    page_number = Column(Integer)
    doc_hash = Column(String, unique=True)
    document = Column(String)
    embedding_model = Column(String, default="embeddinggemma")
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
        self.dburl = (
            dburl
            if dburl
            else os.getenv(
                "EMBED_DB", "postgresql://libby:libby123@localhost:5432/libby"
            )
        )
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
                # Enable pgvector extension (defense in depth - should already be enabled by init script)
                logger.info("Ensuring pgvector extension is enabled...")
                try:
                    with Session(self.engine) as session:
                        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                        session.commit()

                        # Verify extension is loaded
                        result = session.execute(
                            text("SELECT 1 FROM pg_extension WHERE extname='vector'")
                        ).fetchone()

                        if result:
                            logger.success("pgvector extension is enabled and ready")
                        else:
                            logger.error("pgvector extension could not be enabled")
                            raise RuntimeError(
                                "pgvector extension is required but could not be enabled"
                            )
                except Exception as e:
                    logger.error(f"Failed to enable pgvector extension: {e}")
                    logger.error(
                        "Make sure you are using the pgvector/pgvector Docker image"
                    )
                    raise

                self.embedding = Embedding

        self.collection_name = col_name

        # Check if tables exist and create them only if they don't exist
        if self._should_create_tables():
            logger.info("Creating embedding tables...")
            if self.dburl.startswith("sqlite"):
                self._create_sqlite_table(self.connection.cursor())
                logger.success("SQLite embedding table created successfully")
            elif not self.dburl.startswith("duckdb"):
                # PostgreSQL
                logger.info(
                    "Creating PostgreSQL embedding table with pgvector support..."
                )
                Base.metadata.create_all(
                    self.engine,
                    tables=[Base.metadata.sorted_tables[0]],
                    checkfirst=True,
                )
                logger.success("PostgreSQL embedding table created successfully")
        else:
            logger.info("Embedding tables already exist, skipping creation")

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
                embedding_model TEXT DEFAULT 'embeddinggemma',
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
            embedding_model TEXT,
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
            return 1024  # Can be configured up to 1536
        elif self.embedding_model == "embeddinggemma":
            return 768
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
                        INSERT INTO {self.table_name} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            self.collection_name,
                            docname,
                            page_number,
                            document_hash,
                            doctext,
                            self.embedding_model,
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
                    INSERT INTO {self.table_name} (doc_hash, doc_name, collection_name, page_number, document, embedding_model, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    params=[
                        document_hash,
                        docname,
                        self.collection_name,
                        page_number,
                        doctext,
                        self.embedding_model,
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
                    embedding_model=self.embedding_model,
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

    def _migrate_add_embedding_model(self):
        """
        Add embedding_model column to existing tables if it doesn't exist.
        For SQLite, this requires recreating the table.
        """
        if self.dburl.startswith("sqlite"):
            self._migrate_sqlite_add_embedding_model()
        elif self.dburl.startswith("duckdb"):
            self._migrate_duckdb_add_embedding_model()
        else:
            self._migrate_postgres_add_embedding_model()

    def _migrate_sqlite_add_embedding_model(self):
        """
        Migrate SQLite table to add embedding_model column.
        SQLite virtual tables (vec0) don't support ALTER TABLE ADD COLUMN,
        so we need to recreate the table with data preserved.
        """
        with self.connection as conn:
            cursor = conn.cursor()

            # Check if embedding_model column exists
            result = cursor.execute(f"PRAGMA table_info({self.table_name})").fetchall()
            columns = [row[1] for row in result]

            if "embedding_model" in columns:
                logger.info("SQLite table already has embedding_model column")
                return

            logger.info("Migrating SQLite table to add embedding_model column...")

            # Create backup table name
            backup_table = f"{self.table_name}_backup"

            # Rename current table to backup
            cursor.execute(f"ALTER TABLE {self.table_name} RENAME TO {backup_table}")

            # Create new table with embedding_model column
            dimension = self._get_embedding_dimension()
            create_sql = f"""
            CREATE VIRTUAL TABLE {self.table_name} USING vec0(
                collection_name TEXT,
                doc_name TEXT,
                page_number INTEGER,
                doc_hash TEXT,
                document TEXT,
                embedding_model TEXT,
                embedding float[{dimension}]
            );
            """
            cursor.execute(create_sql)

            # Copy data from backup, setting embedding_model to 'unknown' for old records
            cursor.execute(f"""
                INSERT INTO {self.table_name}
                (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                SELECT collection_name, doc_name, page_number, doc_hash, document, 'unknown', embedding
                FROM {backup_table}
            """)

            # Drop backup table
            cursor.execute(f"DROP TABLE {backup_table}")

            conn.commit()
            logger.info("SQLite migration completed successfully")

    def _migrate_duckdb_add_embedding_model(self):
        """
        Migrate DuckDB table to add embedding_model column.
        """
        conn = self.connection

        # Check if column exists
        result = conn.sql(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{self.table_name}' AND column_name = 'embedding_model'
        """).fetchall()

        if len(result) > 0:
            logger.info("DuckDB table already has embedding_model column")
            return

        logger.info("Migrating DuckDB table to add embedding_model column...")
        conn.sql(
            f"ALTER TABLE {self.table_name} ADD COLUMN embedding_model TEXT DEFAULT 'unknown'"
        )
        logger.info("DuckDB migration completed successfully")

    def _migrate_postgres_add_embedding_model(self):
        """
        Migrate PostgreSQL table to add embedding_model column.
        """
        with Session(self.engine) as session:
            # Check if column exists
            result = session.execute(
                text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'embedding' AND column_name = 'embedding_model'
            """)
            ).fetchall()

            if len(result) > 0:
                logger.info("PostgreSQL table already has embedding_model column")
                return

            logger.info("Migrating PostgreSQL table to add embedding_model column...")
            session.execute(
                text(
                    "ALTER TABLE embedding ADD COLUMN embedding_model TEXT DEFAULT 'unknown'"
                )
            )
            session.commit()
            logger.info("PostgreSQL migration completed successfully")

    def reembed(
        self,
        collection_name: str = "",
        new_model: str | None = None,
        batch_size: int = 100,
    ) -> dict:
        """
        Re-embed all documents with a new embedding model.

        :param collection_name: Collection to re-embed (empty = all collections)
        :param new_model: New embedding model (None = use current default from settings)
        :param batch_size: Number of documents to process per batch
        :return: Stats dict with count of updated documents and any errors
        """
        from libbydbot.settings import Settings

        # Determine the new model
        if new_model is None:
            settings = Settings()
            new_model = settings.default_embedding_model

        old_model = self.embedding_model
        logger.info(f"Re-embedding documents from model '{old_model}' to '{new_model}'")

        # Update the embedding model
        self.embedding_model = new_model

        # Reconfigure client if needed
        if "gemini" in new_model.lower():
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.client = ollama.Client(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )

        stats = {
            "total": 0,
            "updated": 0,
            "errors": [],
            "old_model": old_model,
            "new_model": new_model,
        }

        if self.dburl.startswith("sqlite"):
            return self._reembed_sqlite(collection_name, batch_size, stats)
        elif self.dburl.startswith("duckdb"):
            return self._reembed_duckdb(collection_name, batch_size, stats)
        else:
            return self._reembed_postgres(collection_name, batch_size, stats)

    def _reembed_sqlite(
        self, collection_name: str, batch_size: int, stats: dict
    ) -> dict:
        """
        Re-embed documents in SQLite.
        For SQLite, we need to recreate the table due to vec0 limitations.
        """
        import struct

        logger.info("Re-embedding SQLite database (this requires table recreation)...")

        with self.connection as conn:
            cursor = conn.cursor()

            # Get all documents
            if collection_name:
                cursor.execute(
                    f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name} WHERE collection_name = ?",
                    (collection_name,),
                )
            else:
                cursor.execute(
                    f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name}"
                )

            documents = cursor.fetchall()
            stats["total"] = len(documents)

            if len(documents) == 0:
                logger.info("No documents to re-embed")
                return stats

            # Create backup table
            backup_table = f"{self.table_name}_backup_reembed"
            cursor.execute(f"DROP TABLE IF EXISTS {backup_table}")
            cursor.execute(f"ALTER TABLE {self.table_name} RENAME TO {backup_table}")

            # Recreate FTS backup
            fts_backup = f"{self.table_name}_fts_backup_reembed"
            cursor.execute(f"DROP TABLE IF EXISTS {fts_backup}")
            cursor.execute(f"ALTER TABLE {self.table_name}_fts RENAME TO {fts_backup}")

            # Create new tables
            dimension = self._get_embedding_dimension()
            create_table_sql = f"""
            CREATE VIRTUAL TABLE {self.table_name} USING vec0(
                collection_name TEXT,
                doc_name TEXT,
                page_number INTEGER,
                doc_hash TEXT,
                document TEXT,
                embedding_model TEXT,
                embedding float[{dimension}]
            );
            """
            create_fts_sql = f"""
            CREATE VIRTUAL TABLE {self.table_name}_fts USING fts5(
                document,
                doc_hash UNINDEXED,
                content='{self.table_name}',
                content_rowid='rowid'
            );
            """
            cursor.execute(create_table_sql)
            cursor.execute(create_fts_sql)

            # Process documents
            for i, (
                doc_id,
                col_name,
                doc_name,
                page_num,
                doc_hash,
                document,
            ) in enumerate(documents):
                try:
                    # Generate new embedding
                    embedding = self._generate_embedding(document)
                    embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

                    # Insert into new table
                    cursor.execute(
                        f"""
                        INSERT INTO {self.table_name} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            col_name,
                            doc_name,
                            page_num,
                            doc_hash,
                            document,
                            self.embedding_model,
                            embedding_bytes,
                        ),
                    )

                    # Update FTS
                    cursor.execute(
                        f"INSERT INTO {self.table_name}_fts(rowid, document, doc_hash) VALUES (?, ?, ?)",
                        (cursor.lastrowid, document, doc_hash),
                    )

                    stats["updated"] += 1

                    # Progress reporting
                    if (i + 1) % batch_size == 0 or (i + 1) == len(documents):
                        logger.info(
                            f"Progress: {i + 1}/{len(documents)} documents re-embedded"
                        )
                        conn.commit()

                except Exception as e:
                    error_msg = f"Error re-embedding {doc_name} page {page_num}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            # Drop backup tables
            cursor.execute(f"DROP TABLE {backup_table}")
            cursor.execute(f"DROP TABLE {fts_backup}")
            conn.commit()

        logger.info(
            f"SQLite re-embedding complete: {stats['updated']}/{stats['total']} documents updated"
        )
        return stats

    def _reembed_duckdb(
        self, collection_name: str, batch_size: int, stats: dict
    ) -> dict:
        """
        Re-embed documents in DuckDB.
        If the embedding dimension changes, the table is recreated.
        """
        conn = self.connection
        new_dimension = self._get_embedding_dimension()

        # Get current dimension from table schema
        current_dimension_result = conn.sql(f"""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = '{self.table_name}' AND column_name = 'embedding'
        """).fetchone()

        dimension_changed = False
        if current_dimension_result:
            current_type = current_dimension_result[0]
            # Extract dimension from FLOAT[N]
            import re

            match = re.search(r"FLOAT\[(\d+)\]", current_type)
            if match:
                current_dimension = int(match.group(1))
                dimension_changed = current_dimension != new_dimension
                if dimension_changed:
                    logger.info(
                        f"Dimension change detected: {current_dimension} -> {new_dimension}. Table will be recreated."
                    )

        # Get all documents
        if collection_name:
            result = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name} WHERE collection_name = ?",
                params=[collection_name],
            ).fetchall()
        else:
            result = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name}"
            ).fetchall()

        documents = result
        stats["total"] = len(documents)

        if len(documents) == 0:
            logger.info("No documents to re-embed")
            return stats

        if dimension_changed:
            return self._reembed_duckdb_recreate(
                collection_name, batch_size, stats, new_dimension
            )

        # Process documents with same dimension
        for i, (doc_id, col_name, doc_name, page_num, doc_hash, document) in enumerate(
            documents
        ):
            try:
                embedding = self._generate_embedding(document)
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                conn.sql(
                    f"""
                    UPDATE {self.table_name}
                    SET embedding = ?::FLOAT[{new_dimension}], embedding_model = ?
                    WHERE id = ?
                    """,
                    params=[embedding_str, self.embedding_model, doc_id],
                )

                stats["updated"] += 1

                if (i + 1) % batch_size == 0 or (i + 1) == len(documents):
                    logger.info(
                        f"Progress: {i + 1}/{len(documents)} documents re-embedded"
                    )

            except Exception as e:
                error_msg = f"Error re-embedding {doc_name} page {page_num}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        logger.info(
            f"DuckDB re-embedding complete: {stats['updated']}/{stats['total']} documents updated"
        )
        return stats

    def _reembed_duckdb_recreate(
        self, collection_name: str, batch_size: int, stats: dict, new_dimension: int
    ) -> dict:
        """
        Re-embed documents in DuckDB by recreating the table (used when dimension changes).
        This version uses a safe approach: create temp table, verify, then swap.
        """
        conn = self.connection

        # Get all documents with full data
        if collection_name:
            result = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name} WHERE collection_name = ?",
                params=[collection_name],
            ).fetchall()
        else:
            result = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name}"
            ).fetchall()

        documents = result
        stats["total"] = len(documents)

        if len(documents) == 0:
            logger.info("No documents to re-embed")
            return stats

        logger.info(f"Recreating DuckDB table with new dimension {new_dimension}...")
        logger.info(f"Processing {len(documents)} documents...")

        # Use a temporary table name
        temp_table = f"{self.table_name}_temp_new"
        backup_table = f"{self.table_name}_backup_reembed"

        # Clean up any existing temp/backup tables
        conn.sql(f"DROP TABLE IF EXISTS {temp_table}")

        # Create new TEMPORARY table with new dimension
        conn.sql(f"CREATE SEQUENCE IF NOT EXISTS {temp_table}_seq;")
        create_sql = f"""
        CREATE TABLE {temp_table} (
            id INTEGER PRIMARY KEY DEFAULT nextval('{temp_table}_seq'),
            collection_name TEXT,
            doc_name TEXT,
            page_number INTEGER,
            doc_hash TEXT UNIQUE,
            document TEXT,
            embedding_model TEXT DEFAULT 'embeddinggemma',
            embedding FLOAT[{new_dimension}]
        );
        """
        conn.sql(create_sql)
        logger.info(f"Created temporary table {temp_table}")

        # Process documents into temp table
        successful_embeddings = 0
        failed_embeddings = 0

        for i, (doc_id, col_name, doc_name, page_num, doc_hash, document) in enumerate(
            documents
        ):
            try:
                embedding = self._generate_embedding(document)
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                conn.sql(
                    f"""
                    INSERT INTO {temp_table} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    params=[
                        col_name,
                        doc_name,
                        page_num,
                        doc_hash,
                        document,
                        self.embedding_model,
                        embedding_str,
                    ],
                )

                successful_embeddings += 1
                stats["updated"] += 1

                if (i + 1) % batch_size == 0 or (i + 1) == len(documents):
                    logger.info(
                        f"Progress: {i + 1}/{len(documents)} documents processed ({successful_embeddings} successful, {failed_embeddings} failed)"
                    )

            except Exception as e:
                failed_embeddings += 1
                error_msg = f"Error re-embedding {doc_name} page {page_num}: {e}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)

        # Verify that we have documents in the temp table
        temp_count = conn.sql(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
        logger.info(f"Temporary table has {temp_count} documents")

        # Check if we have a reasonable number of documents
        # We allow some failures but at least 50% should succeed
        if temp_count == 0:
            error_msg = "CRITICAL: No documents were successfully embedded! Keeping original table."
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            conn.sql(f"DROP TABLE {temp_table}")
            return stats

        if temp_count < len(documents) * 0.5:
            error_msg = f"WARNING: Only {temp_count}/{len(documents)} documents were successfully embedded. Keeping original table."
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            conn.sql(f"DROP TABLE {temp_table}")
            return stats

        # Load FTS extension
        conn.sql("INSTALL fts;")
        conn.sql("LOAD fts;")

        # Drop HNSW index if it exists on main table
        try:
            conn.sql(f"DROP INDEX IF EXISTS {self.table_name}_index;")
            logger.info("Dropped existing HNSW index")
        except Exception as e:
            logger.warning(f"Could not drop HNSW index (may not exist): {e}")

        # Drop FTS-related tables for main table
        try:
            fts_table_patterns = [
                f"fts_main_{self.table_name}",
                f"fts_data_{self.table_name}",
                f"fts_stats_{self.table_name}",
                f"fts_segments_{self.table_name}",
                f"fts_lists_{self.table_name}",
            ]
            for fts_table in fts_table_patterns:
                try:
                    conn.sql(f"DROP TABLE IF EXISTS {fts_table};")
                except:
                    pass
            logger.info("Cleaned up existing FTS tables")
        except Exception as e:
            logger.info(f"FTS cleanup skipped: {e}")

        # Now swap the tables: main -> backup, temp -> main
        conn.sql(f"DROP TABLE IF EXISTS {backup_table}")
        conn.sql(f"ALTER TABLE {self.table_name} RENAME TO {backup_table}")
        logger.info(f"Renamed {self.table_name} to {backup_table} (backup)")

        conn.sql(f"ALTER TABLE {temp_table} RENAME TO {self.table_name}")
        conn.sql(f"DROP SEQUENCE IF EXISTS {self.table_name}_seq")
        conn.sql(f"ALTER SEQUENCE {temp_table}_seq RENAME TO {self.table_name}_seq")
        logger.info(f"Renamed {temp_table} to {self.table_name}")

        # Recreate FTS index on new main table
        try:
            conn.sql(f"PRAGMA create_fts_index('{self.table_name}', 'id', 'document');")
            logger.info("Created FTS index on new table")
        except Exception as e:
            logger.warning(f"Could not create FTS index: {e}")

        # Recreate HNSW index (only for in-memory or with experimental persistence)
        try:
            conn.sql("SET hnsw_enable_experimental_persistence = true;")
            conn.sql(f"""
                CREATE INDEX {self.table_name}_index
                ON {self.table_name} USING HNSW(embedding)
                WITH (metric='cosine');
            """)
            logger.info("Created HNSW index on embedding column.")
        except Exception as e:
            logger.warning(
                f"Could not create HNSW index (this is OK for disk-based DBs): {e}"
            )

        # Final verification
        final_count = conn.sql(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
        logger.info(f"Final verification: {final_count} documents in new table")

        if final_count != temp_count:
            logger.error(f"Count mismatch! Expected {temp_count}, got {final_count}")
            stats["errors"].append(
                f"Count mismatch: expected {temp_count}, got {final_count}"
            )

        # Add info about backup table
        stats["backup_table"] = backup_table
        logger.info(
            f"Backup table '{backup_table}' has been preserved. You can drop it manually after verification."
        )

        logger.info(
            f"DuckDB re-embedding complete: {stats['updated']}/{stats['total']} documents updated"
        )
        logger.info(
            f"Successfully embedded: {successful_embeddings}, Failed: {failed_embeddings}"
        )
        return stats

    def _reembed_postgres(
        self, collection_name: str, batch_size: int, stats: dict
    ) -> dict:
        """
        Re-embed documents in PostgreSQL.
        """
        with Session(self.engine) as session:
            # Get all documents
            if collection_name:
                statement = select(self.embedding).where(
                    self.embedding.collection_name == collection_name
                )
            else:
                statement = select(self.embedding)

            documents = session.execute(statement).scalars().all()
            stats["total"] = len(documents)

            if len(documents) == 0:
                logger.info("No documents to re-embed")
                return stats

            # Process documents
            for i, doc in enumerate(documents):
                try:
                    # Generate new embedding
                    embedding = self._generate_embedding(doc.document)

                    # Update the record
                    doc.embedding = embedding
                    doc.embedding_model = self.embedding_model

                    stats["updated"] += 1

                    # Progress reporting and commit in batches
                    if (i + 1) % batch_size == 0 or (i + 1) == len(documents):
                        session.commit()
                        logger.info(
                            f"Progress: {i + 1}/{len(documents)} documents re-embedded"
                        )

                except Exception as e:
                    session.rollback()
                    error_msg = (
                        f"Error re-embedding {doc.doc_name} page {doc.page_number}: {e}"
                    )
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

        logger.info(
            f"PostgreSQL re-embedding complete: {stats['updated']}/{stats['total']} documents updated"
        )
        return stats

    def get_embedding_model_info(self) -> dict:
        """
        Get information about embedding models used in the database.

        :return: Dict with model counts per collection
        """
        info = {"models": {}, "total_documents": 0}

        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                result = cursor.execute(
                    f"SELECT embedding_model, collection_name, COUNT(*) as count FROM {self.table_name} GROUP BY embedding_model, collection_name"
                ).fetchall()
                for model, collection, count in result:
                    if model is None:
                        model = "unknown"
                    if model not in info["models"]:
                        info["models"][model] = {}
                    info["models"][model][collection] = count
                    info["total_documents"] += count

        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            result = conn.sql(
                f"SELECT embedding_model, collection_name, COUNT(*) as count FROM {self.table_name} GROUP BY embedding_model, collection_name"
            ).fetchall()
            for model, collection, count in result:
                if model is None:
                    model = "unknown"
                if model not in info["models"]:
                    info["models"][model] = {}
                info["models"][model][collection] = count
                info["total_documents"] += count
        else:
            with Session(self.engine) as session:
                result = session.execute(
                    text("""
                        SELECT embedding_model, collection_name, COUNT(*) as count
                        FROM embedding
                        GROUP BY embedding_model, collection_name
                    """)
                ).fetchall()
                for model, collection, count in result:
                    if model is None:
                        model = "unknown"
                    if model not in info["models"]:
                        info["models"][model] = {}
                    info["models"][model][collection] = count
                    info["total_documents"] += count

        return info

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
