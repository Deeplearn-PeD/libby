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
    UniqueConstraint,
)
from sqlalchemy.exc import IntegrityError, NoSuchModuleError
from sqlalchemy.orm import DeclarativeBase, Session
import duckdb
from duckdb import array_type

dotenv.load_dotenv()
logger = loguru.logger

MODEL_MAX_CHARS: dict[str, int] = {
    "mxbai-embed-large": 800,
    "embeddinggemma": 8000,
    "gemini-embedding-001": 8000,
}

DEFAULT_MAX_EMBED_CHARS = 800


# engine = create_engine(os.getenv("PGURL"))
# with Session(engine) as session:
#     session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))


# create a class to store the embeddings
class Base(DeclarativeBase):
    pass


class Embedding(Base):
    __tablename__ = "embedding"
    __table_args__ = (
        UniqueConstraint("collection_name", "doc_hash", name="uq_collection_doc_hash"),
        {"extend_existing": True},
    )
    id = Column(Integer, autoincrement=True, primary_key=True)
    collection_name = Column(String)
    doc_name = Column(String)
    page_number = Column(Integer)
    doc_hash = Column(String)
    document = Column(String)
    embedding_model = Column(String, default="embeddinggemma")
    embedding = Column(Vector(1024))


class DocEmbedder:
    def __init__(
        self,
        col_name,
        dburl: str = "",
        embedding_model: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
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
                self.table_name = Embedding.__tablename__

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

        # Detect the actual embedding model from the database — this overrides
        # any env-var / hardcoded default with the model that was actually
        # used when the existing embeddings were created.
        self._apply_embedding_model_from_db()

    def _detect_embedding_model_from_db(self) -> str | None:
        """
        Query the database for the most common ``embedding_model`` value.

        Returns ``None`` when the table is empty or the column is absent.
        """
        table = getattr(self, "table_name", None)
        if not table:
            return None
        try:
            if self.dburl.startswith("sqlite"):
                cur = self.connection.cursor()
                cur.execute(
                    f"SELECT embedding_model, COUNT(*) AS cnt FROM \"{table}\" "
                    "WHERE embedding_model IS NOT NULL AND embedding_model != '' "
                    "GROUP BY embedding_model ORDER BY cnt DESC LIMIT 1"
                )
                row = cur.fetchone()
                return row[0] if row else None

            elif self.dburl.startswith("duckdb"):
                row = self.connection.sql(
                    f"SELECT embedding_model, COUNT(*) AS cnt FROM {table} "
                    "WHERE embedding_model IS NOT NULL AND embedding_model != '' "
                    "GROUP BY embedding_model ORDER BY cnt DESC LIMIT 1"
                ).fetchone()
                return row[0] if row else None

            else:
                with Session(self.engine) as session:
                    row = session.execute(
                        text(
                            f"SELECT embedding_model, COUNT(*) AS cnt FROM {table} "
                            "WHERE embedding_model IS NOT NULL AND embedding_model != '' "
                            "GROUP BY embedding_model ORDER BY cnt DESC LIMIT 1"
                        )
                    ).fetchone()
                    return row[0] if row else None
        except Exception:
            return None

    def _apply_embedding_model_from_db(self):
        """Override ``self.embedding_model`` with the model stored in the database."""
        detected = self._detect_embedding_model_from_db()
        if detected and detected != self.embedding_model:
            logger.info(
                f"Detected embedding model '{detected}' from database "
                f"(was '{self.embedding_model}') — applying detected model."
            )
            self.embedding_model = detected
            # Reconfigure client for the detected model
            if "gemini" in detected.lower():
                self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            else:
                self.client = ollama.Client(
                    host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
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
                doc_hash TEXT,
                document TEXT,
                embedding_model TEXT DEFAULT 'embeddinggemma',
                embedding FLOAT[{dimension}],
                UNIQUE (collection_name, doc_hash)
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

    def _get_table_embedding_dimension(self) -> int | None:
        """Detect the actual embedding dimension of the existing table column."""
        try:
            if self.dburl.startswith("sqlite"):
                import re
                cur = self.connection.cursor()
                cur.execute(f"PRAGMA table_info({self.table_name})")
                for col in cur.fetchall():
                    if col[1] == "embedding":
                        m = re.search(r"float\[(\d+)\]", col[2].lower())
                        if m:
                            return int(m.group(1))
                # vec0 virtual tables store types in the CREATE statement
                row = cur.execute(
                    f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{self.table_name}'"
                ).fetchone()
                if row:
                    m = re.search(r"float\[(\d+)\]", row[0].lower())
                    if m:
                        return int(m.group(1))
                return None
            elif self.dburl.startswith("duckdb"):
                row = self.connection.sql(
                    f"SELECT data_type FROM information_schema.columns "
                    f"WHERE table_name = '{self.table_name}' AND column_name = 'embedding'"
                ).fetchone()
                if row:
                    import re
                    m = re.search(r"FLOAT\[(\d+)\]", row[0])
                    return int(m.group(1)) if m else None
            else:
                with self.engine.connect() as conn:
                    row = conn.execute(
                        text(
                            "SELECT data_type FROM information_schema.columns "
                            "WHERE table_name = 'embedding' AND column_name = 'embedding'"
                        )
                    ).fetchone()
                    if row:
                        import re
                        m = re.search(r"vector\((\d+)\)", row[0].lower())
                        return int(m.group(1)) if m else None
        except Exception:
            pass
        return None

    def _target_table_for_dimension(self, new_dim: int | None = None) -> str:
        """
        Return the table name to use for a given embedding dimension.

        If the dimension matches the existing table, returns ``self.table_name``.
        Otherwise creates (or reuses) a dimension-specific table
        ``{table_name}_{dim}`` and returns that name.
        """
        if new_dim is None:
            new_dim = self._get_embedding_dimension()
        table_dim = self._get_table_embedding_dimension()
        if table_dim is not None and new_dim == table_dim:
            return self.table_name

        target = f"{self.table_name}_{new_dim}"
        logger.info(f"Dimension mismatch: table has {table_dim}, model produces {new_dim}. Using table '{target}'")

        # Create the dimension-specific table if it doesn't exist
        if self.dburl.startswith("sqlite"):
            self._create_sqlite_table_for_dim(target, new_dim)
        elif self.dburl.startswith("duckdb"):
            self._create_duckdb_table_for_dim(target, new_dim)
        else:
            self._create_postgres_table_for_dim(target, new_dim)
        return target

    def _get_all_embedding_tables(self) -> list[str]:
        """Discover all embedding-related tables (main + dimension-specific + backups)."""
        tables = []
        try:
            if self.dburl.startswith("sqlite"):
                cur = self.connection.cursor()
                rows = cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'embedding%'"
                ).fetchall()
                tables = [r[0] for r in rows if not r[0].endswith("_fts")]
            elif self.dburl.startswith("duckdb"):
                rows = self.connection.sql(
                    "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'embedding%'"
                ).fetchall()
                tables = [r[0] for r in rows]
            else:
                with self.engine.connect() as conn:
                    rows = conn.execute(text(
                        "SELECT tablename FROM pg_tables WHERE tablename LIKE 'embedding%'"
                    )).fetchall()
                    tables = [r[0] for r in rows]
        except Exception as e:
            logger.warning(f"Error discovering tables: {e}")
        return tables

    def _active_table(self) -> str:
        """Return the table that should be used for queries/retrieval.

        If a dimension-specific table exists and has data for the current model,
        return it.  Otherwise return ``self.table_name``.
        """
        model_dim = self._get_embedding_dimension()
        table_dim = self._get_table_embedding_dimension()
        if table_dim is not None and model_dim == table_dim:
            return self.table_name
        candidate = f"{self.table_name}_{model_dim}"
        try:
            count = 0
            if self.dburl.startswith("sqlite"):
                cur = self.connection.cursor()
                try:
                    row = cur.execute(f"SELECT COUNT(*) FROM {candidate}").fetchone()
                    count = row[0] if row else 0
                except Exception:
                    return self.table_name
            elif self.dburl.startswith("duckdb"):
                try:
                    row = self.connection.sql(f"SELECT COUNT(*) FROM {candidate}").fetchone()
                    count = row[0] if row else 0
                except Exception:
                    return self.table_name
            else:
                with self.engine.connect() as conn:
                    try:
                        count = conn.execute(text(f"SELECT COUNT(*) FROM {candidate}")).scalar() or 0
                    except Exception:
                        return self.table_name
            if count > 0:
                return candidate
        except Exception:
            pass
        return self.table_name

    def _safe_swap_tables(self, source: str, target: str, backup: str, min_ratio: float = 0.9) -> dict:
        """Atomically swap *source* table aside to *backup* and promote *target* to *source*.

        For PostgreSQL the DDL is transactional.  For SQLite and DuckDB
        the rename steps are inherently sequential but we add a row-count
        safety gate: the target must contain at least *min_ratio* of the
        source row count or the swap is refused.

        Returns a dict with keys ``swapped``, ``source_count``,
        ``target_count``, ``backup_table``.
        """
        result = {"swapped": False, "source_count": 0, "target_count": 0, "backup_table": backup}

        def _count(table: str) -> int:
            if self.dburl.startswith("sqlite"):
                cur = self.connection.cursor()
                try:
                    return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except Exception:
                    return 0
            elif self.dburl.startswith("duckdb"):
                try:
                    return self.connection.sql(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except Exception:
                    return 0
            else:
                with self.engine.connect() as conn:
                    try:
                        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
                    except Exception:
                        return 0

        source_count = _count(source)
        target_count = _count(target)
        result["source_count"] = source_count
        result["target_count"] = target_count

        if source_count > 0 and target_count < source_count * min_ratio:
            logger.error(
                f"Refusing swap: target '{target}' has {target_count} rows "
                f"but source '{source}' has {source_count} (< {min_ratio*100:.0f}% threshold)"
            )
            return result

        if self.dburl.startswith("postgres") and not self.dburl.startswith("duckdb"):
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(text(f"LOCK TABLE {source} IN EXCLUSIVE MODE"))
                    conn.execute(text(f"LOCK TABLE {target} IN EXCLUSIVE MODE"))
                    conn.execute(text(f"DROP TABLE IF EXISTS {backup}"))
                    conn.execute(text(f"ALTER TABLE {source} RENAME TO {backup}"))
                    conn.execute(text(f"ALTER TABLE {target} RENAME TO {source}"))
                conn.commit()
        else:
            if self.dburl.startswith("sqlite"):
                cur = self.connection.cursor()
                cur.execute(f"DROP TABLE IF EXISTS {backup}")
                cur.execute(f"ALTER TABLE {source} RENAME TO {backup}")
                cur.execute(f"ALTER TABLE {target} RENAME TO {source}")
                self.connection.commit()
            else:
                conn = self.connection
                conn.sql(f"DROP TABLE IF EXISTS {backup}")
                conn.sql(f"ALTER TABLE {source} RENAME TO {backup}")
                conn.sql(f"ALTER TABLE {target} RENAME TO {source}")

        result["swapped"] = True
        logger.info(f"Safe swap complete: {source} ← {target}, backup → {backup}")
        return result

    def _create_postgres_table_for_dim(self, table_name: str, dimension: int):
        """Create a PostgreSQL embedding table with a specific vector dimension."""
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    collection_name TEXT,
                    doc_name TEXT,
                    page_number INTEGER,
                    doc_hash TEXT,
                    document TEXT,
                    embedding_model TEXT DEFAULT 'unknown',
                    embedding VECTOR({dimension}),
                    UNIQUE (collection_name, doc_hash)
                )
            """))
            conn.commit()
        logger.info(f"Created PostgreSQL table '{table_name}' with VECTOR({dimension})")

    def _create_sqlite_table_for_dim(self, table_name: str, dimension: int):
        """Create a SQLite embedding table (vec0) with a specific dimension."""
        conn = self.connection
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} USING vec0(
                collection_name TEXT,
                doc_name TEXT,
                page_number INTEGER,
                doc_hash TEXT,
                document TEXT,
                embedding_model TEXT,
                embedding float[{dimension}]
            )
        """)
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}_fts USING fts5(
                document,
                doc_hash UNINDEXED,
                content='{table_name}',
                content_rowid='rowid'
            )
        """)
        conn.commit()
        logger.info(f"Created SQLite table '{table_name}' with float[{dimension}]")

    def _create_duckdb_table_for_dim(self, table_name: str, dimension: int):
        """Create a DuckDB embedding table with a specific dimension."""
        conn = self.connection
        conn.sql(f"CREATE SEQUENCE IF NOT EXISTS {table_name}_seq;")
        conn.sql(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY DEFAULT nextval('{table_name}_seq'),
                collection_name TEXT,
                doc_name TEXT,
                page_number INTEGER,
                doc_hash TEXT,
                document TEXT,
                embedding_model TEXT DEFAULT 'embeddinggemma',
                embedding FLOAT[{dimension}],
                UNIQUE (collection_name, doc_hash)
            )
        """)
        conn.sql("INSTALL fts;")
        conn.sql("LOAD fts;")
        conn.sql(f"PRAGMA create_fts_index('{table_name}', 'id', 'document');")
        conn.sql(f"CREATE INDEX IF NOT EXISTS {table_name}_index ON {table_name} USING HNSW(embedding) WITH (metric='cosine');")
        logger.info(f"Created DuckDB table '{table_name}' with FLOAT[{dimension}]")

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
        Check if a document with this hash already exists in the current collection.
        :param hash: SHA256 hash of the document
        :return:
        """
        if self.dburl.startswith("sqlite"):
            target = self._target_table_for_dimension()
            with self.connection as conn:
                cursor = conn.cursor()
                q = f"SELECT doc_hash FROM {target} WHERE collection_name=? AND doc_hash=?"
                result = cursor.execute(q, (self.collection_name, hash)).fetchone()
            return result is not None
        elif self.dburl.startswith("duckdb"):
            target = self._target_table_for_dimension()
            conn = self.connection
            result = conn.sql(
                f"SELECT id FROM {target} WHERE collection_name = ? AND doc_hash = ?",
                params=[self.collection_name, hash],
            ).fetchall()
            return len(result) > 0
        else:
            target = self._target_table_for_dimension()
            with Session(self.engine) as session:
                result = session.execute(
                    text(f"SELECT 1 FROM {target} WHERE collection_name = :col AND doc_hash = :hash"),
                    {"col": self.collection_name, "hash": hash},
                ).fetchone()
            return result is not None

    def _max_embed_chars(self) -> int:
        return MODEL_MAX_CHARS.get(self.embedding_model, DEFAULT_MAX_EMBED_CHARS)

    def embed_text(self, doctext: str, docname: str, page_number: int):
        """
        Embed a page of a document.
        :param doctext: page of a document
        :param docname: name of the document
        :param page_number: page number
        :return:
        """
        limit = self._max_embed_chars()
        if len(doctext) > limit:
            logger.warning(
                f"Text for {docname} page {page_number} is {len(doctext)} chars "
                f"(>{limit}); auto-splitting into smaller chunks."
            )
            from libbydbot.brain.ingest import TextSplitter

            splitter = TextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            sub_chunks = splitter.split_text(doctext)
            for ci, sub in enumerate(sub_chunks):
                self._embed_single(sub, docname, page_number)
            return
        self._embed_single(doctext, docname, page_number)

    def _embed_single(self, doctext: str, docname: str, page_number: int):
        """Insert a single text chunk into the vector store."""
        limit = self._max_embed_chars()
        if len(doctext) > limit:
            logger.warning(
                f"Chunk for {docname} page {page_number} is still {len(doctext)} chars "
                f"after splitting; truncating to {limit}."
            )
            doctext = doctext[:limit]
        document_hash = sha256(doctext.encode()).hexdigest()
        if self._check_existing(document_hash):
            logger.info(
                f"Document {docname} page {page_number} already exists in the database, skipping."
            )
            return
        doctext = doctext.replace("\x00", "\ufffd")
        embedding = self._generate_embedding(doctext)

        if self.dburl.startswith("sqlite"):
            import struct

            target = self._target_table_for_dimension()
            target_fts = f"{target}_fts"
            embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

            with self.connection as conn:
                cursor = conn.cursor()
                try:
                    logger.info(
                        f"Inserting into {target}: {docname} page {page_number}"
                    )
                    cursor.execute(
                        f"""
                        INSERT INTO {target} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
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

                    cursor.execute(
                        f"INSERT INTO {target_fts}(rowid, document, doc_hash) VALUES (?, ?, ?)",
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
            target = self._target_table_for_dimension()
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            conn = self.connection
            try:
                conn.sql(
                    f"""
                    INSERT INTO {target} (doc_hash, doc_name, collection_name, page_number, document, embedding_model, embedding)
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
            target = self._target_table_for_dimension()
            with Session(self.engine) as session:
                try:
                    session.execute(
                        text(f"""
                            INSERT INTO {target}
                            (doc_hash, doc_name, collection_name, page_number, document, embedding_model, embedding)
                            VALUES (:hash, :name, :col, :page, :doc, :model, :embedding)
                        """),
                        {
                            "hash": document_hash,
                            "name": docname,
                            "col": self.collection_name,
                            "page": page_number,
                            "doc": doctext,
                            "model": self.embedding_model,
                            "embedding": embedding,
                        },
                    )
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

    def embed_path(self, corpus_path: str, callback=None):
        """
        Embed all documents in a path using chunking.
        :param corpus_path:  path to a folder containing PDFs
        :param callback: optional callable(doc_name, chunk_index, total_chunks) for progress reporting
        :return:
        """
        from libbydbot.brain.ingest import PDFPipeline, ChunkInfo

        pipeline = PDFPipeline(
            corpus_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        for chunks_or_dict, metadata in pipeline:
            docname = metadata.get("title", "Unknown")

            if isinstance(chunks_or_dict, list) and chunks_or_dict and isinstance(chunks_or_dict[0], ChunkInfo):
                total = len(chunks_or_dict)
                for ci, chunk_info in enumerate(chunks_or_dict):
                    self.embed_text(chunk_info.text, docname, chunk_info.page_number)
                    if callback:
                        callback(docname, ci, total)
            elif isinstance(chunks_or_dict, list):
                total = len(chunks_or_dict)
                for i, chunk in enumerate(chunks_or_dict):
                    self.embed_text(chunk, docname, i)
                    if callback:
                        callback(docname, i, total)
            else:
                total = len(chunks_or_dict)
                for idx, (page_number, page_text) in enumerate(chunks_or_dict.items()):
                    self.embed_text(page_text, docname, page_number)
                    if callback:
                        callback(docname, idx, total)

    def retrieve_docs(self, query: str, collection: str = "", num_docs: int = 5) -> str:
        """
        Retrieve documents based on a query.
        :param query: query string
        :param collection: collection name
        :param num_docs: number of documents to retrieve
        :return: all documents as a string
        """
        query_embedding = self._generate_embedding(query)
        tbl = self._active_table()

        if self.dburl.startswith("sqlite"):
            import struct

            query_embedding_bytes = struct.pack(
                f"{len(query_embedding)}f", *query_embedding
            )
            with self.connection as conn:
                cursor = conn.cursor()

                if collection:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, document, distance FROM {tbl} WHERE collection_name = ? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                        (collection, query_embedding_bytes, num_docs * 2),
                    ).fetchall()
                else:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, document, distance FROM {tbl} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                        (query_embedding_bytes, num_docs * 2),
                    ).fetchall()

                fts_results = cursor.execute(
                    f"SELECT doc_hash, document, rank FROM {tbl}_fts WHERE document MATCH ? ORDER BY rank LIMIT ?",
                    (query, num_docs * 2),
                ).fetchall()

                k = 60
                scores = {}
                docs = {}

                for rank, (d_hash, doc, dist) in enumerate(vector_results):
                    scores[d_hash] = scores.get(d_hash, 0) + 1.0 / (k + rank + 1)
                    docs[d_hash] = doc

                for rank, (d_hash, doc, rank_score) in enumerate(fts_results):
                    scores[d_hash] = scores.get(d_hash, 0) + 1.0 / (k + rank + 1)
                    docs[d_hash] = doc

                sorted_ids = sorted(
                    scores.keys(), key=lambda x: scores[x], reverse=True
                )[:num_docs]
                pages = [docs[d_hash] for d_hash in sorted_ids]
        elif self.dburl.startswith("duckdb"):
            dimension = self._get_embedding_dimension()
            embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
            conn = self.connection

            vector_results = conn.sql(f"""
                SELECT id, document, array_cosine_similarity(embedding, {embedding_str}::FLOAT[{dimension}]) as similarity
                FROM {tbl}
                {f"WHERE collection_name = '{collection}'" if collection else ""}
                ORDER BY similarity DESC
                LIMIT {num_docs * 2}
            """).fetchall()

            try:
                conn.sql("LOAD fts;")
                fts_results = conn.sql(
                    f"""
                    SELECT id, document, fts_main_{tbl}.match_bm25(id, ?) as score
                    FROM {tbl}
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT {num_docs * 2}
                """,
                    params=[query],
                ).fetchall()
            except Exception as e:
                logger.warning(f"FTS search failed: {e}")
                fts_results = []

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
            dimension = self._get_embedding_dimension()
            with Session(self.engine) as session:
                col_filter = f"WHERE collection_name = :col" if collection else ""
                rows = session.execute(
                    text(f"""
                        SELECT document FROM {tbl}
                        {col_filter}
                        ORDER BY embedding <=> :query_embedding::vector({dimension})
                        LIMIT :limit
                    """),
                    {"col": collection, "query_embedding": str(query_embedding), "limit": num_docs},
                ).fetchall()
                pages = [r[0] for r in rows]

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
        tbl = self._active_table()
        results = []

        if self.dburl.startswith("sqlite"):
            import struct

            query_embedding_bytes = struct.pack(
                f"{len(query_embedding)}f", *query_embedding
            )
            with self.connection as conn:
                cursor = conn.cursor()

                if collection:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, doc_name, page_number, document, distance FROM {tbl} WHERE collection_name = ? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                        (collection, query_embedding_bytes, num_docs * 2),
                    ).fetchall()
                else:
                    vector_results = cursor.execute(
                        f"SELECT doc_hash, doc_name, page_number, document, distance FROM {tbl} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                        (query_embedding_bytes, num_docs * 2),
                    ).fetchall()

                fts_results = cursor.execute(
                    f"""
                    SELECT f.doc_hash, t.doc_name, t.page_number, f.document, f.rank
                    FROM {tbl}_fts f
                    JOIN {tbl} t ON f.doc_hash = t.doc_hash
                    WHERE f.document MATCH ? ORDER BY f.rank LIMIT ?
                    """,
                    (query, num_docs * 2),
                ).fetchall()

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

            vector_results = conn.sql(f"""
                SELECT id, doc_name, page_number, document, array_cosine_similarity(embedding, {embedding_str}::FLOAT[{dimension}]) as similarity
                FROM {tbl}
                {f"WHERE collection_name = '{collection}'" if collection else ""}
                ORDER BY similarity DESC
                LIMIT {num_docs * 2}
            """).fetchall()

            try:
                conn.sql("LOAD fts;")
                fts_results = conn.sql(
                    f"""
                    SELECT id, doc_name, page_number, document, fts_main_{tbl}.match_bm25(id, ?) as score
                    FROM {tbl}
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT {num_docs * 2}
                """,
                    params=[query],
                ).fetchall()
            except Exception as e:
                logger.warning(f"FTS search failed: {e}")
                fts_results = []

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
            dimension = self._get_embedding_dimension()
            with Session(self.engine) as session:
                col_filter = "WHERE collection_name = :col" if collection else ""
                rows = session.execute(
                    text(f"""
                        SELECT doc_name, page_number, document
                        FROM {tbl}
                        {col_filter}
                        ORDER BY embedding <=> :query_embedding::vector({dimension})
                        LIMIT :limit
                    """),
                    {"col": collection, "query_embedding": str(query_embedding), "limit": num_docs},
                ).fetchall()
                for rank, (doc_name, page_num, doc) in enumerate(rows):
                    results.append(
                        {
                            "doc_name": doc_name,
                            "page_number": page_num,
                            "content": doc,
                            "score": 1.0 / (rank + 1),
                        }
                    )

        return results

    def get_embedded_documents(self):
        """
        Get a list of all embedded documents.
        :return: List of tuples (doc_name, collection_name) for all embedded documents
        """
        tbl = self._active_table()
        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                result = cursor.execute(
                    f"SELECT DISTINCT doc_name, collection_name FROM {tbl}"
                ).fetchall()
            return [(row[0], row[1]) for row in result]
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            result = conn.sql(
                f"SELECT DISTINCT doc_name, collection_name FROM {tbl}"
            ).fetchall()
            return [(row[0], row[1]) for row in result]
        else:
            with Session(self.engine) as session:
                result = session.execute(
                    text(f"SELECT DISTINCT doc_name, collection_name FROM {tbl}")
                ).fetchall()
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

    def migrate_compound_unique(self) -> dict:
        """
        Migrate existing tables from UNIQUE(doc_hash) to
        UNIQUE(collection_name, doc_hash).

        This is a no-op for SQLite (vec0 has no SQL-level constraints)
        and for dimension-specific tables already created with the new
        constraint.

        Returns a dict describing what was done.
        """
        result = {"backend": "sqlite" if self.dburl.startswith("sqlite") else "duckdb" if self.dburl.startswith("duckdb") else "postgresql", "changes": []}

        if self.dburl.startswith("sqlite"):
            result["changes"].append("Skipped: SQLite vec0 tables have no SQL constraints; dedup is application-level")
            return result

        tables = self._get_all_embedding_tables()

        if self.dburl.startswith("duckdb"):
            for tbl in tables:
                try:
                    constraints = self.connection.sql(
                        f"SELECT constraint_type, constraint_name FROM duckdb_constraints() WHERE table_name = '{tbl}'"
                    ).fetchall()
                    has_compound = any("collection_name" in str(c) and "doc_hash" in str(c) for c in constraints)
                    if has_compound:
                        result["changes"].append(f"{tbl}: already has compound unique constraint")
                        continue
                    self.connection.sql(f"ALTER TABLE {tbl} DROP CONSTRAINT IF EXISTS {tbl}_doc_hash_key")
                    self.connection.sql(f"ALTER TABLE {tbl} ADD CONSTRAINT {tbl}_uq_col_hash UNIQUE (collection_name, doc_hash)")
                    result["changes"].append(f"{tbl}: migrated to UNIQUE(collection_name, doc_hash)")
                except Exception as e:
                    result["changes"].append(f"{tbl}: error: {e}")
        else:
            for tbl in tables:
                try:
                    with self.engine.connect() as conn:
                        rows = conn.execute(text(
                            f"SELECT conname FROM pg_constraint c JOIN pg_class t ON c.conrelid = t.oid "
                            f"WHERE t.relname = '{tbl}' AND c.contype = 'u'"
                        )).fetchall()
                        has_compound = any("collection" in r[0].lower() for r in rows)
                        if has_compound:
                            result["changes"].append(f"{tbl}: already has compound unique constraint")
                            continue
                        for row in rows:
                            conn.execute(text(f'ALTER TABLE {tbl} DROP CONSTRAINT IF EXISTS "{row[0]}"'))
                        conn.execute(text(
                            f"ALTER TABLE {tbl} ADD CONSTRAINT uq_{tbl}_col_hash UNIQUE (collection_name, doc_hash)"
                        ))
                        conn.commit()
                        result["changes"].append(f"{tbl}: migrated to UNIQUE(collection_name, doc_hash)")
                except Exception as e:
                    result["changes"].append(f"{tbl}: error: {e}")

        return result

    def reembed(
        self,
        collection_name: str = "",
        new_model: str | None = None,
        batch_size: int = 100,
        rechunk: bool = False,
        new_chunk_size: int = 1500,
        new_chunk_overlap: int = 200,
    ) -> dict:
        """
        Re-embed all documents with a new embedding model.

        When *rechunk* is True the existing chunks are first reconstructed
        into their source documents, then re-split with the new chunk size
        before being embedded with the new model.  The result is written to
        a **shadow collection** (``{collection}_v2``) so the original
        collection remains fully queryable during the process.

        :param collection_name: Collection to re-embed (empty = all collections)
        :param new_model: New embedding model (None = use current default from settings)
        :param batch_size: Number of documents to process per batch
        :param rechunk: Reconstruct source text and re-chunk before embedding
        :param new_chunk_size: Chunk size to use when rechunking
        :param new_chunk_overlap: Chunk overlap to use when rechunking
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

        if rechunk:
            stats["old_chunk_size"] = self.chunk_size
            stats["new_chunk_size"] = new_chunk_size
            stats["new_chunk_overlap"] = new_chunk_overlap
            stats["total_old_chunks"] = 0
            stats["total_new_chunks"] = 0
            stats["shadow_collection"] = ""
            if self.dburl.startswith("sqlite"):
                return self._rechunk_sqlite(collection_name, new_chunk_size, new_chunk_overlap, batch_size, stats)
            elif self.dburl.startswith("duckdb"):
                return self._rechunk_duckdb(collection_name, new_chunk_size, new_chunk_overlap, batch_size, stats)
            else:
                return self._rechunk_postgres(collection_name, new_chunk_size, new_chunk_overlap, batch_size, stats)

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
                    f"SELECT rowid, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name} WHERE collection_name = ?",
                    (collection_name,),
                )
            else:
                cursor.execute(
                    f"SELECT rowid, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name}"
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

            # Safety check before dropping backup
            new_count = cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
            if stats["total"] > 0 and new_count < stats["total"] * 0.9:
                logger.error(
                    f"SQLite re-embed incomplete: {new_count}/{stats['total']} rows. "
                    f"Keeping backup table '{backup_table}'"
                )
                stats["errors"].append(
                    f"Re-embed incomplete ({new_count}/{stats['total']}); backup preserved as '{backup_table}'"
                )
                stats["backup_table"] = backup_table
            else:
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
        swap_result = self._safe_swap_tables(self.table_name, temp_table, backup_table)
        if not swap_result["swapped"]:
            logger.error("Swap refused — keeping original table, dropping temp")
            conn.sql(f"DROP TABLE IF EXISTS {temp_table}")
            stats["errors"].append("Swap refused: insufficient rows in target")
            return stats

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

        If the new model's embedding dimension differs from the existing table
        column, a new dimension-specific table is created, data is migrated,
        and the tables are swapped (old → backup, new → active).
        """
        new_dim = self._get_embedding_dimension()
        target_table = self._target_table_for_dimension(new_dim)
        dimension_changed = target_table != self.table_name
        source_table = self.table_name

        with Session(self.engine) as session:
            # Get all documents
            if dimension_changed:
                rows = session.execute(
                    text(
                        f"SELECT collection_name, doc_name, page_number, doc_hash, document "
                        f"FROM {source_table}"
                        + (f" WHERE collection_name = :col" if collection_name else "")
                    ),
                    {"col": collection_name} if collection_name else {},
                ).fetchall()
            else:
                stmt = select(self.embedding)
                if collection_name:
                    stmt = stmt.where(self.embedding.collection_name == collection_name)
                rows = [
                    (d.collection_name, d.doc_name, d.page_number, d.doc_hash, d.document)
                    for d in session.execute(stmt).scalars().all()
                ]

            stats["total"] = len(rows)
            if not rows:
                logger.info("No documents to re-embed")
                return stats

            for i, (col_name, doc_name, page_num, doc_hash, document) in enumerate(rows):
                try:
                    embedding = self._generate_embedding(document)

                    if not dimension_changed:
                        session.execute(
                            text(f"""
                                UPDATE {source_table}
                                SET embedding = :embedding, embedding_model = :model
                                WHERE doc_hash = :hash
                            """),
                            {"embedding": embedding, "model": self.embedding_model, "hash": doc_hash},
                        )
                    else:
                        session.execute(
                            text(f"""
                                INSERT INTO {target_table}
                                (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                                VALUES (:col, :doc, :page, :hash, :text, :model, :embedding)
                            """),
                            {
                                "col": col_name, "doc": doc_name, "page": page_num,
                                "hash": doc_hash, "text": document,
                                "model": self.embedding_model, "embedding": embedding,
                            },
                        )

                    stats["updated"] += 1
                    if (i + 1) % batch_size == 0 or (i + 1) == len(rows):
                        session.commit()
                        logger.info(f"Progress: {i + 1}/{len(rows)} documents re-embedded")

                except Exception as e:
                    session.rollback()
                    error_msg = f"Error re-embedding {doc_name} page {page_num}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

            session.commit()

        if dimension_changed:
            backup_table = f"{source_table}_backup_reembed"
            swap_result = self._safe_swap_tables(source_table, target_table, backup_table)
            if swap_result["swapped"]:
                self.table_name = source_table
                stats["backup_table"] = backup_table
            else:
                stats["errors"].append(
                    f"Swap refused: target had {swap_result['target_count']} rows vs source {swap_result['source_count']}"
                )

        logger.info(
            f"PostgreSQL re-embedding complete: {stats['updated']}/{stats['total']} documents updated"
        )
        return stats

    def finalize_rechunk(
        self,
        collection_name: str,
        shadow_collection: str,
        shadow_table: str | None = None,
    ) -> dict:
        """
        Finalize a rechunk operation by cutover from the shadow collection.

        Steps:
        1. Count rows in shadow vs original to ensure completeness.
        2. Delete the original collection from the active table.
        3. Rename the shadow collection to the original name.
        4. If *shadow_table* is a dimension-specific table, swap it to
           become the active table.

        :param collection_name: Original collection name (e.g. ``"my_docs"``)
        :param shadow_collection: Shadow collection name (e.g. ``"my_docs_v2"``)
        :param shadow_table: Dimension-specific table if rechunk used one
        :return: Stats dict
        """
        stats = {
            "success": False,
            "collection": collection_name,
            "shadow_collection": shadow_collection,
            "shadow_table": shadow_table,
            "original_count": 0,
            "shadow_count": 0,
            "deleted_original": 0,
            "renamed": 0,
            "table_swapped": False,
            "errors": [],
        }

        active = self.table_name
        target = shadow_table or active

        def _col_count(table: str, col: str) -> int:
            if self.dburl.startswith("sqlite"):
                cur = self.connection.cursor()
                try:
                    return cur.execute(f"SELECT COUNT(*) FROM {table} WHERE collection_name = ?", (col,)).fetchone()[0]
                except Exception:
                    return 0
            elif self.dburl.startswith("duckdb"):
                try:
                    return self.connection.sql(f"SELECT COUNT(*) FROM {table} WHERE collection_name = ?", params=[col]).fetchone()[0]
                except Exception:
                    return 0
            else:
                with self.engine.connect() as conn:
                    try:
                        return conn.execute(text(f"SELECT COUNT(*) FROM {table} WHERE collection_name = :col"), {"col": col}).scalar() or 0
                    except Exception:
                        return 0

        stats["original_count"] = _col_count(active, collection_name)
        stats["shadow_count"] = _col_count(target, shadow_collection)

        if stats["shadow_count"] == 0:
            stats["errors"].append("Shadow collection has no rows — nothing to finalize")
            return stats

        if stats["original_count"] > 0 and stats["shadow_count"] < stats["original_count"] * 0.9:
            stats["errors"].append(
                f"Shadow has {stats['shadow_count']} rows vs original {stats['original_count']} (< 90%). "
                f"Refusing to finalize."
            )
            return stats

        try:
            if self.dburl.startswith("sqlite"):
                cur = self.connection.cursor()
                cur.execute(f"DELETE FROM {active} WHERE collection_name = ?", (collection_name,))
                stats["deleted_original"] = cur.rowcount
                cur.execute(
                    f"UPDATE {target} SET collection_name = ? WHERE collection_name = ?",
                    (collection_name, shadow_collection),
                )
                stats["renamed"] = cur.rowcount
                self.connection.commit()
            elif self.dburl.startswith("duckdb"):
                conn = self.connection
                conn.sql(f"DELETE FROM {active} WHERE collection_name = ?", params=[collection_name])
                stats["deleted_original"] = stats["original_count"]
                conn.sql(
                    f"UPDATE {target} SET collection_name = ? WHERE collection_name = ?",
                    params=[collection_name, shadow_collection],
                )
                stats["renamed"] = stats["shadow_count"]
            else:
                with self.engine.connect() as conn:
                    with conn.begin():
                        result = conn.execute(
                            text(f"DELETE FROM {active} WHERE collection_name = :col"),
                            {"col": collection_name},
                        )
                        stats["deleted_original"] = result.rowcount
                        result = conn.execute(
                            text(f"UPDATE {target} SET collection_name = :new WHERE collection_name = :old"),
                            {"new": collection_name, "old": shadow_collection},
                        )
                        stats["renamed"] = result.rowcount
                    conn.commit()

            if shadow_table and shadow_table != active:
                backup = f"{active}_backup_reembed"
                swap_result = self._safe_swap_tables(active, shadow_table, backup)
                stats["table_swapped"] = swap_result["swapped"]
                if swap_result["swapped"]:
                    self.table_name = active
                else:
                    stats["errors"].append(
                        f"Collection renamed but table swap refused: {swap_result['target_count']} vs {swap_result['source_count']}"
                    )

            stats["success"] = True
            logger.info(
                f"Finalized rechunk: {collection_name} ({stats['renamed']} rows from shadow)"
            )
        except Exception as e:
            stats["errors"].append(str(e))
            logger.error(f"Error finalizing rechunk: {e}")

        return stats

    @staticmethod
    def _reconstruct_documents(rows: list[tuple]) -> dict[tuple[str, str], str]:
        """
        Reconstruct full source text from stored chunks.

        :param rows: list of (id, collection_name, doc_name, page_number, doc_hash, document)
        :return: dict keyed by (collection_name, doc_name) → reconstructed full text
        """
        from collections import OrderedDict

        grouped: dict[tuple[str, str], list[tuple[int, int, str]]] = OrderedDict()
        for row in rows:
            row_id, col_name, doc_name, page_num, _doc_hash, document = row
            key = (col_name, doc_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((page_num, row_id, document))

        reconstructed: dict[tuple[str, str], str] = OrderedDict()
        for key, chunks in grouped.items():
            chunks.sort(key=lambda c: (c[0], c[1]))
            parts = [c[2] for c in chunks]
            reconstructed[key] = "\n".join(parts)

        return reconstructed

    def _rechunk_sqlite(
        self,
        collection_name: str,
        new_chunk_size: int,
        new_chunk_overlap: int,
        batch_size: int,
        stats: dict,
    ) -> dict:
        import struct
        from libbydbot.brain.ingest import TextSplitter

        new_dim = self._get_embedding_dimension()
        target_table = self._target_table_for_dimension(new_dim)
        target_fts = f"{target_table}_fts"
        splitter = TextSplitter(chunk_size=new_chunk_size, chunk_overlap=new_chunk_overlap)

        with self.connection as conn:
            cursor = conn.cursor()

            if collection_name:
                cursor.execute(
                    f"SELECT rowid, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name} WHERE collection_name = ?",
                    (collection_name,),
                )
            else:
                cursor.execute(
                    f"SELECT rowid, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name}"
                )

            rows = cursor.fetchall()
            stats["total_old_chunks"] = len(rows)

            if not rows:
                logger.info("No documents to re-chunk")
                return stats

            reconstructed = self._reconstruct_documents(rows)

            shadow_suffix = "_v2"
            shadow_map: dict[str, str] = {}
            for (col_name, _doc_name) in reconstructed:
                if col_name not in shadow_map:
                    shadow_map[col_name] = col_name + shadow_suffix

            total_new = 0
            processed = 0

            for (col_name, doc_name), full_text in reconstructed.items():
                shadow_col = shadow_map[col_name]
                chunks = splitter.split_text(full_text)

                for ci, chunk_text in enumerate(chunks):
                    try:
                        doc_hash = sha256(f"{doc_name}:{ci}:{chunk_text}".encode()).hexdigest()
                        chunk_text = chunk_text.replace("\x00", "\ufffd")
                        embedding = self._generate_embedding(chunk_text)
                        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

                        cursor.execute(
                            f"""
                            INSERT INTO {target_table} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (shadow_col, doc_name, ci, doc_hash, chunk_text, self.embedding_model, embedding_bytes),
                        )

                        cursor.execute(
                            f"INSERT INTO {target_fts}(rowid, document, doc_hash) VALUES (?, ?, ?)",
                            (cursor.lastrowid, chunk_text, doc_hash),
                        )

                        total_new += 1
                        stats["updated"] += 1

                    except Exception as e:
                        error_msg = f"Error re-chunking {doc_name} chunk {ci}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

                    processed += 1
                    if processed % batch_size == 0:
                        conn.commit()
                        logger.info(f"Progress: {processed} new chunks written")

                conn.commit()

            stats["total_new_chunks"] = total_new
            stats["shadow_collection"] = shadow_map.get(collection_name, "")
            if target_table != self.table_name:
                stats["shadow_table"] = target_table
            logger.info(
                f"SQLite re-chunk complete: {stats['total_old_chunks']} old → {total_new} new chunks"
                + (f" (into table '{target_table}')" if target_table != self.table_name else "")
            )

        return stats

    def _rechunk_duckdb(
        self,
        collection_name: str,
        new_chunk_size: int,
        new_chunk_overlap: int,
        batch_size: int,
        stats: dict,
    ) -> dict:
        from libbydbot.brain.ingest import TextSplitter

        new_dim = self._get_embedding_dimension()
        target_table = self._target_table_for_dimension(new_dim)
        splitter = TextSplitter(chunk_size=new_chunk_size, chunk_overlap=new_chunk_overlap)
        conn = self.connection

        if collection_name:
            result = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name} WHERE collection_name = ?",
                params=[collection_name],
            ).fetchall()
        else:
            result = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document FROM {self.table_name}"
            ).fetchall()

        rows = result
        stats["total_old_chunks"] = len(rows)

        if not rows:
            logger.info("No documents to re-chunk")
            return stats

        reconstructed = self._reconstruct_documents(rows)

        shadow_suffix = "_v2"
        shadow_map: dict[str, str] = {}
        for (col_name, _doc_name) in reconstructed:
            if col_name not in shadow_map:
                shadow_map[col_name] = col_name + shadow_suffix

        total_new = 0
        processed = 0

        for (col_name, doc_name), full_text in reconstructed.items():
            shadow_col = shadow_map[col_name]
            chunks = splitter.split_text(full_text)

            for ci, chunk_text in enumerate(chunks):
                try:
                    doc_hash = sha256(f"{doc_name}:{ci}:{chunk_text}".encode()).hexdigest()
                    chunk_text = chunk_text.replace("\x00", "\ufffd")
                    embedding = self._generate_embedding(chunk_text)
                    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                    conn.sql(
                        f"""
                        INSERT INTO {target_table} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        params=[shadow_col, doc_name, ci, doc_hash, chunk_text, self.embedding_model, embedding_str],
                    )

                    total_new += 1
                    stats["updated"] += 1

                except Exception as e:
                    error_msg = f"Error re-chunking {doc_name} chunk {ci}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)

                processed += 1
                if processed % batch_size == 0:
                    logger.info(f"Progress: {processed} new chunks written")

        stats["total_new_chunks"] = total_new
        stats["shadow_collection"] = shadow_map.get(collection_name, "")
        if target_table != self.table_name:
            stats["shadow_table"] = target_table
        logger.info(
            f"DuckDB re-chunk complete: {stats['total_old_chunks']} old → {total_new} new chunks"
        )

        return stats

    def _rechunk_postgres(
        self,
        collection_name: str,
        new_chunk_size: int,
        new_chunk_overlap: int,
        batch_size: int,
        stats: dict,
    ) -> dict:
        from libbydbot.brain.ingest import TextSplitter

        new_dim = self._get_embedding_dimension()
        target_table = self._target_table_for_dimension(new_dim)
        splitter = TextSplitter(chunk_size=new_chunk_size, chunk_overlap=new_chunk_overlap)

        with Session(self.engine) as session:
            # Fetch source documents
            if collection_name:
                stmt = select(self.embedding).where(
                    self.embedding.collection_name == collection_name
                )
            else:
                stmt = select(self.embedding)

            documents = session.execute(stmt).scalars().all()
            stats["total_old_chunks"] = len(documents)

            if not documents:
                logger.info("No documents to re-chunk")
                return stats

            rows = [
                (d.id, d.collection_name, d.doc_name, d.page_number, d.doc_hash, d.document)
                for d in documents
            ]
            reconstructed = self._reconstruct_documents(rows)

            shadow_suffix = "_v2"
            shadow_map: dict[str, str] = {}
            for (col_name, _doc_name) in reconstructed:
                if col_name not in shadow_map:
                    shadow_map[col_name] = col_name + shadow_suffix

            total_new = 0
            processed = 0

            for (col_name, doc_name), full_text in reconstructed.items():
                shadow_col = shadow_map[col_name]
                chunks = splitter.split_text(full_text)

                for ci, chunk_text in enumerate(chunks):
                    try:
                        doc_hash = sha256(f"{doc_name}:{ci}:{chunk_text}".encode()).hexdigest()
                        chunk_text = chunk_text.replace("\x00", "\ufffd")
                        embedding = self._generate_embedding(chunk_text)

                        session.execute(
                            text(f"""
                                INSERT INTO {target_table}
                                (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                                VALUES (:col, :doc, :page, :hash, :text, :model, :embedding)
                            """),
                            {
                                "col": shadow_col, "doc": doc_name, "page": ci,
                                "hash": doc_hash, "text": chunk_text,
                                "model": self.embedding_model, "embedding": embedding,
                            },
                        )

                        total_new += 1
                        stats["updated"] += 1

                    except IntegrityError:
                        session.rollback()
                        logger.warning(f"Skipping duplicate chunk {doc_name} chunk {ci}")
                    except Exception as e:
                        error_msg = f"Error re-chunking {doc_name} chunk {ci}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

                    processed += 1
                    if processed % batch_size == 0:
                        session.commit()
                        logger.info(f"Progress: {processed} new chunks written")

                session.commit()

            stats["total_new_chunks"] = total_new
            stats["shadow_collection"] = shadow_map.get(collection_name, "")
            if target_table != self.table_name:
                stats["shadow_table"] = target_table
            logger.info(
                f"PostgreSQL re-chunk complete: {stats['total_old_chunks']} old → {total_new} new chunks"
                + (f" (into table '{target_table}')" if target_table != self.table_name else "")
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

    def delete_collection(self, collection_name: str) -> dict:
        """
        Delete all documents in a collection.

        :param collection_name: Collection to delete
        :return: Stats dict with count of deleted documents
        """
        stats = {"deleted": 0, "errors": []}

        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {self.table_name} WHERE collection_name = ?",
                        (collection_name,),
                    )
                    count = cursor.fetchone()[0]
                    cursor.execute(
                        f"DELETE FROM {self.table_name} WHERE collection_name = ?",
                        (collection_name,),
                    )
                    conn.commit()
                    stats["deleted"] = count
                    logger.info(f"Deleted {count} documents from collection '{collection_name}'")
                except Exception as e:
                    conn.rollback()
                    stats["errors"].append(str(e))
                    logger.error(f"Error deleting collection '{collection_name}': {e}")
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            try:
                count = conn.sql(
                    f"SELECT COUNT(*) FROM {self.table_name} WHERE collection_name = ?",
                    params=[collection_name],
                ).fetchone()[0]
                conn.sql(
                    f"DELETE FROM {self.table_name} WHERE collection_name = ?",
                    params=[collection_name],
                )
                stats["deleted"] = count
                logger.info(f"Deleted {count} documents from collection '{collection_name}'")
            except Exception as e:
                stats["errors"].append(str(e))
                logger.error(f"Error deleting collection '{collection_name}': {e}")
        else:
            with Session(self.engine) as session:
                try:
                    count = session.execute(
                        select(func.count()).where(
                            self.embedding.collection_name == collection_name
                        )
                    ).scalar()
                    session.execute(
                        self.embedding.__table__.delete().where(
                            self.embedding.collection_name == collection_name
                        )
                    )
                    session.commit()
                    stats["deleted"] = count
                    logger.info(f"Deleted {count} documents from collection '{collection_name}'")
                except Exception as e:
                    session.rollback()
                    stats["errors"].append(str(e))
                    logger.error(f"Error deleting collection '{collection_name}': {e}")

        return stats

    def delete_document(self, doc_name: str, collection_name: str = "") -> dict:
        """
        Delete all chunks of a specific document.

        :param doc_name: Document name to delete
        :param collection_name: Optional collection filter
        :return: Stats dict with count of deleted chunks
        """
        stats = {"deleted": 0, "errors": []}

        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                try:
                    if collection_name:
                        cursor.execute(
                            f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ? AND collection_name = ?",
                            (doc_name, collection_name),
                        )
                    else:
                        cursor.execute(
                            f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ?",
                            (doc_name,),
                        )
                    count = cursor.fetchone()[0]

                    if collection_name:
                        cursor.execute(
                            f"DELETE FROM {self.table_name} WHERE doc_name = ? AND collection_name = ?",
                            (doc_name, collection_name),
                        )
                    else:
                        cursor.execute(
                            f"DELETE FROM {self.table_name} WHERE doc_name = ?",
                            (doc_name,),
                        )
                    conn.commit()
                    stats["deleted"] = count
                    logger.info(f"Deleted {count} chunks of document '{doc_name}'")
                except Exception as e:
                    conn.rollback()
                    stats["errors"].append(str(e))
                    logger.error(f"Error deleting document '{doc_name}': {e}")
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            try:
                if collection_name:
                    count = conn.sql(
                        f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ? AND collection_name = ?",
                        params=[doc_name, collection_name],
                    ).fetchone()[0]
                    conn.sql(
                        f"DELETE FROM {self.table_name} WHERE doc_name = ? AND collection_name = ?",
                        params=[doc_name, collection_name],
                    )
                else:
                    count = conn.sql(
                        f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ?",
                        params=[doc_name],
                    ).fetchone()[0]
                    conn.sql(
                        f"DELETE FROM {self.table_name} WHERE doc_name = ?",
                        params=[doc_name],
                    )
                stats["deleted"] = count
                logger.info(f"Deleted {count} chunks of document '{doc_name}'")
            except Exception as e:
                stats["errors"].append(str(e))
                logger.error(f"Error deleting document '{doc_name}': {e}")
        else:
            with Session(self.engine) as session:
                try:
                    stmt = self.embedding.__table__.delete().where(
                        self.embedding.doc_name == doc_name
                    )
                    if collection_name:
                        stmt = stmt.where(self.embedding.collection_name == collection_name)
                    result = session.execute(stmt)
                    session.commit()
                    stats["deleted"] = result.rowcount
                    logger.info(f"Deleted {result.rowcount} chunks of document '{doc_name}'")
                except Exception as e:
                    session.rollback()
                    stats["errors"].append(str(e))
                    logger.error(f"Error deleting document '{doc_name}': {e}")

        return stats

    def reassign_document(self, doc_name: str, new_collection: str, old_collection: str = "") -> dict:
        """
        Move a document from one collection to another.

        :param doc_name: Document name to move
        :param new_collection: Target collection name
        :param old_collection: Optional source collection filter
        :return: Stats dict with count of moved chunks
        """
        stats = {"moved": 0, "errors": []}

        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                try:
                    if old_collection:
                        cursor.execute(
                            f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ? AND collection_name = ?",
                            (doc_name, old_collection),
                        )
                    else:
                        cursor.execute(
                            f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ?",
                            (doc_name,),
                        )
                    count = cursor.fetchone()[0]

                    if old_collection:
                        cursor.execute(
                            f"UPDATE {self.table_name} SET collection_name = ? WHERE doc_name = ? AND collection_name = ?",
                            (new_collection, doc_name, old_collection),
                        )
                    else:
                        cursor.execute(
                            f"UPDATE {self.table_name} SET collection_name = ? WHERE doc_name = ?",
                            (new_collection, doc_name),
                        )
                    conn.commit()
                    stats["moved"] = count
                    logger.info(f"Moved {count} chunks of '{doc_name}' to collection '{new_collection}'")
                except Exception as e:
                    conn.rollback()
                    stats["errors"].append(str(e))
                    logger.error(f"Error moving document '{doc_name}': {e}")
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            try:
                if old_collection:
                    count = conn.sql(
                        f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ? AND collection_name = ?",
                        params=[doc_name, old_collection],
                    ).fetchone()[0]
                    conn.sql(
                        f"UPDATE {self.table_name} SET collection_name = ? WHERE doc_name = ? AND collection_name = ?",
                        params=[new_collection, doc_name, old_collection],
                    )
                else:
                    count = conn.sql(
                        f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_name = ?",
                        params=[doc_name],
                    ).fetchone()[0]
                    conn.sql(
                        f"UPDATE {self.table_name} SET collection_name = ? WHERE doc_name = ?",
                        params=[new_collection, doc_name],
                    )
                stats["moved"] = count
                logger.info(f"Moved {count} chunks of '{doc_name}' to collection '{new_collection}'")
            except Exception as e:
                stats["errors"].append(str(e))
                logger.error(f"Error moving document '{doc_name}': {e}")
        else:
            with Session(self.engine) as session:
                try:
                    stmt = (
                        self.embedding.__table__.update()
                        .where(self.embedding.doc_name == doc_name)
                        .values(collection_name=new_collection)
                    )
                    if old_collection:
                        stmt = stmt.where(self.embedding.collection_name == old_collection)
                    result = session.execute(stmt)
                    session.commit()
                    stats["moved"] = result.rowcount
                    logger.info(f"Moved {result.rowcount} chunks of '{doc_name}' to collection '{new_collection}'")
                except Exception as e:
                    session.rollback()
                    stats["errors"].append(str(e))
                    logger.error(f"Error moving document '{doc_name}': {e}")

        return stats

    def reassign_collection(self, old_collection: str, new_collection: str) -> dict:
        """
        Rename a collection by updating all documents in it.

        :param old_collection: Current collection name
        :param new_collection: New collection name
        :return: Stats dict with count of moved documents
        """
        stats = {"moved": 0, "errors": []}

        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                try:
                    count = cursor.execute(
                        f"SELECT COUNT(*) FROM {self.table_name} WHERE collection_name = ?",
                        (old_collection,),
                    ).fetchone()[0]
                    cursor.execute(
                        f"UPDATE {self.table_name} SET collection_name = ? WHERE collection_name = ?",
                        (new_collection, old_collection),
                    )
                    conn.commit()
                    stats["moved"] = count
                    logger.info(f"Renamed collection '{old_collection}' to '{new_collection}' ({count} documents)")
                except Exception as e:
                    conn.rollback()
                    stats["errors"].append(str(e))
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            try:
                count = conn.sql(
                    f"SELECT COUNT(*) FROM {self.table_name} WHERE collection_name = ?",
                    params=[old_collection],
                ).fetchone()[0]
                conn.sql(
                    f"UPDATE {self.table_name} SET collection_name = ? WHERE collection_name = ?",
                    params=[new_collection, old_collection],
                )
                stats["moved"] = count
                logger.info(f"Renamed collection '{old_collection}' to '{new_collection}' ({count} documents)")
            except Exception as e:
                stats["errors"].append(str(e))
        else:
            with Session(self.engine) as session:
                try:
                    result = session.execute(
                        self.embedding.__table__.update()
                        .where(self.embedding.collection_name == old_collection)
                        .values(collection_name=new_collection)
                    )
                    session.commit()
                    stats["moved"] = result.rowcount
                    logger.info(f"Renamed collection '{old_collection}' to '{new_collection}' ({result.rowcount} documents)")
                except Exception as e:
                    session.rollback()
                    stats["errors"].append(str(e))

        return stats

    @staticmethod
    def resolve_target_dburl(backend: str) -> str:
        from libbydbot.settings import Settings
        settings = Settings()
        if backend == "postgresql":
            url = settings.target_postgres_url
            if not url:
                raise ValueError("PostgreSQL target not configured. Set PGURL in .env or environment.")
            return url
        elif backend == "duckdb":
            return f"duckdb:///{settings.target_duckdb_path}"
        elif backend == "sqlite":
            return f"sqlite:///{settings.target_sqlite_path}"
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from: postgresql, duckdb, sqlite")

    def rollback_embedding(self, dry_run: bool = False) -> dict:
        """
        Rollback to the backup table created by a previous re-embed operation.

        Supported for DuckDB and PostgreSQL (where backup tables are preserved
        after dimension-changing re-embeds).  SQLite drops its backup after
        successful re-embed and does not support rollback.

        :param dry_run: If True, report what would be done without making changes.
        :return: Dict with rollback status and statistics.
        """
        backup_table = f"{self.table_name}_backup_reembed"
        backend = (
            "duckdb" if self.dburl.startswith("duckdb")
            else "sqlite" if self.dburl.startswith("sqlite")
            else "postgresql"
        )

        result = {
            "success": False,
            "backend": backend,
            "backup_table": backup_table,
            "current_model": self.embedding_model,
            "backup_model": "",
            "backup_count": 0,
            "current_count": 0,
            "restored_count": 0,
            "dry_run": dry_run,
            "message": "",
        }

        if backend == "sqlite":
            result["message"] = (
                "SQLite does not preserve backup tables after re-embed. "
                "Rollback is not available."
            )
            return result

        if backend == "postgresql":
            return self._rollback_postgres(result, dry_run)

        # --- DuckDB rollback ---
        return self._rollback_duckdb(result, dry_run)

    def _rollback_duckdb(self, result: dict, dry_run: bool) -> dict:
        backup_table = result["backup_table"]
        conn = self.connection

        try:
            backup_exists = conn.sql(
                "SELECT COUNT(*) FROM information_schema.tables "
                f"WHERE table_name = '{backup_table}'"
            ).fetchone()[0] > 0
        except Exception:
            backup_exists = False

        if not backup_exists:
            result["message"] = (
                f"Backup table '{backup_table}' does not exist. "
                "No rollback backup is available."
            )
            return result

        try:
            result["current_count"] = conn.sql(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()[0]
        except Exception:
            result["current_count"] = 0

        try:
            result["backup_count"] = conn.sql(
                f"SELECT COUNT(*) FROM {backup_table}"
            ).fetchone()[0]
        except Exception as e:
            result["message"] = f"Error reading backup table: {e}"
            return result

        if result["backup_count"] == 0:
            result["message"] = (
                f"Backup table '{backup_table}' exists but is empty. "
                "Refusing to rollback."
            )
            return result

        try:
            row = conn.sql(
                f"SELECT embedding_model FROM {backup_table} LIMIT 1"
            ).fetchone()
            if row:
                result["backup_model"] = row[0] or "unknown"
        except Exception:
            result["backup_model"] = "unknown"

        result["message"] = (
            f"Backup table '{backup_table}' found with "
            f"{result['backup_count']} records (current: {result['current_count']})."
        )

        if dry_run:
            result["success"] = True
            return result

        logger.info(
            f"Rolling back: swapping {backup_table} -> {self.table_name} "
            f"({result['backup_count']} records)"
        )

        try:
            try:
                conn.sql(f"DROP INDEX IF EXISTS {self.table_name}_index;")
            except Exception:
                pass

            for tbl in [
                f"fts_main_{self.table_name}",
                f"fts_data_{self.table_name}",
                f"fts_stats_{self.table_name}",
                f"fts_segments_{self.table_name}",
                f"fts_lists_{self.table_name}",
            ]:
                try:
                    conn.sql(f"DROP TABLE IF EXISTS {tbl};")
                except Exception:
                    pass

            swap_result = self._safe_swap_tables(self.table_name, backup_table, f"{self.table_name}_rollback_temp")
            if not swap_result["swapped"]:
                result["message"] = "Rollback swap failed"
                return result

            if result["backup_model"] and result["backup_model"] != "unknown":
                self.embedding_model = result["backup_model"]
                self._reconfigure_client()
                logger.info(f"Restored embedding model to '{result['backup_model']}'")

            result["restored_count"] = conn.sql(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()[0]
            result["success"] = True
            result["message"] = (
                f"Rollback complete. Restored {result['restored_count']} records "
                f"from backup. Embedding model set to '{result['backup_model']}'."
            )

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            result["message"] = f"Rollback failed: {e}"

        return result

    def _rollback_postgres(self, result: dict, dry_run: bool) -> dict:
        backup_table = result["backup_table"]

        with self.engine.connect() as conn:
            backup_exists = conn.execute(text(
                f"SELECT COUNT(*) FROM pg_tables WHERE tablename = '{backup_table}'"
            )).scalar() > 0

        if not backup_exists:
            result["message"] = (
                f"Backup table '{backup_table}' does not exist. "
                "No rollback backup is available. "
                "(Backups only exist after dimension-changing re-embeds.)"
            )
            return result

        with self.engine.connect() as conn:
            try:
                result["current_count"] = conn.execute(
                    text(f"SELECT COUNT(*) FROM {self.table_name}")
                ).scalar() or 0
            except Exception:
                result["current_count"] = 0

            try:
                result["backup_count"] = conn.execute(
                    text(f"SELECT COUNT(*) FROM {backup_table}")
                ).scalar() or 0
            except Exception as e:
                result["message"] = f"Error reading backup table: {e}"
                return result

        if result["backup_count"] == 0:
            result["message"] = (
                f"Backup table '{backup_table}' exists but is empty. "
                "Refusing to rollback."
            )
            return result

        with self.engine.connect() as conn:
            try:
                row = conn.execute(
                    text(f"SELECT embedding_model FROM {backup_table} LIMIT 1")
                ).fetchone()
                if row:
                    result["backup_model"] = row[0] or "unknown"
            except Exception:
                result["backup_model"] = "unknown"

        result["message"] = (
            f"Backup table '{backup_table}' found with "
            f"{result['backup_count']} records (current: {result['current_count']})."
        )

        if dry_run:
            result["success"] = True
            return result

        logger.info(
            f"Rolling back PostgreSQL: swapping {backup_table} -> {self.table_name}"
        )

        try:
            temp_drop = f"{self.table_name}_rollback_temp"
            swap_result = self._safe_swap_tables(self.table_name, backup_table, temp_drop)
            if not swap_result["swapped"]:
                result["message"] = "Rollback swap failed"
                return result

            if result["backup_model"] and result["backup_model"] != "unknown":
                self.embedding_model = result["backup_model"]
                self._reconfigure_client()
                logger.info(f"Restored embedding model to '{result['backup_model']}'")

            with self.engine.connect() as conn:
                result["restored_count"] = conn.execute(
                    text(f"SELECT COUNT(*) FROM {self.table_name}")
                ).scalar() or 0

            result["success"] = True
            result["message"] = (
                f"Rollback complete. Restored {result['restored_count']} records "
                f"from backup. Embedding model set to '{result['backup_model']}'."
            )

        except Exception as e:
            logger.error(f"PostgreSQL rollback failed: {e}")
            result["message"] = f"Rollback failed: {e}"

        return result

    def _reconfigure_client(self):
        """Reconfigure the LLM client for the current embedding_model."""
        if "gemini" in self.embedding_model.lower():
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.client = ollama.Client(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )

    def verify_and_fix(
        self,
        collection_name: str = "",
        dry_run: bool = True,
        checks: list[str] | None = None,
        auto_finalize: bool = False,
    ) -> dict:
        """
        Verify embedding data integrity and optionally fix issues.

        :param collection_name: Collection to verify (empty = all)
        :param dry_run: Preview only, do not apply fixes
        :param checks: Specific checks to run (None = all)
        :param auto_finalize: If True, finalize any orphaned shadow collections found
        :return: Verification report
        """
        ALL_CHECKS = [
            "duplicate_hashes",
            "hash_integrity",
            "missing_models",
            "mixed_models",
            "dimension_consistency",
            "partial_documents",
            "orphaned_shadows",
            "orphaned_tables",
            "stale_backups",
            "empty_embeddings",
            "duplicate_doc_pages",
        ]
        active_checks = checks if checks else ALL_CHECKS
        tbl = self._active_table()

        result = {
            "collection": collection_name or "(all)",
            "dry_run": dry_run,
            "table": tbl,
            "checks": [],
            "summary": {"errors": 0, "warnings": 0, "info": 0, "fixes_applied": 0},
            "errors": [],
            "finalized": [],
        }

        col_filter_sql = f"WHERE collection_name = '{collection_name}'" if collection_name else ""

        for check_name in active_checks:
            handler = getattr(self, f"_verify_{check_name}", None)
            if handler is None:
                continue
            try:
                check_result = handler(tbl, col_filter_sql, dry_run, collection_name)
                result["checks"].append(check_result)
                sev = check_result.get("severity", "warning")
                if sev == "error":
                    result["summary"]["errors"] += 1
                elif sev == "warning":
                    result["summary"]["warnings"] += 1
                else:
                    result["summary"]["info"] += 1
                if check_result.get("fix_applied"):
                    result["summary"]["fixes_applied"] += check_result["fix_applied"]
            except Exception as e:
                result["errors"].append(f"{check_name}: {e}")
                logger.error(f"Verify check '{check_name}' failed: {e}")

        if auto_finalize and not dry_run:
            result["finalized"] = self._finalize_orphaned_shadows()

        return result

    def _finalize_orphaned_shadows(self) -> list[dict]:
        """Find and finalize all orphaned shadow collections across all tables."""
        finalized = []
        tables = self._get_all_embedding_tables()

        for tbl in tables:
            if "_backup" in tbl or tbl.endswith("_fts"):
                continue
            try:
                rows = self._verify_query(tbl,
                    f"SELECT DISTINCT collection_name FROM {tbl} WHERE collection_name LIKE '%_v2'"
                )
                shadows = [r[0] for r in rows]
            except Exception:
                continue

            for shadow in shadows:
                original = shadow[:-3]
                shadow_table = tbl if tbl != self.table_name else None
                logger.info(f"Auto-finalizing shadow '{shadow}' -> '{original}' in table '{tbl}'")
                try:
                    stats = self.finalize_rechunk(
                        collection_name=original,
                        shadow_collection=shadow,
                        shadow_table=shadow_table,
                    )
                    finalized.append(stats)
                except Exception as e:
                    logger.error(f"Failed to finalize '{shadow}': {e}")
                    finalized.append({
                        "success": False,
                        "collection": original,
                        "shadow_collection": shadow,
                        "errors": [str(e)],
                    })

        return finalized

    def _verify_query(self, tbl: str, sql: str) -> list[tuple]:
        if self.dburl.startswith("sqlite"):
            cur = self.connection.cursor()
            return cur.execute(sql).fetchall()
        elif self.dburl.startswith("duckdb"):
            return self.connection.sql(sql).fetchall()
        else:
            with self.engine.connect() as conn:
                return conn.execute(text(sql)).fetchall()

    def _verify_execute(self, tbl: str, sql: str):
        if self.dburl.startswith("sqlite"):
            cur = self.connection.cursor()
            cur.execute(sql)
            self.connection.commit()
        elif self.dburl.startswith("duckdb"):
            self.connection.sql(sql)
        else:
            with self.engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()

    def _verify_duplicate_hashes(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "duplicate_hashes", "severity": "error", "count": 0, "details": [], "fix_applied": None}
        rows = self._verify_query(tbl,
            f"SELECT collection_name, doc_hash, COUNT(*) as cnt FROM {tbl} "
            f"{col_filter or ''} "
            f"{'AND' if col_filter else 'WHERE'} doc_hash IS NOT NULL "
            f"GROUP BY collection_name, doc_hash HAVING COUNT(*) > 1"
        )
        check["count"] = len(rows)
        check["details"] = [f"{r[0]}/{r[1]} (x{r[2]})" for r in rows[:10]]

        if not dry_run and rows:
            total_deleted = 0
            for col, doc_hash, cnt in rows:
                if self.dburl.startswith("sqlite"):
                    cur = self.connection.cursor()
                    cur.execute(
                        f"DELETE FROM {tbl} WHERE rowid NOT IN "
                        f"(SELECT MIN(rowid) FROM {tbl} WHERE collection_name=? AND doc_hash=? LIMIT 1)",
                        (col, doc_hash),
                    )
                    total_deleted += cur.rowcount
                    self.connection.commit()
                elif self.dburl.startswith("duckdb"):
                    self.connection.sql(
                        f"DELETE FROM {tbl} WHERE id NOT IN "
                        f"(SELECT MIN(id) FROM {tbl} WHERE collection_name='{col}' AND doc_hash='{doc_hash}')"
                    )
                else:
                    with self.engine.connect() as conn:
                        r = conn.execute(text(
                            f"DELETE FROM {tbl} WHERE id NOT IN "
                            f"(SELECT MIN(id) FROM {tbl} WHERE collection_name=:col AND doc_hash=:hash)"
                        ), {"col": col, "hash": doc_hash})
                        total_deleted += r.rowcount
                        conn.commit()
            check["fix_applied"] = total_deleted

        if check["count"] == 0:
            check["severity"] = "info"
        return check

    def _verify_hash_integrity(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "hash_integrity", "severity": "error", "count": 0, "details": [], "fix_applied": None}
        rows = self._verify_query(tbl, f"SELECT id, doc_hash, document FROM {tbl} {col_filter} LIMIT 500")
        bad = []
        for row in rows:
            rid, stored_hash, document = row[0], row[1], row[2]
            if document is None:
                continue
            expected = sha256(document.encode()).hexdigest()
            if stored_hash != expected:
                bad.append((rid, stored_hash, expected, document[:50] if document else ""))
        check["count"] = len(bad)
        check["details"] = [f"id={b[0]}: stored={b[1][:12]}... expected={b[2][:12]}..." for b in bad[:10]]

        if not dry_run and bad:
            for rid, _, expected, _ in bad:
                self._verify_execute(tbl, f"UPDATE {tbl} SET doc_hash = '{expected}' WHERE id = {rid}")
            check["fix_applied"] = len(bad)

        if check["count"] == 0:
            check["severity"] = "info"
        return check

    def _verify_missing_models(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "missing_models", "severity": "warning", "count": 0, "details": [], "fix_applied": None}
        rows = self._verify_query(tbl,
            f"SELECT COUNT(*) FROM {tbl} {col_filter} "
            f"{'AND' if col_filter else 'WHERE'} (embedding_model IS NULL OR embedding_model = '' OR embedding_model = 'unknown')"
        )
        check["count"] = rows[0][0] if rows else 0

        if not dry_run and check["count"] > 0:
            self._verify_execute(tbl,
                f"UPDATE {tbl} SET embedding_model = '{self.embedding_model}' "
                f"WHERE embedding_model IS NULL OR embedding_model = '' OR embedding_model = 'unknown'"
            )
            check["fix_applied"] = check["count"]

        if check["count"] > 0:
            check["details"] = [f"{check['count']} rows with missing/unknown embedding_model"]
        else:
            check["severity"] = "info"
        return check

    def _verify_mixed_models(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "mixed_models", "severity": "warning", "count": 0, "details": [], "fix_applied": None}
        rows = self._verify_query(tbl,
            f"SELECT collection_name, embedding_model, COUNT(*) as cnt FROM {tbl} "
            f"{col_filter or ''} "
            f"{'AND' if col_filter else 'WHERE'} embedding_model IS NOT NULL "
            f"GROUP BY collection_name, embedding_model"
        )
        collections = {}
        for col, model, cnt in rows:
            collections.setdefault(col, []).append((model, cnt))

        mixed = {col: models for col, models in collections.items() if len(models) > 1}
        check["count"] = len(mixed)
        check["details"] = [
            f"{col}: {', '.join(f'{m} ({c})' for m, c in models)}"
            for col, models in list(mixed.items())[:10]
        ]
        if check["count"] == 0:
            check["severity"] = "info"
        return check

    def _verify_dimension_consistency(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "dimension_consistency", "severity": "error", "count": 0, "details": [], "fix_applied": None}
        expected_dim = self._get_embedding_dimension()
        table_dim = self._get_table_embedding_dimension()
        if table_dim is None:
            check["details"] = ["Could not detect table dimension"]
            check["severity"] = "info"
            return check
        if table_dim == expected_dim:
            check["severity"] = "info"
            return check
        check["count"] = 1
        check["details"] = [f"Table dimension ({table_dim}) != model dimension ({expected_dim})"]
        return check

    def _verify_partial_documents(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "partial_documents", "severity": "warning", "count": 0, "details": [], "fix_applied": None}
        rows = self._verify_query(tbl,
            f"SELECT collection_name, doc_name, MIN(page_number) as min_p, MAX(page_number) as max_p, COUNT(*) as cnt "
            f"FROM {tbl} {col_filter} "
            f"GROUP BY collection_name, doc_name"
        )
        partial = []
        for col, doc, min_p, max_p, cnt in rows:
            expected_pages = max_p - min_p + 1
            if cnt < expected_pages:
                partial.append((col, doc, cnt, expected_pages))
        check["count"] = len(partial)
        check["details"] = [f"{col}/{doc}: {cnt}/{exp} pages" for col, doc, cnt, exp in partial[:10]]
        if check["count"] == 0:
            check["severity"] = "info"
        return check

    def _verify_orphaned_shadows(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "orphaned_shadows", "severity": "warning", "count": 0, "details": [], "fix_applied": None}
        rows = self._verify_query(tbl,
            f"SELECT DISTINCT collection_name FROM {tbl} WHERE collection_name LIKE '%_v2'"
        )
        shadows = [r[0] for r in rows]
        check["count"] = len(shadows)
        check["details"] = shadows[:10]
        if check["count"] == 0:
            check["severity"] = "info"
        return check

    def _verify_orphaned_tables(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "orphaned_tables", "severity": "info", "count": 0, "details": [], "fix_applied": None}
        tables = self._get_all_embedding_tables()
        active = self.table_name
        dim_specific = [t for t in tables if t != active and "_backup" not in t and not t.endswith("_fts")]
        check["count"] = len(dim_specific)
        for t in dim_specific:
            try:
                cnt = self._verify_query(t, f"SELECT COUNT(*) FROM {t}")[0][0]
                check["details"].append(f"{t}: {cnt} rows")
            except Exception:
                check["details"].append(f"{t}: (error reading)")
        if check["count"] == 0:
            check["severity"] = "info"
        return check

    def _verify_stale_backups(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "stale_backups", "severity": "info", "count": 0, "details": [], "fix_applied": None}
        tables = self._get_all_embedding_tables()
        backups = [t for t in tables if "_backup" in t]
        check["count"] = len(backups)
        for t in backups:
            try:
                cnt = self._verify_query(t, f"SELECT COUNT(*) FROM {t}")[0][0]
                check["details"].append(f"{t}: {cnt} rows")
            except Exception:
                check["details"].append(f"{t}: (error reading)")
        return check

    def _verify_empty_embeddings(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "empty_embeddings", "severity": "error", "count": 0, "details": [], "fix_applied": None}
        if self.dburl.startswith("sqlite"):
            rows = self._verify_query(tbl, f"SELECT COUNT(*) FROM {tbl} {col_filter} WHERE document IS NULL OR document = ''")
        elif self.dburl.startswith("duckdb"):
            rows = self._verify_query(tbl, f"SELECT COUNT(*) FROM {tbl} {col_filter} {'AND' if col_filter else 'WHERE'} (document IS NULL OR document = '')")
        else:
            rows = self._verify_query(tbl, f"SELECT COUNT(*) FROM {tbl} {col_filter} {'AND' if col_filter else 'WHERE'} (document IS NULL OR document = '')")
        check["count"] = rows[0][0] if rows else 0
        if check["count"] > 0:
            check["details"] = [f"{check['count']} rows with empty document text"]
        else:
            check["severity"] = "info"
        return check

    def _verify_duplicate_doc_pages(self, tbl: str, col_filter: str, dry_run: bool, collection_name: str) -> dict:
        check = {"name": "duplicate_doc_pages", "severity": "warning", "count": 0, "details": [], "fix_applied": None}
        rows = self._verify_query(tbl,
            f"SELECT collection_name, doc_name, page_number, COUNT(*) as cnt FROM {tbl} "
            f"{col_filter or ''} "
            f"GROUP BY collection_name, doc_name, page_number HAVING COUNT(*) > 1"
        )
        check["count"] = len(rows)
        check["details"] = [f"{r[0]}/{r[1]} page {r[2]} (x{r[3]})" for r in rows[:10]]

        if not dry_run and rows:
            total_deleted = 0
            for col, doc, page, cnt in rows:
                if self.dburl.startswith("sqlite"):
                    cur = self.connection.cursor()
                    cur.execute(
                        f"DELETE FROM {tbl} WHERE rowid NOT IN "
                        f"(SELECT MIN(rowid) FROM {tbl} WHERE collection_name=? AND doc_name=? AND page_number=?)",
                        (col, doc, page),
                    )
                    total_deleted += cur.rowcount
                    self.connection.commit()
                elif self.dburl.startswith("duckdb"):
                    self.connection.sql(
                        f"DELETE FROM {tbl} WHERE id NOT IN "
                        f"(SELECT MIN(id) FROM {tbl} WHERE collection_name='{col}' AND doc_name='{doc}' AND page_number={page})"
                    )
                else:
                    with self.engine.connect() as conn:
                        r = conn.execute(text(
                            f"DELETE FROM {tbl} WHERE id NOT IN "
                            f"(SELECT MIN(id) FROM {tbl} WHERE collection_name=:col AND doc_name=:doc AND page_number=:page)"
                        ), {"col": col, "doc": doc, "page": page})
                        total_deleted += r.rowcount
                        conn.commit()
            check["fix_applied"] = total_deleted

        if check["count"] == 0:
            check["severity"] = "info"
        return check

    def list_backup_tables(self) -> list[dict]:
        """
        Discover backup embedding tables in the database.

        Queries the database catalog for tables matching known backup
        naming patterns (``%_backup%``) and returns metadata about each.
        """
        backups = []

        if self.dburl.startswith("duckdb"):
            conn = self.connection
            rows = conn.sql(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' "
                "AND table_name LIKE '%\\_backup%' ESCAPE '\\'"
            ).fetchall()
            for (tbl,) in rows:
                if tbl == self.table_name:
                    continue
                count = conn.sql(
                    f"SELECT COUNT(*) FROM {tbl}"
                ).fetchone()[0]
                model = ""
                try:
                    row = conn.sql(
                        f"SELECT embedding_model FROM {tbl} LIMIT 1"
                    ).fetchone()
                    if row:
                        model = row[0] or ""
                except Exception:
                    pass
                backups.append({
                    "table_name": tbl,
                    "row_count": count,
                    "embedding_model": model,
                })

        elif self.dburl.startswith("sqlite"):
            conn = self.connection
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name LIKE '%_backup%'"
            )
            for (tbl,) in cursor.fetchall():
                if tbl == self.table_name:
                    continue
                cursor.execute(f"SELECT COUNT(*) FROM \"{tbl}\"")
                count = cursor.fetchone()[0]
                model = ""
                try:
                    cursor.execute(
                        f"SELECT embedding_model FROM \"{tbl}\" LIMIT 1"
                    )
                    row = cursor.fetchone()
                    if row:
                        model = row[0] or ""
                except Exception:
                    pass
                backups.append({
                    "table_name": tbl,
                    "row_count": count,
                    "embedding_model": model,
                })

        else:
            # PostgreSQL
            engine = self.engine
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' "
                        "AND table_name LIKE '%\\_backup%'"
                    )
                )
                for (tbl,) in result:
                    if tbl == self.table_name:
                        continue
                    count = conn.execute(
                        text(f"SELECT COUNT(*) FROM \"{tbl}\"")
                    ).scalar() or 0
                    model = ""
                    try:
                        row = conn.execute(
                            text(
                                f"SELECT embedding_model FROM \"{tbl}\" LIMIT 1"
                            )
                        ).fetchone()
                        if row:
                            model = row[0] or ""
                    except Exception:
                        pass
                    backups.append({
                        "table_name": tbl,
                        "row_count": count,
                        "embedding_model": model,
                    })

        return backups

    def list_backends(self) -> list[dict]:
        from libbydbot.settings import Settings
        settings = Settings()

        current_backend = "unknown"
        if self.dburl.startswith("sqlite"):
            current_backend = "sqlite"
        elif self.dburl.startswith("duckdb"):
            current_backend = "duckdb"
        else:
            current_backend = "postgresql"

        backends = []
        pg_url = settings.target_postgres_url
        backends.append({
            "name": "postgresql",
            "display_name": "PostgreSQL",
            "is_current": current_backend == "postgresql",
            "is_configured": bool(pg_url),
            "location": self._safe_location(pg_url) if pg_url else "",
        })
        backends.append({
            "name": "duckdb",
            "display_name": "DuckDB",
            "is_current": current_backend == "duckdb",
            "is_configured": True,
            "location": settings.target_duckdb_path,
        })
        backends.append({
            "name": "sqlite",
            "display_name": "SQLite",
            "is_current": current_backend == "sqlite",
            "is_configured": True,
            "location": settings.target_sqlite_path,
        })
        return backends

    @staticmethod
    def _safe_location(url: str) -> str:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.hostname:
                port = f":{parsed.port}" if parsed.port else ""
                path = parsed.path.lstrip("/") or ""
                return f"{parsed.hostname}{port}/{path}"
            return url
        except Exception:
            return ""

    def _detect_source_dimension(self) -> int:
        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT embedding FROM {self.table_name} LIMIT 1")
                row = cursor.fetchone()
                if row is None:
                    return self._get_embedding_dimension()
                import struct
                raw = row[0]
                return len(raw) // 4
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            result = conn.sql(f"SELECT embedding FROM {self.table_name} LIMIT 1").fetchone()
            if result is None:
                return self._get_embedding_dimension()
            return len(result[0])
        else:
            with Session(self.engine) as session:
                result = session.execute(
                    select(self.embedding.embedding).limit(1)
                ).scalars().first()
                if result is None:
                    return self._get_embedding_dimension()
                return len(result)

    def _read_vectors_batch(self, offset: int, limit: int, collection_name: str) -> list[dict]:
        if self.dburl.startswith("sqlite"):
            return self._read_vectors_sqlite(offset, limit, collection_name)
        elif self.dburl.startswith("duckdb"):
            return self._read_vectors_duckdb(offset, limit, collection_name)
        else:
            return self._read_vectors_postgres(offset, limit, collection_name)

    def _read_vectors_sqlite(self, offset: int, limit: int, collection_name: str) -> list[dict]:
        import struct
        with self.connection as conn:
            cursor = conn.cursor()
            if collection_name:
                cursor.execute(
                    f"SELECT rowid, collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding FROM {self.table_name} WHERE collection_name = ? ORDER BY rowid LIMIT ? OFFSET ?",
                    (collection_name, limit, offset),
                )
            else:
                cursor.execute(
                    f"SELECT rowid, collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding FROM {self.table_name} ORDER BY rowid LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            embedding_bytes = row[7]
            dim = len(embedding_bytes) // 4
            embedding = list(struct.unpack(f"{dim}f", embedding_bytes))
            results.append({
                "collection_name": row[1],
                "doc_name": row[2],
                "page_number": row[3],
                "doc_hash": row[4],
                "document": row[5],
                "embedding_model": row[6] or "unknown",
                "embedding": embedding,
            })
        return results

    def _read_vectors_duckdb(self, offset: int, limit: int, collection_name: str) -> list[dict]:
        conn = self.connection
        if collection_name:
            rows = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding FROM {self.table_name} WHERE collection_name = ? ORDER BY id LIMIT ? OFFSET ?",
                params=[collection_name, limit, offset],
            ).fetchall()
        else:
            rows = conn.sql(
                f"SELECT id, collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding FROM {self.table_name} ORDER BY id LIMIT ? OFFSET ?",
                params=[limit, offset],
            ).fetchall()

        results = []
        for row in rows:
            embedding = list(row[7]) if row[7] else []
            results.append({
                "collection_name": row[1],
                "doc_name": row[2],
                "page_number": row[3],
                "doc_hash": row[4],
                "document": row[5],
                "embedding_model": row[6] or "unknown",
                "embedding": embedding,
            })
        return results

    def _read_vectors_postgres(self, offset: int, limit: int, collection_name: str) -> list[dict]:
        with Session(self.engine) as session:
            if collection_name:
                statement = (
                    select(self.embedding)
                    .where(self.embedding.collection_name == collection_name)
                    .order_by(self.embedding.id)
                    .offset(offset)
                    .limit(limit)
                )
            else:
                statement = (
                    select(self.embedding)
                    .order_by(self.embedding.id)
                    .offset(offset)
                    .limit(limit)
                )
            docs = session.execute(statement).scalars().all()

        results = []
        for doc in docs:
            results.append({
                "collection_name": doc.collection_name,
                "doc_name": doc.doc_name,
                "page_number": doc.page_number,
                "doc_hash": doc.doc_hash,
                "document": doc.document,
                "embedding_model": doc.embedding_model or "unknown",
                "embedding": list(doc.embedding) if doc.embedding else [],
            })
        return results

    def _insert_existing_vector(self, record: dict) -> None:
        collection_name = record["collection_name"]
        doc_name = record["doc_name"]
        page_number = record["page_number"]
        doc_hash = record["doc_hash"]
        document = record["document"].replace("\x00", "\ufffd")
        embedding_model = record["embedding_model"]
        embedding = record["embedding"]

        if self.dburl.startswith("sqlite"):
            import struct
            embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
            with self.connection as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding_bytes),
                )
                cursor.execute(
                    f"INSERT INTO {self.table_name}_fts(rowid, document, doc_hash) VALUES (?, ?, ?)",
                    (cursor.lastrowid, document, doc_hash),
                )
                conn.commit()
        elif self.dburl.startswith("duckdb"):
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
            conn = self.connection
            conn.sql(
                f"""
                INSERT INTO {self.table_name} (collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                params=[collection_name, doc_name, page_number, doc_hash, document, embedding_model, embedding_str],
            )
        else:
            with Session(self.engine) as session:
                doc_vector = self.embedding(
                    collection_name=collection_name,
                    doc_name=doc_name,
                    page_number=page_number,
                    doc_hash=doc_hash,
                    document=document,
                    embedding_model=embedding_model,
                    embedding=embedding,
                )
                session.add(doc_vector)
                session.commit()

    def _rebuild_fts(self) -> None:
        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}_fts")
                    cursor.execute(f"""
                        CREATE VIRTUAL TABLE {self.table_name}_fts USING fts5(
                            document,
                            doc_hash UNINDEXED,
                            content='{self.table_name}',
                            content_rowid='rowid'
                        );
                    """)
                    cursor.execute(f"""
                        INSERT INTO {self.table_name}_fts(rowid, document, doc_hash)
                        SELECT rowid, document, doc_hash FROM {self.table_name}
                    """)
                    conn.commit()
                    logger.info("Rebuilt SQLite FTS index after migration")
                except Exception as e:
                    logger.warning(f"Failed to rebuild SQLite FTS: {e}")
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            try:
                conn.sql("INSTALL fts;")
                conn.sql("LOAD fts;")
                conn.sql(f"PRAGMA create_fts_index('{self.table_name}', 'id', 'document');")
                logger.info("Rebuilt DuckDB FTS index after migration")
            except Exception as e:
                logger.warning(f"Failed to rebuild DuckDB FTS: {e}")

    def _count_source_records(self, collection_name: str) -> int:
        if self.dburl.startswith("sqlite"):
            with self.connection as conn:
                cursor = conn.cursor()
                if collection_name:
                    cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE collection_name = ?", (collection_name,))
                else:
                    cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cursor.fetchone()[0]
        elif self.dburl.startswith("duckdb"):
            conn = self.connection
            if collection_name:
                result = conn.sql(f"SELECT COUNT(*) FROM {self.table_name} WHERE collection_name = ?", params=[collection_name]).fetchone()
            else:
                result = conn.sql(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()
            return result[0]
        else:
            with Session(self.engine) as session:
                if collection_name:
                    statement = select(func.count()).select_from(self.embedding).where(self.embedding.collection_name == collection_name)
                else:
                    statement = select(func.count()).select_from(self.embedding)
                return session.execute(statement).scalar()

    def migrate_backend(
        self,
        target_backend: str,
        collection_name: str = "",
        batch_size: int = 1000,
        dry_run: bool = False,
        resume: bool = False,
    ) -> dict:
        target_dburl = self.resolve_target_dburl(target_backend)
        source_backend = "postgresql"
        if self.dburl.startswith("sqlite"):
            source_backend = "sqlite"
        elif self.dburl.startswith("duckdb"):
            source_backend = "duckdb"

        if source_backend == target_backend:
            return {
                "success": False,
                "total": 0,
                "migrated": 0,
                "skipped": 0,
                "errors": [f"Source and target are the same backend ({source_backend})"],
                "source_backend": source_backend,
                "target_backend": target_backend,
                "source_dimension": 0,
                "target_dimension": 0,
                "message": "Cannot migrate to the same backend",
            }

        source_dim = self._detect_source_dimension()
        total = self._count_source_records(collection_name)

        logger.info(f"Starting migration: {source_backend} → {target_backend} ({total} records)")

        target_embedder = DocEmbedder(
            col_name="migration_temp",
            dburl=target_dburl,
            embedding_model=self.embedding_model,
        )
        target_dim = target_embedder._get_embedding_dimension()

        if source_dim != target_dim:
            return {
                "success": False,
                "total": total,
                "migrated": 0,
                "skipped": 0,
                "errors": [f"Dimension mismatch: source has {source_dim}-dim vectors but target expects {target_dim}-dim. Use reembed --rechunk to change models first."],
                "source_backend": source_backend,
                "target_backend": target_backend,
                "source_dimension": source_dim,
                "target_dimension": target_dim,
                "message": "Dimension mismatch — cannot migrate without re-embedding",
            }

        stats = {
            "success": True,
            "total": total,
            "migrated": 0,
            "skipped": 0,
            "errors": [],
            "source_backend": source_backend,
            "target_backend": target_backend,
            "source_dimension": source_dim,
            "target_dimension": target_dim,
            "message": "",
        }

        offset = 0
        processed = 0

        while offset < total:
            batch = self._read_vectors_batch(offset, batch_size, collection_name)
            if not batch:
                break

            for record in batch:
                if resume:
                    target_embedder.collection_name = record["collection_name"]
                    if target_embedder._check_existing(record["doc_hash"]):
                        stats["skipped"] += 1
                        processed += 1
                        continue

                if not dry_run:
                    try:
                        target_embedder._insert_existing_vector(record)
                        stats["migrated"] += 1
                    except Exception as e:
                        error_msg = f"Error migrating {record['doc_name']} page {record['page_number']}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)
                else:
                    stats["migrated"] += 1

                processed += 1

            offset += batch_size
            if processed % batch_size == 0 or processed >= total:
                logger.info(f"Progress: {processed}/{total} records processed")

        if not dry_run and stats["migrated"] > 0:
            target_embedder._rebuild_fts()

        stats["message"] = f"{'Would migrate' if dry_run else 'Migrated'} {stats['migrated']} records from {source_backend} to {target_backend}"
        logger.info(stats["message"])
        return stats

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
