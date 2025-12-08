# import chromadb
import os
from glob import glob
from hashlib import sha256
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
from sqlalchemy import event, Column, Integer, String, Sequence, text, create_engine, select, Table, insert, func
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
    __tablename__ = 'embedding'
    __table_args__ = {'extend_existing': True}
    # id_seq = Sequence("id_seq", metadata=Base.metadata)
    id = Column(Integer, autoincrement=True, primary_key=True)
    collection_name = Column(String)
    doc_name = Column(String)
    page_number = Column(Integer)
    doc_hash = Column(String, unique=True)
    document = Column(String)
    embedding = Column(Vector(1024))






class DocEmbedder:
    def __init__(self, col_name, dburl: str = 'duckdb:///:memory:', embedding_model: str = 'gemini-embedding-001'):
        self.dburl = dburl if dburl is not None else os.getenv("PGURL")
        self.embedding_model = embedding_model
        self._connection = None

        # Configure Google AI if using Gemini model
        if "gemini" in embedding_model.lower():
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

        if self.dburl.startswith("sqlite"):
            self.table_name = 'embedding_sqlite'
            # For SQLite, use native connection instead of SQLAlchemy
            self._create_sqlite_table(self.connection.cursor())  # Create table if it doesn't exist
            self.engine = None  # No SQLAlchemy engine for SQLite
            self.embedding = None  # Will use direct SQL queries
        elif self.dburl.startswith("duckdb"):
            # For DuckDB, we'll use SQL directly via a connection
            self.table_name = 'embedding_duckdb'
            try:
                logger.info(f"Connecting to database with dburl: {self.dburl}")
                if ':memory:' in self.dburl:
                    self.engine = duckdb.connect()
                else:
                    self.engine = duckdb.connect(self.dburl.split(":///")[-1])
            except NoSuchModuleError as exc:
                logger.error(f"Invalid dburl string passed to DocEmbedder: \n{exc}")
                self.engine = duckdb.connect()
            self.embedding = None  # Not using SQLAlchemy ORM for DuckDB
            # Ensure vss extension is loaded, table and index exist
            if self._should_create_tables():
                self._init_duckdb()
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
                Base.metadata.create_all(self.engine, tables=[Base.metadata.sorted_tables[0]], checkfirst=True)

    def _init_duckdb(self):
        """
        Initialize DuckDB: install vss extension, create table if not exists,
        and create HNSW index.
        """
        dimension = self._get_embedding_dimension()
        with self.connection as conn:
            # Install and load vss extension
            conn.sql("INSTALL vss;")
            conn.sql("LOAD vss;")
            
            # Check if table exists
            result = conn.sql(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.table_name}';"
            ).fetchall()
            # result is a list of tuples; we check if count > 0
            if result[0][0] == 0:
                # Create table (DuckDB uses INTEGER PRIMARY KEY for auto‑increment)
                create_sql = f"""
                CREATE TABLE {self.table_name} (
                    id INTEGER PRIMARY KEY,
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

                if ":memory:" in self.dburl:
                    # Create HNSW index. only works with :memory: databases
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
                        "SELECT name FROM sqlite_schema WHERE type='table' AND name='embedding_sqlite';").fetchone()
                return result is None
            elif self.dburl.startswith("duckdb"):
                with self.connection as conn:
                    result = conn.sql(
                        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self.table_name}';"
                    ).fetchall()
                # result is a list of tuples; we check if count == 0
                return result[0][0] == 0
            else:
                # PostgreSQL
                with Session(self.engine) as session:
                    result = session.execute(text("SELECT tablename FROM pg_tables WHERE tablename='embedding';")).fetchone()
                    return result is None
        except Exception as e:
            logger.warning(f"Error checking if tables exist: {e}. Will attempt to create tables.")
            return True

    @property
    def connection(self):
        """
        Get database connection. For SQLite, returns the native connection.
        For DuckDB, returns the duckdb connection object (self.engine).
        For PostgreSQL, returns the SQLAlchemy engine.
        """
        if self.dburl.startswith("sqlite"):
            return self._get_sqlite_connection()
        elif self.dburl.startswith("duckdb"):
            # self.engine is already a duckdb connection
            return duckdb.connect() if ":memory" in self.dburl else duckdb.connect(self.dburl.split(":///")[-1])
        else: # PostgreSQL
            return self.engine

    def _get_sqlite_connection(self):
        dbpath = urlparse(self.dburl).path
        # Handle in-memory databases
        if dbpath == "/:memory:":
            connection = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            connection = sqlite3.connect(dbpath[1:], check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        return connection

    @property
    def embeddings_list(self):
        """Return list of available embedding table names."""
        if self.dburl.startswith("duckdb"):
            return ['embedding_duckdb']
        elif self.dburl.startswith("sqlite"):
            return ['embedding_sqlite']
        else:
            return ['embedding']


    def _create_sqlite_table(self, cursor):
        """
        Create the embedding table in SQLite database using native SQL
        """
        dimension = self._get_embedding_dimension()

        # Create the table with all necessary columns
        create_table_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING vec0(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_name TEXT,
            doc_name TEXT,
            page_number INTEGER,
            doc_hash TEXT UNIQUE,
            document TEXT,
            embedding float[{dimension}]
        );
        """
        try:
            cursor.execute(create_table_sql)
            # maybe commit?
            logger.info("Created SQLite embedding table with vector support.")
        except Exception as e:
            logger.error(f"Failed to create SQLite embedding table: {e}")
            raise
        # conn.close()

    def _get_embedding_dimension(self):
        """
        Get the embedding dimension based on the model
        """
        if self.embedding_model == 'gemini-embedding-001':
            return 1536
        else:  # mxbai-embed-large and other Ollama models
            return 1024

    def _generate_embedding(self, text: str):
        """
        Generate embedding using the specified model
        """
        if self.embedding_model == 'gemini-embedding-001':
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config = types.EmbedContentConfig(output_dimensionality=self._get_embedding_dimension(), task_type="retrieval_document")
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
                session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
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
                    logger.info(f"SQLite version: {sqlite_version}, vector extension version: {vec_version}")
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
                self._create_sqlite_table(cursor)
                q = f"SELECT * FROM {self.table_name} WHERE doc_hash='{hash}'"
                result = cursor.execute(q).fetchall()
            conn.commit()
            return result
        elif self.dburl.startswith("duckdb"):
            with self.connection as conn:
                if ":memory:" in self.dburl:
                    conn = self._init_duckdb()

                result = conn.sql(
                    f"SELECT * FROM {self.table_name} WHERE doc_hash = '{hash}'").fetchall()
            return result
        else:
            # PostgreSQL
            with Session(self.engine) as session:
                statement = select(self.embedding).where(self.embedding.doc_hash == hash)
                result = session.execute(statement).all()
            return result

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
            logger.info(f"Document {docname} page {page_number} already exists in the database, skipping.")
            return
        doctext = doctext.replace("\x00", "\uFFFD")
        embedding = self._generate_embedding(doctext)
        
        if self.dburl.startswith("sqlite"):
            # Use native SQLite operations
            import struct
            embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)

            with self.connection as conn:
                cursor = conn.cursor()
                if self._should_create_tables():
                    self._create_sqlite_table(cursor)
                try:
                    cursor.execute("""
                        INSERT INTO embedding_sqlite (doc_hash, doc_name, collection_name, page_number, document, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (document_hash, docname, self.collection_name, page_number, doctext, embedding_bytes))
                    conn.commit()
                except sqlite3.IntegrityError as e:
                    logger.warning(f"Document {docname} page {page_number} already exists in the database: {e}")
                except Exception as e:
                    logger.error(f"Error: {e} generated when attempting to embed the following text: \n{doctext}")
        elif self.dburl.startswith("duckdb"):
            # Use direct SQL for DuckDB
            dimension = self._get_embedding_dimension()
            # Convert embedding list to a string representation for DuckDB array literal
            embedding_str = '[' + ','.join(str(v) for v in embedding) + ']'
            with self.connection as conn:
                if ":memory:" in self.dburl:
                    conn = self._init_duckdb()
                try:
                    # Use conn.sql() for direct DuckDB SQL execution
                    conn.sql(f"""
                        INSERT INTO {self.table_name} (doc_hash, doc_name, collection_name, page_number, document, embedding)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, parameters=[
                        document_hash,
                        docname,
                        self.collection_name,
                        page_number,
                        doctext,
                        embedding_str
                    ])
                except Exception as e:
                    logger.warning(f"Document {docname} page {page_number} may already exist or error: {e}")
        else:
            # PostgreSQL
            with Session(self.engine) as session:
                doc_vector = self.embedding(
                    doc_hash=document_hash,
                    doc_name=docname,
                    collection_name=self.collection_name,
                    page_number=page_number,
                    document=doctext,
                    embedding=embedding)
                try:
                    session.add(doc_vector)
                    session.commit()
                except IntegrityError as e:
                    session.rollback()
                    logger.warning(f"Document {docname} page {page_number} already exists in the database: {e}")
                except ValueError as e:
                    logger.error(f"Error: {e} generated when attempting to embed the following text: \n{doctext}")
                    session.rollback()

    def embed_path(self, corpus_path: str):
        """
        Embed all documents in a path
        :param corpus_path:  path to a folder containing PDFs
        :return:
        """
        for d in glob(os.path.join(corpus_path, '*.pdf')):
            try:
                doc = fitz.open(d)
            except EmptyFileError:
                continue
            n = doc.name
            for page_number, page in enumerate(doc):
                text = page.get_text()
                if not text:
                    continue
                self.embed_text(text, n, page_number)

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
            query_embedding_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)
            with self.connection as conn:
                cursor = conn.cursor()
                if collection:
                    # Simple similarity search
                    result = cursor.execute(
                        "SELECT document FROM embedding_sqlite WHERE collection_name = ? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                        (collection, query_embedding_bytes, num_docs)
                    ).fetchall()
                else:
                    result = cursor.execute(
                        f"SELECT document FROM {self.table_name} WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                        (query_embedding_bytes, num_docs)
                    ).fetchall()
                pages = [row[0] for row in result]
        elif self.dburl.startswith("duckdb"):
            # Use direct SQL for DuckDB
            dimension = self._get_embedding_dimension()
            embedding_str = '[' + ','.join(str(v) for v in query_embedding) + ']'
            with self.connection as conn:
                if collection:
                    result = conn.sql(f"""
                        SELECT document 
                        FROM {self.table_name} 
                        WHERE collection_name = ? 
                        ORDER BY array_cosine_similarity(embedding, ?) 
                        LIMIT ?
                    """, parameters=[collection, embedding_str, num_docs]).fetchall()
                else:
                    result = conn.sql(f"""
                        SELECT document 
                        FROM {self.table_name} 
                        ORDER BY array_cosine_similarity(embedding, ?) 
                        LIMIT ?
                    """, parameters=[embedding_str, num_docs]).fetchall()
                pages = [row[0] for row in result]
        else:
            # PostgreSQL
            dimension = self._get_embedding_dimension()
            with Session(self.engine) as session:
                if collection:
                    statement = (
                        select(self.embedding.document).where(self.embedding.collection_name == collection)
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
            with self.connection as conn:
                result = conn.sql(
                    f"SELECT DISTINCT doc_name, collection_name FROM {self.table_name}"
                ).fetchall()
            return [(row[0], row[1]) for row in result]
        else:
            # PostgreSQL
            with Session(self.engine) as session:
                statement = select(self.embedding.doc_name, self.embedding.collection_name).distinct()
                result = session.execute(statement).fetchall()
                return [(row[0], row[1]) for row in result]

    def __del__(self):
        if self.dburl.startswith("sqlite") and self._connection:
            self._connection.close()
        # For DuckDB, self.engine is a duckdb connection, not an SQLAlchemy engine
        # It doesn't have a dispose() method, so we should close it if it exists
        if hasattr(self, 'engine') and self.engine:
            if self.dburl.startswith("duckdb"):
                self.engine.close()
            else:
                self.engine.dispose()
