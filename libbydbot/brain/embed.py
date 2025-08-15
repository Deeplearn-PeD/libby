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

# DuckDB does not support vector types natively, so we will use a different class for DuckDB
class EmbeddingDuckdb(Base):
    __tablename__ = 'embedding_duckdb'
    __table_args__ = {'extend_existing': True}
    user_id_seq = Sequence('user_id_seq')
    id = Column(Integer, user_id_seq, server_default=user_id_seq.next_value(), primary_key=True)
    collection_name = Column(String)
    doc_name = Column(String)
    page_number = Column(Integer)
    doc_hash = Column(String, unique=True)
    document = Column(String)
    # embedding = Column(Vector(1024, dimensions=1024)) # DuckDB does not support vector types natively




class DocEmbedder:
    def __init__(self, col_name, dburl: str = 'duckdb:///:memory:', embedding_model: str = 'mxbai-embed-large'):
        self.dburl = dburl if dburl is not None else os.getenv("PGURL")
        self.embedding_model = embedding_model

        # Configure Google AI if using Gemini model
        if "gemini" in embedding_model.lower():
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

        if self.dburl.startswith("sqlite"):
            # For SQLite, use native connection instead of SQLAlchemy
            self.sqlite_connection = self._get_sqlite_connection()
            self.engine = None  # No SQLAlchemy engine for SQLite
            self.embedding = None  # Will use direct SQL queries
        else:
            try:
                logger.info(f"Connecting to database with dburl: {self.dburl}")
                self.engine = create_engine(self.dburl)
            except NoSuchModuleError as exc:
                logger.error(f"Invalid dburl string passed to DocEmbedder: \n{exc}")
                self.engine = create_engine("sqlite:///data/embedding.db")  # Fallback to in-memory DuckDB
                # raise exc
            
            if self.dburl.startswith("duckdb"):
                self.embedding = EmbeddingDuckdb
            else:
                self.embedding = Embedding

        self.collection_name = col_name
        
        # Check if tables exist and create them only if they don't exist
        if self._should_create_tables():
            if self.dburl.startswith("duckdb"):
                Base.metadata.create_all(self.engine, tables=[Base.metadata.sorted_tables[1]], checkfirst=True)
                # Add vector columns compatible with this engine
                self._add_duckdb_vector_column()
                Base.metadata.remove(Base.metadata.tables['embedding_duckdb'])
                self.embedding = Table('embedding_duckdb', Base.metadata, autoload_with=self.engine)
            elif self.dburl.startswith("sqlite"):
                self._create_sqlite_table()
            else:
                Base.metadata.create_all(self.engine, tables=[Base.metadata.sorted_tables[0]], checkfirst=True)

    def _should_create_tables(self):
        """
        Check if the appropriate embedding table exists in the database
        """
        try:
            if self.dburl.startswith("sqlite"):
                cursor = self.sqlite_connection.cursor()
                result = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_sqlite';").fetchone()
                return result is None
            else:
                with Session(self.engine) as session:
                    if self.dburl.startswith("duckdb"):
                        result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_duckdb';")).fetchone()
                    else:
                        # PostgreSQL
                        result = session.execute(text("SELECT tablename FROM pg_tables WHERE tablename='embedding';")).fetchone()
                    return result is None
        except Exception as e:
            logger.warning(f"Error checking if tables exist: {e}. Will attempt to create tables.")
            return True

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
        embedding_list = list(Base.metadata.tables.keys())
        return embedding_list

    def _add_duckdb_vector_column(self):
        """
        Add a vector column named embedding to the database
        """
        # Check if the column embedding already exists
        session = Session(self.engine)
        result = session.execute(text("SELECT * FROM information_schema.columns WHERE table_name = 'embedding_duckdb' AND column_name = 'embedding';")).first()
        if result is None:
            # If it does not exist, add the column
            logger.info("Adding vector column to DuckDB embedding table.")
            dimension = self._get_embedding_dimension()
            session.execute(text(f"ALTER TABLE embedding_duckdb ADD COLUMN embedding FLOAT[{dimension}];"))

        session.commit()

    def _create_sqlite_table(self):
        """
        Create the embedding table in SQLite database using native SQL
        """
        dimension = self._get_embedding_dimension()
        cursor = self.sqlite_connection.cursor()
        
        # Create the table with all necessary columns
        create_table_sql = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS embedding_sqlite  using vec0(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection_name TEXT,
            doc_name TEXT,
            page_number INTEGER,
            doc_hash TEXT UNIQUE,
            document TEXT,
            embedding float[{dimension}]
        );
        """
        cursor.execute(create_table_sql)
        
        self.sqlite_connection.commit()
        logger.info("Created SQLite embedding table with vector support.")


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
        session = Session(self.engine)
        if self.dburl.startswith("postgresql"):
            session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        elif self.dburl.startswith("duckdb"):
            session.execute(text('INSTALL vss;LOAD vss;'))
        elif self.dburl.startswith("sqlite"):
            try:
                sqlite_version, vec_version = session.execute(text("SELECT sqlite_version(), vec_version()")).fetchone()
                logger.info(f"SQLite version: {sqlite_version}, vector extension version: {vec_version}")
            except Exception as e:
                logger.error(f"Error checking vector extension: {e}")
        session.commit()

    def _check_existing(self, hash: str):
        """
        Check if a document with this hash already exists in the database
        :param hash: SHA256 hash of the document
        :return:
        """
        if self.dburl.startswith("sqlite"):
            cursor = self.sqlite_connection.cursor()
            result = cursor.execute("SELECT * FROM embedding_sqlite WHERE doc_hash = ?", (hash,)).fetchall()
            return result
        elif self.dburl.startswith("duckdb"):
            statement = select(self.embedding).where(self.embedding.c.doc_hash == hash)
        else:
            statement = select(self.embedding).where(self.embedding.doc_hash == hash)
        
        if not self.dburl.startswith("sqlite"):
            with Session(self.engine) as session:
                result = session.execute(statement).all()
            return result

    def embed_text(self, doctext: str, docname: str, page_number: str):
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
            
            cursor = self.sqlite_connection.cursor()
            try:
                cursor.execute("""
                    INSERT INTO embedding_sqlite (doc_hash, doc_name, collection_name, page_number, document, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (document_hash, docname, self.collection_name, page_number, doctext, embedding_bytes))
                self.sqlite_connection.commit()
            except sqlite3.IntegrityError as e:
                logger.warning(f"Document {docname} page {page_number} already exists in the database: {e}")
            except Exception as e:
                logger.error(f"Error: {e} generated when attempting to embed the following text: \n{doctext}")
        else:
            # Use SQLAlchemy for other databases
            with Session(self.engine) as session:
                if self.dburl.startswith("duckdb"):
                    doc_vector_insert = insert(self.embedding).values(
                        doc_hash=document_hash,
                        doc_name=docname,
                        collection_name=self.collection_name,
                        page_number=page_number,
                        document=doctext,
                        embedding=embedding)

                    session.execute(doc_vector_insert)
                    session.commit()
                else:
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
            cursor = self.sqlite_connection.cursor()
            
            if collection:
                # Simple similarity search
                result = cursor.execute(
                    "SELECT document FROM embedding_sqlite WHERE collection_name = ? AND embedding MATCH ? ORDER BY distance LIMIT ?",
                    (collection,query_embedding_bytes, num_docs)
                ).fetchall()
            else:
                result = cursor.execute(
                    "SELECT document FROM embedding_sqlite WHERE embedding MATCH ? ORDER BY distance  LIMIT ?",
                    (query_embedding_bytes, num_docs)
                ).fetchall()
            
            pages = [row[0] for row in result]
        else:
            # Use SQLAlchemy for other databases
            dimension = self._get_embedding_dimension()
            with Session(self.engine) as session:
                if collection:
                    if self.dburl.startswith("duckdb"):
                        query_text = f'select document from embedding_duckdb where collection_name = :collection_name order by array_cosine_similarity(embedding, CAST(:embedding as FLOAT[{dimension}])) limit :num_docs;'
                        query = text(query_text)
                        result = session.execute(query, {'collection_name': collection, 'embedding': query_embedding, 'num_docs': num_docs})
                        pages = [row[0] for row in result.fetchall()]
                    else:
                        # For PostgreSQL
                        statement = (
                            select(self.embedding.document).where(self.embedding.collection_name == collection)
                            .order_by(self.embedding.embedding.l2_distance(query_embedding))
                            .limit(num_docs)
                        )
                        pages = session.scalars(statement)
                else:
                    if self.dburl.startswith("duckdb"):
                        query_text = f'select document from embedding_duckdb order by array_cosine_similarity(embedding, CAST(:embedding as FLOAT[{dimension}])) limit :num_docs;'
                        query = text(query_text)
                        result = session.execute(query, {'embedding': query_embedding, 'num_docs': num_docs})
                        pages = [row[0] for row in result.fetchall()]
                    else:
                        # For PostgreSQL
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
            cursor = self._get_sqlite_connection().cursor()
            result = cursor.execute(
                "SELECT DISTINCT doc_name, collection_name FROM embedding_sqlite"
            ).fetchall()
            return [(row[0], row[1]) for row in result]
        else:
            with Session(self.engine) as session:
                if self.dburl.startswith("duckdb"):
                    statement = select(self.embedding.c.doc_name, self.embedding.c.collection_name).distinct()
                else:
                    statement = select(self.embedding.doc_name, self.embedding.collection_name).distinct()
                
                result = session.execute(statement).fetchall()
                return [(row[0], row[1]) for row in result]

    def __del__(self):
        if self.dburl.startswith("sqlite") and hasattr(self, 'sqlite_connection'):
            self.sqlite_connection.close()
        elif hasattr(self, 'engine') and self.engine:
            with Session(self.engine) as session:
                session.close()
