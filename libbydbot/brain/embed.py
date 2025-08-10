# import chromadb
import os
from glob import glob
from hashlib import sha256

import dotenv
import fitz
import loguru
import ollama
from google import genai
from google.genai import types
from fitz import EmptyFileError
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Sequence, text, create_engine, select, Table, insert, func
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

# SQLite with sqlite-vec extension
class EmbeddingSqlite(Base):
    __tablename__ = 'embedding_sqlite'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, autoincrement=True, primary_key=True)
    collection_name = Column(String)
    doc_name = Column(String)
    page_number = Column(Integer)
    doc_hash = Column(String, unique=True)
    document = Column(String)
    # embedding will be handled as BLOB for sqlite-vec


class DocEmbedder:
    def __init__(self, col_name, dburl: str = 'duckdb:///:memory:', create=True, embedding_model: str = 'mxbai-embed-large'):
        self.dburl = dburl if dburl is not None else os.getenv("PGURL")
        self.embedding_model = embedding_model
        
        # Configure Google AI if using Gemini model
        if "gemini" in embedding_model.lower():
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            self.client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

        try:
            logger.info(f"Connecting to database with dburl: {self.dburl}")
            self.engine = create_engine(self.dburl)
        except NoSuchModuleError as exc:
            logger.error(f"Invalid dburl string passed to DocEmbedder: \n{exc}")
            self.engine = create_engine("duckdb:///data/embedding.duckdb")  # Fallback to in-memory DuckDB
            # raise exc
        # self.session = Session(self.engine)
        self._check_vector_exists()
        if self.dburl.startswith("duckdb"):
            self.embedding = EmbeddingDuckdb
        elif self.dburl.startswith("sqlite"):
            self.embedding = EmbeddingSqlite
        else:
            self.embedding = Embedding
        self.collection_name = col_name
        if create:
            if self.dburl.startswith("duckdb"):
                Base.metadata.create_all(self.engine, tables=[Base.metadata.sorted_tables[1]], checkfirst=True)
                # Add vector columns compatible with this engine
                self._add_duckdb_vector_column()
                Base.metadata.remove(Base.metadata.tables['embedding_duckdb'])
                self.embedding = Table('embedding_duckdb', Base.metadata, autoload_with=self.engine)
            elif self.dburl.startswith("sqlite"):
                Base.metadata.create_all(self.engine, tables=[Base.metadata.sorted_tables[2]], checkfirst=True)
                # Add vector columns compatible with sqlite-vec
                self._add_sqlite_vector_column()
                Base.metadata.remove(Base.metadata.tables['embedding_sqlite'])
                self.embedding = Table('embedding_sqlite', Base.metadata, autoload_with=self.engine)
            else:
                Base.metadata.create_all(self.engine, tables=[Base.metadata.sorted_tables[0]], checkfirst=True)

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

    def _add_sqlite_vector_column(self):
        """
        Add a vector column named embedding to the SQLite database using sqlite-vec
        """
        session = Session(self.engine)
        # Check if the column embedding already exists
        result = session.execute(text("PRAGMA table_info(embedding_sqlite);")).fetchall()
        column_names = [row[1] for row in result]
        
        if 'embedding' not in column_names:
            # If it does not exist, add the column as BLOB for sqlite-vec
            logger.info("Adding vector column to SQLite embedding table.")
            session.execute(text("ALTER TABLE embedding_sqlite ADD COLUMN embedding BLOB;"))
        
        session.commit()


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
            # Load sqlite-vec extension
            try:
                session.execute(text(".load sqlite-vec"))
            except Exception:
                # Try alternative loading method
                session.execute(text("SELECT load_extension('sqlite-vec')"))
        session.commit()

    def _check_existing(self, hash: str):
        """
        Check if a document with this hash already exists in the database
        :param hash: SHA256 hash of the document
        :return:
        """
        if self.dburl.startswith("duckdb") or self.dburl.startswith("sqlite"):
            statement = select(self.embedding).where(self.embedding.c.doc_hash == hash)
        else:
            statement = select(self.embedding).where(self.embedding.doc_hash == hash)
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
        # print(len(embedding))
        with Session(self.engine) as session:
            if self.dburl.startswith("duckdb") or self.dburl.startswith("sqlite"):
                # Convert embedding to bytes for sqlite-vec if using SQLite
                if self.dburl.startswith("sqlite"):
                    import struct
                    embedding_bytes = struct.pack(f'{len(embedding)}f', *embedding)
                    embedding_value = embedding_bytes
                else:
                    embedding_value = embedding
                
                doc_vector_insert = insert(self.embedding).values(
                    doc_hash=document_hash,
                    doc_name=docname,
                    collection_name=self.collection_name,
                    page_number=page_number,
                    document=doctext,
                    embedding=embedding_value)

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
        # print(dir(self.schema.columns))
        dimension = self._get_embedding_dimension()
        with Session(self.engine) as session:
            if collection:
                if self.dburl.startswith("duckdb"):
                    query_text = f'select document from embedding_duckdb where collection_name = :collection_name order by array_cosine_similarity(embedding, CAST(:embedding as FLOAT[{dimension}])) limit :num_docs;'
                    query = text(query_text)
                    result = session.execute(query, {'collection_name': collection, 'embedding': query_embedding, 'num_docs': num_docs})
                    pages = [row[0] for row in result.fetchall()]
                elif self.dburl.startswith("sqlite"):
                    # Convert query embedding to bytes for sqlite-vec
                    import struct
                    query_embedding_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)
                    query_text = 'select document from embedding_sqlite where collection_name = :collection_name order by vec_distance_cosine(embedding, :embedding) limit :num_docs;'
                    query = text(query_text)
                    result = session.execute(query, {'collection_name': collection, 'embedding': query_embedding_bytes, 'num_docs': num_docs})
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
                elif self.dburl.startswith("sqlite"):
                    # Convert query embedding to bytes for sqlite-vec
                    import struct
                    query_embedding_bytes = struct.pack(f'{len(query_embedding)}f', *query_embedding)
                    query_text = 'select document from embedding_sqlite order by vec_distance_cosine(embedding, :embedding) limit :num_docs;'
                    query = text(query_text)
                    result = session.execute(query, {'embedding': query_embedding_bytes, 'num_docs': num_docs})
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
        with Session(self.engine) as session:
            if self.dburl.startswith("duckdb") or self.dburl.startswith("sqlite"):
                statement = select(self.embedding.c.doc_name, self.embedding.c.collection_name).distinct()
            else:
                statement = select(self.embedding.doc_name, self.embedding.collection_name).distinct()
            
            result = session.execute(statement).fetchall()
            return [(row[0], row[1]) for row in result]

    def __del__(self):
        with Session(self.engine) as session:
            session.close()
