# import chromadb
import os
from glob import glob
from hashlib import sha256

import dotenv
import fitz
import loguru
import ollama
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


class DocEmbedder:
    def __init__(self, col_name, dburl: str = None, create=True):
        self.dburl = dburl if dburl is not None else os.getenv("PGURL")
        try:
            if ':memory:' in self.dburl:
                self.engine = create_engine(self.dburl)
            else:
                self.engine = create_engine(self.dburl)
                # self.engine = create_engine(self.dburl.split('/')[-1])
        except NoSuchModuleError as exc:
            logger.error(f"Invalid dburl string passed to DocEmbedder: \n{exc}")
            raise exc
        self.session = Session(self.engine)
        self._check_vector_exists()
        if self.dburl.startswith("duckdb"):
            self.embedding = EmbeddingDuckdb
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
        result = self.session.execute(text("SELECT * FROM information_schema.columns WHERE table_name = 'embedding_duckdb' AND column_name = 'embedding';")).first()
        if result is None:
            # If it does not exist, add the column
            logger.info("Adding vector column to DuckDB embedding table.")
            self.session.execute(text("ALTER TABLE embedding_duckdb ADD COLUMN embedding FLOAT[1024];"))

        self.session.commit()


    def _check_vector_exists(self):
        """
        Check if the vector extension exists in the database
        """
        if self.dburl.startswith("postgresql"):
            self.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))

        elif self.dburl.startswith("duckdb"):
            self.session.execute(text('INSTALL vss;LOAD vss;'))
        self.session.commit()

    def _check_existing(self, hash: str):
        """
        Check if a document with this hash already exists in the database
        :param hash: SHA256 hash of the document
        :return:
        """
        if self.dburl.startswith("duckdb"):
            statement = select(self.embedding).where(self.embedding.c.doc_hash == hash)
        else:
            statement = select(self.embedding).where(self.embedding.doc_hash == hash)

        result = self.session.execute(statement).all()
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
        response = ollama.embeddings(model="mxbai-embed-large", prompt=doctext)
        embedding = response["embedding"]
        # print(len(embedding))
        if self.dburl.startswith("duckdb"):
            doc_vector_insert = insert(self.embedding).values(doc_name=docname,
                                                       collection_name=self.collection_name,
                                                       page_number=page_number,
                                                       document=doctext,
                                                       embedding=embedding)
            self.session.execute(doc_vector_insert)
            self.session.commit()
        else:
            doc_vector = self.embedding(
                doc_hash=document_hash,
                doc_name=docname,
                collection_name=self.collection_name,
                page_number=page_number,
                document=doctext,
                embedding=embedding)
            try:
                self.session.add(doc_vector)
                self.session.commit()
            except IntegrityError as e:
                self.session.rollback()
                logger.warning(f"Document {docname} page {page_number} already exists in the database: {e}")
            except ValueError as e:
                logger.error(f"Error: {e} generated when attempting to embed the following text: \n{doctext}")
                self.session.rollback()

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
        response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
        # print(dir(self.schema.columns))
        if collection:
            if self.dburl.startswith("duckdb"):
                query = text('select document from embedding_duckdb where collection_name = :collection_name order by array_cosine_similarity(embedding, CAST(:embedding as FLOAT[1024])) limit :num_docs;')
                result = self.session.execute(query, {'collection_name': collection, 'embedding': response["embedding"], 'num_docs': num_docs})
                pages = [row[0] for row in result.fetchall()]
                # statement = (
                #     select(self.embedding.c.document).where(self.embedding.c.collection_name == collection)
                #     .order_by(func.array_distance(self.embedding.c.embedding, response["embedding"]))
                #     .limit(num_docs)
                # )
            else:
                # For PostgreSQL
                statement = (
                    select(self.embedding.document).where(self.embedding.collection_name == collection)
                    .order_by(self.embedding.embedding.l2_distance(response["embedding"]))
                    .limit(num_docs)
                )
                pages = self.session.scalars(statement)
        else:
            if self.dburl.startswith("duckdb"):
                query = text('select document from embedding_duckdb order by array_cosine_similarity(embedding, CAST(:embedding as FLOAT[1024])) limit :num_docs;')
                result = self.session.execute(query, {'embedding': response["embedding"], 'num_docs': num_docs})
                pages = [row[0] for row in result.fetchall()]
                # statement = (
                #     select(self.embedding.c.document)
                #     .order_by(func.array_distance(self.embedding.c.embedding, response["embedding"]))
                #     .limit(num_docs)
                # )
            else:
                # For PostgreSQL
                statement = (
                    select(self.embedding.document)
                    .order_by(self.embedding.embedding.l2_distance(response["embedding"]))
                    .limit(num_docs)
                )
                pages = self.session.scalars(statement)
        data = "\n".join(pages)
        return data

    def __del__(self):
        self.session.close()
