# import chromadb
import os
from hashlib import sha256

import dotenv
import loguru
import ollama
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, text, create_engine, select
from sqlalchemy.exc import IntegrityError, NoSuchModuleError
from sqlalchemy.orm import DeclarativeBase, Session

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
    id = Column(Integer, autoincrement=True, primary_key=True)
    collection_name = Column(String)
    doc_name = Column(String)
    page_number = Column(Integer)
    doc_hash = Column(String, unique=True)
    document = Column(String)
    embedding = Column(Vector(1024))


class DocEmbedder:
    def __init__(self, col_name, dburl: str = None, create=True):
        self.dburl = dburl
        try:
            self.engine = create_engine(os.getenv("PGURL")) if dburl is None else create_engine(dburl)
        except NoSuchModuleError as exc:
            logger.error(f"Invalid dburl string passed to DocEmbedder: \n{exc}")
            raise exc
        self.session = Session(self.engine)
        self._check_vector_exists()
        self.embedding = Embedding
        self.collection_name = col_name
        if create:
            Base.metadata.create_all(self.engine, checkfirst=True)

    @property
    def embeddings_list(self):
        embedding_list = list(Base.metadata.tables.keys())
        return embedding_list

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
            statement = (
                select(self.embedding.document).where(self.embedding.collection_name == collection)
                .order_by(self.embedding.embedding.l2_distance(response["embedding"]))
                .limit(num_docs)
            )
        else:
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
