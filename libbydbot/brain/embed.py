import ollama
# import chromadb
import os
import pgvector
from hashlib import sha256
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import Column, Integer, String, text, create_engine, select, MetaData
from pgvector.sqlalchemy import Vector
import dotenv
import loguru

dotenv.load_dotenv()
logger = loguru.logger

engine = create_engine(os.getenv("PGURL"))
with Session(engine) as session:
    session.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))


# create a class to store the embeddings
class Base(DeclarativeBase):
    pass


def table_factory(name):
    """
    Create the schema of a Table
    :param name: name of the table
    :return:
    """

    class Table(Base):
        __tablename__ = name
        __table_args__ = {'extend_existing': True}
        id = Column(Integer, autoincrement=True, primary_key=True)
        doc_name = Column(String)
        page_number = Column(Integer)
        hash = Column(String, unique=True)
        document = Column(String)
        embedding = Column(Vector(1024))

    return Table


class DocEmbedder:
    def __init__(self, name="embeddings", create=False):
        self.engine = create_engine(os.getenv("PGURL"))
        self.session = Session(self.engine)
        self.schema = table_factory(name)
        # if not self.engine.dialect.has_schema(self.engine, name):
        if create:
            Base.metadata.create_all(self.engine, checkfirst=True)

    @property
    def embeddings_list(self):
        embedding_list = list(Base.metadata.tables.keys())
        return embedding_list

    def set_schema(self, name):
        metadata = MetaData()
        metadata.reflect(self.engine, extend_existing=True)
        self.schema = metadata.tables[name]

    def _check_existing(self, hash: str):
        """
        Check if a document with this hash already exists in the database
        :param hash: SHA256 hash of the document
        :return:
        """
        statement = select(self.schema).where(self.schema.hash == hash)
        result = self.session.execute(statement)
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
        doc_vector = self.schema(
            hash=document_hash,
            doc_name=docname,
            page_number=page_number,
            document=doctext,
            embedding=embedding)
        try:
            self.session.add(doc_vector)
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            logger.warning(f"Document {docname} page {page_number} already exists in the database.")
        except ValueError as e:
            logger.error(f"Error: {e} generated when attempting to embed the following text: \n{doctext}")
            self.session.rollback()

    def retrieve_docs(self, query: str, num_docs: int = 5) -> str:
        """
        Retrieve documents based on a query.
        :param query: query string
        :param num_docs: number of documents to retrieve
        :return: all documents as a string
        """
        response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
        # print(dir(self.schema.columns))
        statement = (
            select(self.schema.document)
            .order_by(self.schema.embedding.l2_distance(response["embedding"]))
            .limit(num_docs)
        )
        pages = self.session.scalars(statement)
        data = "\n".join(pages)
        return data

    def __del__(self):
        self.session.close()
