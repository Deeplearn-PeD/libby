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
    def __init__(self, col_name, create=True):
        self.engine = create_engine(os.getenv("PGURL"))
        self.session = Session(self.engine)
        self.embedding = Embedding
        self.collection_name = col_name
        if create:
            Base.metadata.create_all(self.engine, checkfirst=True)

    @property
    def embeddings_list(self):
        embedding_list = list(Base.metadata.tables.keys())
        return embedding_list

    # def set_schema(self, name):
    #     metadata = MetaData()
    #     metadata.reflect(self.engine, extend_existing=True)
    #     self.schema = metadata.tables[name]

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
        except IntegrityError:
            self.session.rollback()
            logger.warning(f"Document {docname} page {page_number} already exists in the database.")
        except ValueError as e:
            logger.error(f"Error: {e} generated when attempting to embed the following text: \n{doctext}")
            self.session.rollback()

    def retrieve_docs(self, query: str, collection: str="", num_docs: int = 5) -> str:
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
