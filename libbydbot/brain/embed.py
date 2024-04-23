import ollama
# import chromadb
import os
import pgvector
from hashlib import sha256
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.ext.automap import automap_base
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
        id = Column(Integer, autoincrement=True, primary_key=True)
        doc_name = Column(String)
        page_number = Column(Integer)
        hash = Column(String, unique=True)
        document = Column(String)
        embedding = Column(Vector(1024))
    return Table



class DocEmbedder:
    def __init__(self, name="embeddings"):
        self.engine = create_engine(os.getenv("PGURL"))
        self.session = Session(self.engine)
        self.schema = table_factory(name)
        # if not self.engine.dialect.has_schema(self.engine, name):
        Base.metadata.create_all(self.engine, checkfirst=True)

    @property
    def embeddings_list(self):
        embedding_list = Base.metadata.tables.keys()
        return embedding_list

    def set_schema(self, name):
        metadata = MetaData()
        metadata.reflect(self.engine)
        self.schema = metadata.tables[name]

    def embed_text(self, doctext: object, docname:str, page_number: object) -> object:
        doctext = doctext.replace("\x00", "\uFFFD")
        response = ollama.embeddings(model="mxbai-embed-large", prompt=doctext)
        embedding = response["embedding"]
        # print(len(embedding))
        docv = self.schema(
            hash=sha256(doctext.encode()).hexdigest(),
            doc_name=docname,
            page_number=page_number,
            document=doctext,
            embedding=embedding)
        try:
            self.session.add(docv)
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            logger.warning(f"Document {docname} page {page_number} already exists in the database.")
        except ValueError as e:
            logger.error(f"Error: {e} generated when attempting to embed the following text: \n{doctext}")
            self.session.rollback()

    def retrieve_docs(self, query):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
        # print(dir(self.schema.columns))
        statement = (
            select(self.schema.document)
            .order_by(self.schema.embedding.l2_distance(response["embedding"]))
            .limit(5)
        )
        pages = self.session.scalars(statement)
        data = "\n".join(pages)
        return data

    def generate_response(self, question):
        context = self.retrieve_docs(question)
        # print(context)
        response = ollama.generate(
            model="gemma",
            prompt=f"{question}",
            system=f"You are Libby D. Bot, a research Assistant, you should answer questions "
                   f"based on the context provided below.\n{context}"
        )
        return response["response"]

    def __del__(self):
        self.session.close()



if __name__ == '__main__':
    embed_text()
