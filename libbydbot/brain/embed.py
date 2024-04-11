import ollama
# import chromadb
import os
import pgvector
from hashlib import sha256
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import Column, Integer, String, text, create_engine, select
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


class DocVector(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, autoincrement=True, primary_key=True)
    doc_name = Column(String)
    page_number = Column(Integer)
    hash = Column(String, unique=True)
    document = Column(String)
    embedding = Column(Vector(1024))

class DocEmbedder:
    def __init__(self):
        self.engine = create_engine(os.getenv("PGURL"))
        self.session = Session(self.engine)
        Base.metadata.create_all(self.engine)

    def embed_text(self, doctext: object, docname:str, page_number: object) -> object:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=doctext)
        embedding = response["embedding"]
        # print(len(embedding))
        docv = DocVector(
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


    def retrieve_docs(self, query):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
        statement = (
            select(DocVector.document)
            .order_by(DocVector.embedding.l2_distance(response["embedding"]))
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
            prompt=f"Using this context: {context} \n\nAnswer this question: {question}",
            system="You are Libby D. Bot, a research Assistant, you should answer questions based on the context provided."
        )
        return response["response"]

    def __del__(self):
        self.session.close()



if __name__ == '__main__':
    embed_text()
