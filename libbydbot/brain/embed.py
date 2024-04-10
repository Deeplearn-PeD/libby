import ollama
import chromadb
import pgvector
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
import dotenv

dotenv.load_dotenv()

# create a class to store the embeddings
class Base(DeclarativeBase):
    pass
class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    document = Column(String)
    embedding = Column(Vector)





def embed_docs(documents):
    # store each document in a vector embedding database
    for i, d in enumerate(documents):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[d]
        )

def retrieve_docs(query):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=query)
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=1
    )
    data = results['documents'][0][0]
    return data

def generate_response(question):
    context = retrieve_docs(question)
    print(context)
    response = ollama.generate(
        model="gemma",
        prompt=f"Using this context: {context} \n\nAnswer this question: {question}",
        system=context
    )
    return response["response"]