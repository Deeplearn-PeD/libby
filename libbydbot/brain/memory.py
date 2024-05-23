"""
This module implements LibbyDBot's memory system

LibbyDBot's memory system is a simple data model that persists data between sessions
"""
import os
from sqlmodel import SQLModel, Field, create_engine, Session
import datetime


class User(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    name: str = Field(index=True)
    email: str = Field()
    password: str = Field()


class Memory(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, index=True)
    question: str = Field()
    context: str = Field()
    response: str = Field()
    user_id: int | None = Field(default=None, foreign_key="user.id")


engine = create_engine(os.getenv("PGURL"))
SQLModel.metadata.create_all(engine)

def memorize(user_id: int, question: str, response: str, context: str):
    """
    Persist a chat in the database
    :param user_id:
    :param question:
    :param response:
    :param context:
    :return: memory object
    """
    with Session(engine) as session:
        memory = Memory(question=question, response=response, context=context, user_id=user_id)
        session.add(memory)
        session.commit()
        session.refresh(memory)
        return memory
def remember(user_id, since: datetime.datetime = None):
    with Session(engine) as session:
        if since:
            return session.query(Memory).filter(Memory.user_id == user_id, Memory.timestamp >= since).all()
        return session.query(Memory).filter(Memory.user_id == user_id).all()
