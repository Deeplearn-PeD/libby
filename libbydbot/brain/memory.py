"""
This module implements LibbyDBot's memory system

LibbyDBot's memory system is a simple data model that persists data between sessions
"""
import os
from sqlmodel import SQLModel, Field, create_engine, Session, select
import datetime
import loguru

logger = loguru.logger


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


class History:
    def __init__(self, dburl: str = "sqlite:///memory.db"):
        self.engine = self.setup_db(dburl)
    def setup_db(self, dburl: str):
        """
        Setup the database
        :param dburl: database url
        :return:
        """
        engine = create_engine(dburl)
        SQLModel.metadata.create_all(engine)
        return engine


    def memorize(self, user_id: int, question: str, response: str, context: str):
        """
        Persist a chat in the database
        :param user_id:
        :param question:
        :param response:
        :param context:
        :return: memory object
        """
        with Session(self.engine) as session:
            stmt = select(User).where(User.id == user_id)
            memory = Memory(question=question, response=response, context=context, user_id=user_id)
            results = session.exec(stmt)
            if not results: # Does not memorize if user is not registered
                logger.warning(f"User {user_id} not found in the database")
                return
            session.add(memory)
            session.commit()
            session.refresh(memory)
            return memory
    def remember(self, user_id, since: datetime.datetime = None):
        with Session(self.engine) as session:
            if since:
                return session.query(Memory).filter(Memory.user_id == user_id, Memory.timestamp >= since).all()
            return session.query(Memory).filter(Memory.user_id == user_id).all()
