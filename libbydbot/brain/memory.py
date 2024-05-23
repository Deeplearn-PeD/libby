"""
This module implements LibbyDBot's memory system

LibbyDBot's memory system is a simple data model that persists data between sessions
"""
import os
from sqlmodel import SQLModel, Field, create_engine
import datetime

class Memory(SQLModel, table=True):
    id: int = Field(primary_key=True, default=None)
    timestamp: datetime.datetime = Field(default=datetime.datetime.now)
    question: str = Field()
    context: str = Field()
    response: str = Field()

engine = create_engine(os.getenv("PGURL"))
SQLModel.metadata.create_all(engine)