"""
Database Schemas for Rabbit AI

Each Pydantic model becomes a MongoDB collection with the lowercase class name.
Examples:
- Task -> "task"
- Note -> "note"
- Idea -> "idea"
- Message -> "message"
- Memory -> "memory"
- Setting -> "setting"
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class Task(BaseModel):
    title: str = Field(..., description="Task title")
    done: bool = Field(False, description="Completion status")
    ts: Optional[str] = Field(None, description="Client timestamp ISO")

class Note(BaseModel):
    body: str = Field(..., description="Note content")
    ts: Optional[str] = Field(None)

class Idea(BaseModel):
    body: str = Field(..., description="Idea content")
    ts: Optional[str] = Field(None)

class Summary(BaseModel):
    body: str = Field(..., description="Summary content")
    ts: Optional[str] = Field(None)

class Message(BaseModel):
    role: Literal['user','assistant','system']
    text: str
    ts: Optional[str] = None
    session: Optional[str] = Field(None, description="Conversation session id")

class Memory(BaseModel):
    key: str
    value: str

class Setting(BaseModel):
    active: str = Field('openai', description="Active provider id")
    keys: dict = Field(default_factory=dict, description="API keys per provider id")
    endpoint: Optional[str] = Field(None, description="Custom endpoint url")
    customPrompt: Optional[str] = Field(None, description="System prompt / personality")
