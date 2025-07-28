from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AgentStatusUpdateSchema(BaseModel):
    status: str

class AgentCreateSchema(BaseModel):
    name: str
    email: str
    mobile: str
    dob: str
    country: str
    gender: str

class AgentResponseSchema(BaseModel):
    id: int
    agent_id: str
    name: str
    email: str
    status: str
    created_at: Optional[datetime]

    class Config:
        from_attributes = True

class TaskProgressSchema(BaseModel):
    agent_id: str
    current_index: int
    total_images: int
    progress_percentage: float

class SubmissionResponseSchema(BaseModel):
    message: str
    success: bool
    submission_id: int