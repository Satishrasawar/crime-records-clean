from sqlalchemy import Column, String, Integer, Date, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    mobile = Column(String, nullable=False)
    dob = Column(String)
    country = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    submitted_forms = relationship("SubmittedForm", back_populates="agent")
    task_progress = relationship("TaskProgress", back_populates="agent", uselist=False)
    login_sessions = relationship("AgentSession", back_populates="agent")

class SubmittedForm(Base):
    __tablename__ = "submitted_forms"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=False)
    form_data = Column(Text, nullable=False)  # JSON string containing all form data
    submitted_at = Column(DateTime, default=datetime.utcnow)

    agent = relationship("Agent", back_populates="submitted_forms")

class TaskProgress(Base):
    __tablename__ = "task_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), unique=True)
    current_index = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    agent = relationship("Agent", back_populates="task_progress")

# NEW: Agent Session Tracking
class AgentSession(Base):
    __tablename__ = "agent_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=False)
    login_time = Column(DateTime, default=datetime.utcnow)
    logout_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Float, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    agent = relationship("Agent", back_populates="login_sessions")

# Optional: Track individual image assignments
class ImageAssignment(Base):
    __tablename__ = "image_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"))
    image_filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    is_completed = Column(String, default="pending")  # pending, completed, skipped
    
    # This can help track which specific images were assigned to which agents
