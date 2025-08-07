from sqlalchemy import Column, String, Integer, Date, DateTime, Text, ForeignKey, Float, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    mobile = Column(String, nullable=False)
    dob = Column(String)  # Keep as String for compatibility
    country = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    password = Column(String, nullable=False)  # Direct password storage (current system)
    hashed_password = Column(String, nullable=True)  # Keep for backwards compatibility
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    submitted_forms = relationship("SubmittedForm", back_populates="agent")
    login_sessions = relationship("AgentSession", back_populates="agent")

class TaskProgress(Base):
    __tablename__ = "task_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, nullable=False)  # Remove ForeignKey constraint for flexibility
    image_filename = Column(String, nullable=False)  # Required for upload system
    image_path = Column(String, nullable=False)      # Required for upload system
    status = Column(String, default="pending")       # pending, in_progress, completed
    assigned_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Keep old fields for compatibility
    current_index = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SubmittedForm(Base):
    __tablename__ = "submitted_forms"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), nullable=False)
    task_id = Column(Integer, nullable=True)  # Link to TaskProgress
    image_filename = Column(String, nullable=True)  # Track which image
    form_data = Column(JSON, nullable=False)  # Store as JSON object, not text
    submitted_at = Column(DateTime, default=datetime.utcnow)

    agent = relationship("Agent", back_populates="submitted_forms")

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

class ImageAssignment(Base):
    __tablename__ = "image_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"))
    image_filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    is_completed = Column(String, default="pending")  # pending, completed, skipped

class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
