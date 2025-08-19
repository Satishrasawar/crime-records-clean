# app/models.py - COMPLETE FIXED VERSION
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Agent(Base):
    __tablename__ = "agents"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(10), unique=True, index=True, nullable=False)  # AGT123456
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    mobile = Column(String(20), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)  # Plain text for now
    status = Column(String(20), default="active", nullable=False)  # active, inactive, pending, suspended
    
    # Optional demographic fields (for compatibility with admin form)
    dob = Column(Date, nullable=True)  # Date of birth
    country = Column(String(100), nullable=True)
    gender = Column(String(10), nullable=True)  # Male, Female, Other
    
    # Timestamp fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional tracking fields
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<Agent(agent_id='{self.agent_id}', name='{self.name}', email='{self.email}')>"

class Admin(Base):
    """Admin model for admin login"""
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TaskProgress(Base):
    """Task progress tracking"""
    __tablename__ = "task_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    image_filename = Column(String(255), nullable=True)
    image_path = Column(String(500), nullable=True)
    status = Column(String(20), default="pending")  # pending, in_progress, completed, skipped
    current_index = Column(Integer, default=0)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SubmittedForm(Base):
    """Submitted form data"""
    __tablename__ = "submitted_forms"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    task_id = Column(Integer, ForeignKey("task_progress.id"), nullable=True)
    image_filename = Column(String(255), nullable=True)
    form_data = Column(Text, nullable=False)  # JSON string
    submitted_at = Column(DateTime, default=datetime.utcnow)

class AgentSession(Base):
    """Agent session tracking"""
    __tablename__ = "agent_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    login_time = Column(DateTime, default=datetime.utcnow)
    logout_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
