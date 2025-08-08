# app/models.py - Complete File with Fixed Agent Model
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Agent(Base):
    """Fixed Agent model with all required fields for registration"""
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    mobile = Column(String(20), nullable=False)
    dob = Column(String(20), nullable=False)  # Store as string for compatibility
    country = Column(String(50), nullable=False)
    gender = Column(String(20), nullable=False)
    password = Column(String(255), nullable=False)  # Store plain password for now
    hashed_password = Column(String(255), nullable=True)  # For future bcrypt implementation
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Admin(Base):
    """Admin user model"""
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TaskProgress(Base):
    """Task progress tracking for agents"""
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
    """Submitted form data from agents"""
    __tablename__ = "submitted_forms"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    task_id = Column(Integer, ForeignKey("task_progress.id"), nullable=True)
    image_filename = Column(String(255), nullable=True)
    form_data = Column(Text, nullable=False)  # JSON string of form data
    submitted_at = Column(DateTime, default=datetime.utcnow)

class AgentSession(Base):
    """Agent login session tracking"""
    __tablename__ = "agent_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    login_time = Column(DateTime, default=datetime.utcnow)
    logout_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)

class ImageAssignment(Base):
    """Image assignment tracking (if needed)"""
    __tablename__ = "image_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    image_filename = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="assigned")  # assigned, completed, skipped
