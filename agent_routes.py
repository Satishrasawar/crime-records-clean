from fastapi import APIRouter, Form, Depends, HTTPException, UploadFile, File, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime, timedelta, date
import os
import uuid
import secrets
import string
import re
import json
import zipfile
import io
import pandas as pd
import shutil
import aiofiles
from io import BytesIO
import logging
import hashlib

# Database and model imports
from app.database import get_db
from app.models import Agent, TaskProgress, SubmittedForm, AgentSession

router = APIRouter()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Utility functions
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number format"""
    if not mobile:
        return False
        
    # Remove all non-digit characters except leading +
    clean_mobile = re.sub(r'[^\d+]', '', mobile)
    
    # Check if it has a country code (starts with +) or is just digits
    if clean_mobile.startswith('+'):
        # International format: + followed by 10-15 digits
        return re.match(r'^\+\d{10,15}$', clean_mobile) is not None
    else:
        # Local format: 10-15 digits
        return re.match(r'^\d{10,15}$', clean_mobile) is not None

def generate_unique_agent_id(db: Session):
    """Generate a unique agent ID"""
    max_attempts = 100
    for attempt in range(max_attempts):
        agent_number = secrets.randbelow(900000) + 100000
        agent_id = f"AGT{agent_number}"
        existing = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not existing:
            return agent_id
    
    # Fallback to timestamp-based ID
    import time
    fallback_id = f"AGT{str(int(time.time()))[-6:]}"
    existing_fallback = db.query(Agent).filter(Agent.agent_id == fallback_id).first()
    if existing_fallback:
        raise HTTPException(status_code=500, detail="Failed to generate unique agent ID")
    return fallback_id

def generate_secure_password():
    """Generate a secure password ensuring variety"""
    characters = string.ascii_letters + string.digits + "!@#$%"
    password_parts = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%")
    ]
    for _ in range(6):  # Total length 10
        password_parts.append(secrets.choice(characters))
    secrets.SystemRandom().shuffle(password_parts)
    return ''.join(password_parts)

# ===================== AGENT REGISTRATION ENDPOINTS =====================

@router.post("/agents/register")
async def register_agent(
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Register a new agent with proper credentials"""
    try:
        logger.info(f"🆕 Agent registration attempt: {name}, {email}, {mobile}")
        
        # Clean inputs
        name = name.strip()
        email = email.strip().lower()
        mobile = re.sub(r'[\s\-\(\)]', '', mobile)
        
        logger.info(f"Cleaned inputs - Name: {name}, Email: {email}, Mobile: {mobile}")
        
        # Collect all validation errors
        validation_errors = []
        
        if not name or len(name) < 2:
            validation_errors.append("Name must be at least 2 characters long")
        
        if not validate_email(email):
            validation_errors.append("Invalid email format")
        
        if not validate_mobile(mobile):
            validation_errors.append(f"Invalid mobile number format: {mobile}. Use 10-15 digits")
        
        # Validate optional fields
        dob_date = None
        if dob:
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                age = (date.today() - dob_date).days // 365
                if age < 16 or age > 80:
                    validation_errors.append("Agent must be between 16 and 80 years old")
            except ValueError:
                validation_errors.append("Invalid date format. Use YYYY-MM-DD")
        
        if gender and gender not in ['Male', 'Female', 'Other']:
            validation_errors.append("Gender must be Male, Female, or Other")
        
        if validation_errors:
            logger.error(f"Validation errors: {validation_errors}")
            raise HTTPException(status_code=400, detail="Validation errors: " + "; ".join(validation_errors))
        
        # Check if agent already exists
        existing_agent = db.query(Agent).filter(
            (Agent.email == email) | (Agent.mobile == mobile)
        ).first()
        
        if existing_agent:
            if existing_agent.email == email:
                logger.error(f"Email already registered: {email}")
                raise HTTPException(status_code=409, detail="Email already registered")
            else:
                logger.error(f"Mobile number already registered: {mobile}")
                raise HTTPException(status_code=409, detail="Mobile number already registered")
        
        # Generate unique agent credentials
        agent_id = generate_unique_agent_id(db)
        password = generate_secure_password()
        hashed_password = hash_password(password)
        
        logger.info(f"Generated credentials - Agent ID: {agent_id}, Password: {password}")
        
        # Create new agent
        agent_data = {
            'agent_id': agent_id,
            'name': name,
            'email': email,
            'mobile': mobile,
            'hashed_password': hashed_password,
            'status': "active",
            'created_at': datetime.utcnow()
        }
        
        if dob:
            agent_data['dob'] = dob
        if country:
            agent_data['country'] = country.strip()
        if gender:
            agent_data['gender'] = gender
        
        new_agent = Agent(**agent_data)
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        logger.info(f"✅ Agent registered successfully: {agent_id}")
        
        # Build response
        response_data = {
            "success": True,
            "message": "Registration successful! Please save your credentials.",
            "agent_id": agent_id,
            "password": password,
            "name": name,
            "email": email,
            "mobile": mobile,
            "status": "active",
            "instructions": [
                "Save your Agent ID and Password securely",
                "You can now log in with these credentials"
            ]
        }
        
        if dob:
            response_data["dob"] = dob
        if country:
            response_data["country"] = country
        if gender:
            response_data["gender"] = gender
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Registration error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/agents/check-availability")
async def check_availability(
    email: Optional[str] = Form(None),
    mobile: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Check if email or mobile is available for registration"""
    try:
        result = {"available": True, "message": "Available"}
        
        if email:
            email = email.strip().lower()
            if not validate_email(email):
                return {"available": False, "message": "Invalid email format"}
            
            existing_email = db.query(Agent).filter(Agent.email == email).first()
            if existing_email:
                return {"available": False, "message": "Email already registered"}
        
        if mobile:
            mobile = re.sub(r'[\s\-\(\)]', '', mobile)
            if not validate_mobile(mobile):
                return {"available": False, "message": "Invalid mobile format"}
            
            existing_mobile = db.query(Agent).filter(Agent.mobile == mobile).first()
            if existing_mobile:
                return {"available": False, "message": "Mobile number already registered"}
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Availability check error: {e}")
        return {"available": False, "message": "Check failed"}

@router.get("/agents/test-registration")
async def test_registration_system(db: Session = Depends(get_db)):
    """Test the registration system"""
    try:
        agent_count = db.query(Agent).count()
        test_agent_id = generate_unique_agent_id(db)
        test_password = generate_secure_password()
        
        return {
            "system_status": "healthy",
            "database_connected": True,
            "current_agent_count": agent_count,
            "sample_credentials": {
                "agent_id": test_agent_id,
                "password": test_password
            },
            "registration_endpoint": "/api/agents/register",
            "required_fields": ["name", "email", "mobile"],
            "optional_fields": ["dob", "country", "gender"]
        }
        
    except Exception as e:
        return {
            "system_status": "error",
            "error": str(e),
            "database_connected": False
        }

# ===================== AGENT AUTHENTICATION ENDPOINTS =====================

@router.post("/agents/login")
async def agent_login(
    agent_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Agent login"""
    try:
        agent = db.query(Agent).filter(
            Agent.agent_id == agent_id,
            Agent.status == "active"
        ).first()
        
        if not agent or not verify_password(password, agent.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Update login status
        agent.last_login = datetime.utcnow()
        agent.is_currently_logged_in = True
        db.commit()
        
        # Create session token
        session_token = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=8)
        
        agent_session = AgentSession(
            agent_id=agent_id,
            session_token=session_token,
            expires_at=expires_at,
            is_active=True
        )
        db.add(agent_session)
        db.commit()
        
        return {
            "success": True,
            "agent_id": agent.agent_id,
            "name": agent.name,
            "session_token": session_token,
            "expires_at": expires_at.isoformat(),
            "message": "Login successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/agents/logout")
async def agent_logout(
    agent_id: str = Form(...),
    session_token: str = Form(...),
    db: Session = Depends(get_db)
):
    """Agent logout"""
    try:
        # Find and invalidate session
        session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.session_token == session_token,
            AgentSession.is_active == True
        ).first()
        
        if session:
            session.is_active = False
            session.expires_at = datetime.utcnow()
        
        # Update agent status
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if agent:
            agent.is_currently_logged_in = False
        
        db.commit()
        
        return {
            "success": True,
            "message": "Logout successful"
        }
        
    except Exception as e:
        logger.error(f"❌ Agent logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

# ===================== AGENT TASK ENDPOINTS =====================

@router.get("/agents/{agent_id}/tasks")
async def get_agent_tasks(
    agent_id: str,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get tasks for agent"""
    try:
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        query = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id)
        
        if status:
            query = query.filter(TaskProgress.status == status)
        
        tasks = query.order_by(TaskProgress.created_at.desc()).all()
        
        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "image_name": task.image_name,
                    "image_path": f"/static/task_images/{agent_id}/{task.image_name}",
                    "status": task.status,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None
                }
                for task in tasks
            ],
            "total": len(tasks)
        }
        
    except Exception as e:
        logger.error(f"Tasks fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")

@router.get("/agents/{agent_id}/current-task")
async def get_current_task(agent_id: str, db: Session = Depends(get_db)):
    """Get current task for an agent"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get the first pending task
        task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == "pending"
        ).first()
        
        if not task:
            return {
                "completed": True,
                "message": "No tasks available",
                "agent_id": agent_id
            }
        
        return {
            "completed": False,
            "message": "Task available",
            "agent_id": agent_id,
            "task_id": task.task_id,
            "image_name": task.image_name,
            "image_url": f"/static/task_images/{agent_id}/{task.image_name}"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting current task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task: {str(e)}")

# ===================== AGENT PROFILE ENDPOINTS =====================

@router.get("/agents/{agent_id}/profile")
async def get_agent_profile(agent_id: str, db: Session = Depends(get_db)):
    """Get agent profile information"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get task statistics
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == "completed"
        ).count()
        
        # Get recent submissions
        recent_submissions = db.query(SubmittedForm).filter(
            SubmittedForm.agent_id == agent_id
        ).order_by(SubmittedForm.submitted_at.desc()).limit(5).all()
        
        return {
            "agent": {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "mobile": agent.mobile,
                "status": agent.status,
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "last_login": agent.last_login.isoformat() if agent.last_login else None,
                "tasks_completed": agent.tasks_completed,
                "dob": agent.dob,
                "country": agent.country,
                "gender": agent.gender
            },
            "stats": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "pending_tasks": total_tasks - completed_tasks,
                "completion_rate": round((completed_tasks / total_tasks * 100), 2) if total_tasks > 0 else 0
            },
            "recent_submissions": [
                {
                    "task_id": sub.task_id,
                    "image_name": sub.image_name,
                    "crime_type": sub.crime_type,
                    "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None
                }
                for sub in recent_submissions
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting agent profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent profile")

# ===================== HEALTH CHECK ENDPOINT =====================

@router.get("/agents/health")
async def agent_health_check():
    """Agent service health check"""
    return {
        "status": "healthy",
        "service": "Agent Management Service",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints_available": [
            "/api/agents/register",
            "/api/agents/login",
            "/api/agents/{agent_id}/tasks",
            "/api/agents/{agent_id}/profile"
        ]
    }
