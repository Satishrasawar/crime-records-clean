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

# Import from your main application
from app.database import get_db
from app.models import Agent, TaskProgress, SubmittedForm, AgentSession, Admin, ChunkedUpload
from app.security import hash_password, verify_password, create_access_token, SECRET_KEY, ALGORITHM
from app.main import UPLOAD_DIR, TASKS_DIR, TEMP_DIR, CHUNK_UPLOAD_DIR

router = APIRouter()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Utility functions
def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number format"""
    clean_mobile = re.sub(r'[\s\-\(\)]', '', mobile)
    return re.match(r'^\+?\d{10,15}$', clean_mobile) is not None

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

def get_agent_image_files(agent_id: str):
    """Get all image files assigned to a specific agent"""
    agent_folder = f"static/task_images/{agent_id}"
    if not os.path.exists(agent_folder):
        agent_folder = "static/task_images/crime_records_wide"
    if not os.path.exists(agent_folder):
        return []
    return sorted([f for f in os.listdir(agent_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# ===================== AGENT REGISTRATION ENDPOINTS =====================

@router.post("/api/agents/register")
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
        logger.info(f"🆕 Agent registration attempt: {name}, {email}")
        
        # Clean inputs
        name = name.strip()
        email = email.strip().lower()
        mobile = re.sub(r'[\s\-\(\)]', '', mobile)
        
        # Collect all validation errors
        validation_errors = []
        
        if not name or len(name) < 2:
            validation_errors.append("Name must be at least 2 characters long")
        
        if not validate_email(email):
            validation_errors.append("Invalid email format")
        
        if not validate_mobile(mobile):
            validation_errors.append("Invalid mobile number format. Use 10-15 digits")
        
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
            raise HTTPException(status_code=400, detail="Validation errors: " + "; ".join(validation_errors))
        
        # Check if agent already exists
        existing_agent = db.query(Agent).filter(
            (Agent.email == email) | (Agent.mobile == mobile)
        ).first()
        
        if existing_agent:
            if existing_agent.email == email:
                raise HTTPException(status_code=409, detail="Email already registered")
            else:
                raise HTTPException(status_code=409, detail="Mobile number already registered")
        
        # Generate unique agent credentials
        agent_id = generate_unique_agent_id(db)
        password = generate_secure_password()
        hashed_password = hash_password(password)
        
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
        logger.error(f"❌ Registration error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/api/agents/check-availability")
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

@router.get("/api/agents/test-registration")
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

@router.post("/api/agents/login")
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

@router.post("/api/agents/logout")
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

@router.get("/api/agents/{agent_id}/tasks")
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

@router.get("/api/agents/{agent_id}/current-task")
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

@router.post("/api/agents/{agent_id}/submit")
async def submit_task_form(
    agent_id: str,
    task_id: str = Form(...),
    crime_type: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    date_time: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    suspect_info: Optional[str] = Form(None),
    witness_info: Optional[str] = Form(None),
    evidence_details: Optional[str] = Form(None),
    priority_level: Optional[str] = Form(None),
    reporter_name: Optional[str] = Form(None),
    reporter_contact: Optional[str] = Form(None),
    case_number: Optional[str] = Form(None),
    investigating_officer: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Submit completed task form"""
    try:
        # Verify agent and task
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        task = db.query(TaskProgress).filter(
            TaskProgress.task_id == task_id,
            TaskProgress.agent_id == agent_id
        ).first()
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        # Check if already submitted
        existing_submission = db.query(SubmittedForm).filter(
            SubmittedForm.task_id == task_id,
            SubmittedForm.agent_id == agent_id
        ).first()
        
        if existing_submission:
            # Update existing submission
            existing_submission.crime_type = crime_type
            existing_submission.location = location
            existing_submission.date_time = date_time
            existing_submission.description = description
            existing_submission.suspect_info = suspect_info
            existing_submission.witness_info = witness_info
            existing_submission.evidence_details = evidence_details
            existing_submission.priority_level = priority_level
            existing_submission.reporter_name = reporter_name
            existing_submission.reporter_contact = reporter_contact
            existing_submission.case_number = case_number
            existing_submission.investigating_officer = investigating_officer
            existing_submission.submitted_at = datetime.utcnow()
            message = "Form updated successfully"
        else:
            # Create new submission
            submission = SubmittedForm(
                agent_id=agent_id,
                task_id=task_id,
                image_name=task.image_name,
                crime_type=crime_type,
                location=location,
                date_time=date_time,
                description=description,
                suspect_info=suspect_info,
                witness_info=witness_info,
                evidence_details=evidence_details,
                priority_level=priority_level,
                reporter_name=reporter_name,
                reporter_contact=reporter_contact,
                case_number=case_number,
                investigating_officer=investigating_officer
            )
            db.add(submission)
            
            # Update task status
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            
            # Update agent task count
            agent.tasks_completed += 1
            
            message = "Form submitted successfully"
        
        db.commit()
        
        return {
            "success": True,
            "message": message,
            "agent_id": agent_id,
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error submitting task: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Task submission failed: {str(e)}")

# ===================== AGENT PROFILE ENDPOINTS =====================

@router.get("/api/agents/{agent_id}/profile")
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

@router.put("/api/agents/{agent_id}/profile")
async def update_agent_profile(
    agent_id: str,
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    mobile: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Update agent profile information"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Validate inputs if provided
        if email and not validate_email(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        if mobile and not validate_mobile(mobile):
            raise HTTPException(status_code=400, detail="Invalid mobile number format")
        
        # Check for duplicate email
        if email and email != agent.email:
            existing_agent = db.query(Agent).filter(Agent.email == email).first()
            if existing_agent:
                raise HTTPException(status_code=409, detail="Email already registered")
        
        # Check for duplicate mobile
        if mobile and mobile != agent.mobile:
            existing_agent = db.query(Agent).filter(Agent.mobile == mobile).first()
            if existing_agent:
                raise HTTPException(status_code=409, detail="Mobile number already registered")
        
        # Update fields
        if name:
            agent.name = name.strip()
        if email:
            agent.email = email.strip().lower()
        if mobile:
            agent.mobile = re.sub(r'[\s\-\(\)]', '', mobile)
        
        db.commit()
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating agent profile: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update profile")

@router.post("/api/agents/{agent_id}/change-password")
async def change_agent_password(
    agent_id: str,
    current_password: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Change agent password"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Verify current password
        if not verify_password(current_password, agent.hashed_password):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        
        # Update password
        agent.hashed_password = hash_password(new_password)
        db.commit()
        
        return {
            "success": True,
            "message": "Password changed successfully",
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error changing password: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to change password")

# ===================== AGENT DASHBOARD ENDPOINTS =====================

@router.get("/api/agents/{agent_id}/dashboard")
async def get_agent_dashboard(agent_id: str, db: Session = Depends(get_db)):
    """Get agent dashboard data"""
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
        
        # Get recent activity
        recent_submissions = db.query(SubmittedForm).filter(
            SubmittedForm.agent_id == agent_id
        ).order_by(SubmittedForm.submitted_at.desc()).limit(5).all()
        
        # Get performance metrics (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_completed = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == "completed",
            TaskProgress.completed_at >= seven_days_ago
        ).count()
        
        return {
            "agent": {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "tasks_completed": agent.tasks_completed
            },
            "stats": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "pending_tasks": total_tasks - completed_tasks,
                "completion_rate": round((completed_tasks / total_tasks * 100), 2) if total_tasks > 0 else 0,
                "recent_completed": recent_completed
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
        logger.error(f"❌ Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dashboard")

# ===================== AGENT SESSION MANAGEMENT =====================

@router.get("/api/agents/{agent_id}/sessions")
async def get_agent_sessions(agent_id: str, db: Session = Depends(get_db)):
    """Get agent session history"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        sessions = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id
        ).order_by(AgentSession.created_at.desc()).limit(20).all()
        
        return {
            "sessions": [
                {
                    "session_token": session.session_token,
                    "created_at": session.created_at.isoformat() if session.created_at else None,
                    "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                    "is_active": session.is_active,
                    "duration_minutes": (
                        (session.expires_at - session.created_at).total_seconds() / 60 
                        if session.expires_at and session.created_at else None
                    )
                }
                for session in sessions
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sessions")

@router.post("/api/agents/{agent_id}/invalidate-session")
async def invalidate_agent_session(
    agent_id: str,
    session_token: str = Form(...),
    db: Session = Depends(get_db)
):
    """Invalidate a specific agent session"""
    try:
        session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.session_token == session_token
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session.is_active = False
        session.expires_at = datetime.utcnow()
        db.commit()
        
        return {
            "success": True,
            "message": "Session invalidated successfully",
            "agent_id": agent_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error invalidating session: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to invalidate session")

# ===================== HEALTH CHECK ENDPOINT =====================

@router.get("/api/agents/health")
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
            "/api/agents/{agent_id}/profile",
            "/api/agents/{agent_id}/dashboard"
        ]
    }
