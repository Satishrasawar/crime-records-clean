from fastapi import APIRouter, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from database import get_db
from models import Agent, TaskProgress, SubmittedForm, AgentSession
import os
import secrets
import string
import hashlib
from datetime import datetime
import json
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

# Simple schemas defined here instead of separate file
class AgentStatusUpdateSchema(BaseModel):
    status: str

# Simple password functions (no external security.py needed)
def hash_password(password: str) -> str:
    salt = "client_records_salt_2024"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

# Utility function to generate agent ID and password
def generate_agent_credentials():
    agent_id = "AGT" + "".join(secrets.choice(string.digits) for _ in range(6))
    password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
    return agent_id, password

@router.post("/api/agents/register")
async def register_agent(
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(...),
    country: str = Form(...),
    gender: str = Form(...),
    db: Session = Depends(get_db)
):
    """Register a new agent with auto-generated credentials"""
    
    # Check if email already exists
    existing_agent = db.query(Agent).filter(Agent.email == email).first()
    if existing_agent:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Generate unique credentials
    agent_id, password = generate_agent_credentials()
    
    # Ensure unique agent ID
    max_attempts = 10
    attempt = 0
    while db.query(Agent).filter(Agent.agent_id == agent_id).first() and attempt < max_attempts:
        agent_id, password = generate_agent_credentials()
        attempt += 1
    
    if attempt >= max_attempts:
        raise HTTPException(status_code=500, detail="Failed to generate unique agent ID")
    
    try:
        # Create new agent
        new_agent = Agent(
            agent_id=agent_id,
            name=name,
            email=email,
            mobile=mobile,
            dob=dob,
            country=country,
            gender=gender,
            password=password,
            hashed_password=hash_password(password),
            status="active"
        )
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        print(f"✅ New agent registered: {agent_id}")
        
        return {
            "success": True,
            "message": "Agent registered successfully",
            "agent_id": agent_id,
            "password": password,
            "agent_details": {
                "name": name,
                "email": email,
                "mobile": mobile,
                "status": "active"
            }
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.get("/api/agents")
def get_all_agents(db: Session = Depends(get_db)):
    """Get all agents with their statistics"""
    agents = db.query(Agent).all()
    result = []
    
    for agent in agents:
        # Get task completion count
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent.agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent.agent_id,
            TaskProgress.status == 'completed'
        ).count()
        
        # Get latest session info
        try:
            latest_session = db.query(AgentSession).filter(
                AgentSession.agent_id == agent.agent_id
            ).order_by(AgentSession.login_time.desc()).first()
        except Exception as e:
            print(f"Error querying sessions for agent {agent.agent_id}: {e}")
            latest_session = None
        
        result.append({
            "id": agent.id,
            "agent_id": agent.agent_id,
            "name": agent.name,
            "email": agent.email,
            "password": agent.password,
            "status": agent.status,
            "tasks_completed": completed_tasks,
            "total_tasks": total_tasks,
            "created_at": agent.created_at.isoformat() if hasattr(agent, 'created_at') and agent.created_at else None,
            "last_login": latest_session.login_time.strftime('%Y-%m-%d %H:%M:%S') if latest_session and latest_session.login_time else None,
            "last_logout": latest_session.logout_time.strftime('%Y-%m-%d %H:%M:%S') if latest_session and latest_session.logout_time else None,
            "is_currently_logged_in": latest_session and latest_session.logout_time is None if latest_session else False
        })
    
    return result

@router.post("/api/agents/login")
async def login_agent(
    agent_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Agent login"""
    print(f"🔑 Login attempt for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent {agent_id} not found")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check password - try direct password first, then hashed
    password_valid = False
    if agent.password and agent.password == password:
        password_valid = True
    elif hasattr(agent, 'hashed_password') and agent.hashed_password and verify_password(password, agent.hashed_password):
        password_valid = True
    
    if not password_valid:
        print(f"❌ Invalid password for agent {agent_id}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if agent.status != "active":
        print(f"❌ Agent {agent_id} is not active")
        raise HTTPException(status_code=403, detail="Agent account is not active")
    
    try:
        # End any existing active sessions
        active_sessions = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).all()
        
        for session in active_sessions:
            session.logout_time = datetime.utcnow()
            if session.login_time:
                duration = (session.logout_time - session.login_time).total_seconds() / 60
                session.duration_minutes = round(duration, 2)
        
        # Create new session
        new_session = AgentSession(
            agent_id=agent_id,
            login_time=datetime.utcnow(),
            ip_address="127.0.0.1",
            user_agent="Web Browser"
        )
        
        db.add(new_session)
        db.commit()
        
        print(f"✅ Agent {agent_id} logged in successfully")
        
        return {
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "name": agent.name,
            "session_id": new_session.id,
            "login_time": new_session.login_time.isoformat()
        }
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        # Even if session tracking fails, allow login
        return {
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "name": agent.name
        }

@router.post("/api/agents/{agent_id}/logout")
async def logout_agent(agent_id: str, db: Session = Depends(get_db)):
    """Handle agent logout"""
    print(f"👋 Logout request for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        # Find active session
        active_session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).first()
        
        if active_session:
            active_session.logout_time = datetime.utcnow()
            if active_session.login_time:
                duration = (active_session.logout_time - active_session.login_time).total_seconds() / 60
                active_session.duration_minutes = round(duration, 2)
            db.commit()
            
            return {
                "message": "Logout successful",
                "session_duration": f"{active_session.duration_minutes} minutes" if active_session.duration_minutes else "Unknown"
            }
    except Exception as e:
        print(f"❌ Logout session error: {e}")
    
    return {"message": "Logout successful"}

# ===== TASK ENDPOINTS =====

@router.get("/api/agents/{agent_id}/current-task")
def get_current_task(agent_id: str, db: Session = Depends(get_db)):
    """Get current task for an agent"""
    print(f"🔄 Getting current task for agent: {agent_id}")
    
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent {agent_id} not found")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Find next pending or in-progress task
    next_task = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status.in_(['pending', 'in_progress'])
    ).order_by(TaskProgress.assigned_at).first()
    
    if not next_task:
        # Check completed tasks for progress
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'completed'
        ).count()
        
        print(f"✅ All tasks completed for agent {agent_id}: {completed_tasks}/{total_tasks}")
        
        return {
            "completed": True,
            "message": "All tasks completed",
            "total_completed": completed_tasks,
            "total_tasks": total_tasks
        }
    
    # Mark task as in_progress if pending
    if next_task.status == 'pending':
        next_task.status = 'in_progress'
        next_task.started_at = datetime.utcnow()
        db.commit()
        print(f"📋 Marked task {next_task.id} as in progress")
    
    # Get statistics
    total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
    completed_tasks = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'completed'
    ).count()
    current_index = completed_tasks
    
    print(f"📊 Task stats for agent {agent_id}: {completed_tasks}/{total_tasks} completed")
    
    # Fix image path
    image_path = next_task.image_path
    if not image_path.startswith('/'):
        image_path = '/' + image_path
    
    return {
        "task": {
            "id": next_task.id,
            "agent_id": next_task.agent_id,
            "image_path": image_path,
            "image_filename": next_task.image_filename,
            "status": next_task.status,
            "assigned_at": next_task.assigned_at.isoformat()
        },
        "image_url": image_path,
        "image_name": next_task.image_filename,
        "current_index": current_index,
        "total_images": total_tasks,
        "progress": f"{current_index + 1}/{total_tasks}",
        "next_available": True
    }

@router.get("/api/agents/{agent_id}/tasks")  
def get_agent_tasks(agent_id: str, db: Session = Depends(get_db)):
    """Get all tasks for an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    tasks = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id
    ).order_by(TaskProgress.assigned_at).all()
    
    task_list = []
    for task in tasks:
        task_data = {
            "id": task.id,
            "agent_id": task.agent_id,
            "image_path": task.image_path,
            "image_filename": task.image_filename,
            "status": task.status,
            "assigned_at": task.assigned_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }
        task_list.append(task_data)
    
    return task_list

# ===== SUBMIT ENDPOINTS - FIXED =====

@router.post("/api/agents/{agent_id}/submit")
async def submit_task_data(
    agent_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Submit task data - FIXED VERSION"""
    print(f"🔄 Processing submission for agent: {agent_id}")
    
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent {agent_id} not found")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get current task
    current_task = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'in_progress'
    ).order_by(TaskProgress.assigned_at).first()
    
    if not current_task:
        # Try pending task
        current_task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'pending'
        ).order_by(TaskProgress.assigned_at).first()
    
    if not current_task:
        print(f"❌ No active task found for agent {agent_id}")
        raise HTTPException(status_code=404, detail="No active task found for submission")
    
    # Parse form data
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            form_data = await request.json()
        else:
            form_data_raw = await request.form()
            form_data = {}
            for key, value in form_data_raw.items():
                if key not in ['agent_id', 'task_id']:  # Skip metadata
                    form_data[key] = str(value)  # Convert to string
        
        print(f"📝 Parsed form data: {len(form_data)} fields")
        
    except Exception as e:
        print(f"❌ Error parsing form data: {e}")
        form_data = {}
    
    # Validate data
    if not form_data:
        print(f"❌ No form data received for agent {agent_id}")
        raise HTTPException(status_code=400, detail="No form data received")
    
    try:
        # Create submission
        submission = SubmittedForm(
            agent_id=agent_id,
            task_id=current_task.id,
            image_filename=current_task.image_filename,
            form_data=form_data,
            submitted_at=datetime.utcnow()
        )
        db.add(submission)
        
        # Mark task as completed
        current_task.status = 'completed'
        current_task.completed_at = datetime.utcnow()
        
        # Commit changes
        db.commit()
        
        print(f"✅ Task {current_task.id} completed successfully by agent {agent_id}")
        
        return {
            "message": "Task submitted successfully", 
            "success": True,
            "status": "success",
            "submission_id": submission.id,
            "task_id": current_task.id,
            "completed_task": current_task.image_filename
        }
        
    except Exception as e:
        print(f"❌ Database error during submission: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")

# Additional submit endpoints for compatibility
@router.post("/api/submit")
async def submit_generic(request: Request, db: Session = Depends(get_db)):
    """Generic submit endpoint"""
    try:
        form_data = await request.form()
        agent_id = form_data.get('agent_id')
        
        if not agent_id:
            raise HTTPException(status_code=400, detail="Agent ID required")
        
        return await submit_task_data(agent_id, request, db)
        
    except Exception as e:
        print(f"❌ Generic submit error: {e}")
        raise HTTPException(status_code=500, detail=f"Submit failed: {str(e)}")

@router.post("/api/agents/submit")
async def submit_agents_generic(request: Request, db: Session = Depends(get_db)):
    """Generic agents submit endpoint"""
    try:
        form_data = await request.form()
        agent_id = form_data.get('agent_id')
        
        if not agent_id:
            raise HTTPException(status_code=400, detail="Agent ID required")
        
        return await submit_task_data(agent_id, request, db)
        
    except Exception as e:
        print(f"❌ Agents submit error: {e}")
        raise HTTPException(status_code=500, detail=f"Submit failed: {str(e)}")

@router.post("/api/agents/tasks/{task_id}/submit")
async def submit_by_task_id(task_id: int, request: Request, db: Session = Depends(get_db)):
    """Submit by task ID"""
    try:
        task = db.query(TaskProgress).filter(TaskProgress.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return await submit_task_data(task.agent_id, request, db)
        
    except Exception as e:
        print(f"❌ Task submit error: {e}")
        raise HTTPException(status_code=500, detail=f"Submit failed: {str(e)}")

# Alternative task endpoints
@router.get("/api/agents/{agent_id}/tasks/next")
def get_next_task(agent_id: str, db: Session = Depends(get_db)):
    """Get next task - same as current"""
    return get_current_task(agent_id, db)

@router.get("/api/tasks/current/{agent_id}")  
def get_current_task_alt(agent_id: str, db: Session = Depends(get_db)):
    """Alternative current task endpoint"""
    return get_current_task(agent_id, db)

@router.get("/api/tasks/next/{agent_id}")
def get_next_task_alt(agent_id: str, db: Session = Depends(get_db)):
    """Alternative next task endpoint"""  
    return get_current_task(agent_id, db)

# ===== ADMIN ENDPOINTS =====

@router.get("/api/admin/agent-password/{agent_id}")
def get_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Get agent password for admin"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent_id,
        "message": f"Password for agent {agent_id} is: {agent.password}",
        "password": agent.password
    }

@router.post("/api/admin/reset-password/{agent_id}")
def reset_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Reset agent password"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Generate new password
    _, new_password = generate_agent_credentials()
    
    # Update password
    agent.password = new_password
    if hasattr(agent, 'hashed_password'):
        agent.hashed_password = hash_password(new_password)
    db.commit()
    
    return {
        "agent_id": agent_id,
        "new_password": new_password,
        "message": "Password reset successfully"
    }

@router.patch("/api/agents/{agent_id}/status")
def update_agent_status(agent_id: str, status_data: AgentStatusUpdateSchema, db: Session = Depends(get_db)):
    """Update agent status"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.status = status_data.status
    db.commit()
    return {"message": "Agent status updated successfully"}

@router.post("/api/admin/force-logout/{agent_id}")
async def force_logout_agent(agent_id: str, db: Session = Depends(get_db)):
    """Admin force logout an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        # End active session
        active_session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).first()
        
        if active_session:
            active_session.logout_time = datetime.utcnow()
            if active_session.login_time:
                duration = (active_session.logout_time - active_session.login_time).total_seconds() / 60
                active_session.duration_minutes = round(duration, 2)
            db.commit()
            
            return {
                "message": f"Agent {agent_id} has been forcefully logged out",
                "session_duration": f"{active_session.duration_minutes} minutes" if active_session.duration_minutes else "Unknown"
            }
    except Exception as e:
        print(f"Force logout error: {e}")
    
    return {"message": "Agent was not logged in"}

@router.get("/api/admin/statistics")
def get_admin_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        total_agents = db.query(Agent).count()
        active_agents = db.query(Agent).filter(Agent.status == "active").count()
        total_tasks = db.query(TaskProgress).count()
        completed_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'completed').count()
        pending_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'pending').count()
        in_progress_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'in_progress').count()
        total_submissions = db.query(SubmittedForm).count()
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks,
            "total_submissions": total_submissions
        }
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return {
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0,
            "total_submissions": 0
        }

@router.get("/api/admin/session-report")
async def get_session_report(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get detailed session report for admin"""
    try:
        query = db.query(AgentSession)
        
        if agent_id:
            query = query.filter(AgentSession.agent_id == agent_id)
        
        if date_from:
            from_date = datetime.fromisoformat(date_from)
            query = query.filter(AgentSession.login_time >= from_date)
        
        if date_to:
            to_date = datetime.fromisoformat(date_to)
            query = query.filter(AgentSession.login_time <= to_date)
        
        sessions = query.order_by(AgentSession.login_time.desc()).all()
        
        result = []
        for session in sessions:
            agent = db.query(Agent).filter(Agent.agent_id == session.agent_id).first()
            result.append({
                "session_id": session.id,
                "agent_id": session.agent_id,
                "agent_name": agent.name if agent else "Unknown",
                "login_time": session.login_time.isoformat(),
                "logout_time": session.logout_time.isoformat() if session.logout_time else None,
                "duration_minutes": session.duration_minutes,
                "is_active": session.logout_time is None
            })
        
        return result
    except Exception as e:
        print(f"Session report error: {e}")
        return []

@router.get("/api/admin/preview-data")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Preview submitted data"""
    try:
        query = db.query(SubmittedForm)
        
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        
        if date_from:
            query = query.filter(SubmittedForm.submitted_at >= datetime.strptime(date_from, '%Y-%m-%d'))
        
        if date_to:
            query = query.filter(SubmittedForm.submitted_at <= datetime.strptime(date_to, '%Y-%m-%d'))
        
        submissions = query.limit(100).all()
        
        result = []
        for submission in submissions:
            # Handle both JSON and string format
            try:
                if isinstance(submission.form_data, str):
                    form_data = json.loads(submission.form_data)
                else:
                    form_data = submission.form_data
            except (json.JSONDecodeError, TypeError):
                form_data = {}
                
            result.append({
                "id": submission.id,
                "agent_id": submission.agent_id,
                "task_id": submission.task_id,
                "image_filename": submission.image_filename,
                "submitted_at": submission.submitted_at.isoformat(),
                "form_data": form_data
            })
        
        return result
        
    except Exception as e:
        print(f"❌ Error in data preview: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@router.get("/api/admin/test-data")
async def test_data_availability(db: Session = Depends(get_db)):
    """Test data availability"""
    try:
        # Count records in each table
        agent_count = db.query(Agent).count()
        task_count = db.query(TaskProgress).count()
        submission_count = db.query(SubmittedForm).count()
        session_count = db.query(AgentSession).count()
        
        return {
            "success": True,
            "message": f"Data available - Agents: {agent_count}, Tasks: {task_count}, Submissions: {submission_count}, Sessions: {session_count}",
            "counts": {
                "agents": agent_count,
                "tasks": task_count,
                "submissions": submission_count,
                "sessions": session_count
            }
        }
        
    except Exception as e:
        print(f"❌ Error testing data: {e}")
        raise HTTPException(status_code=500, detail=f"Data test failed: {str(e)}")
