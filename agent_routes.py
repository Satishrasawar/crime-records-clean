from fastapi import APIRouter, Form, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
from models import Agent, TaskProgress, SubmittedForm, AgentSession
from schemas import AgentStatusUpdateSchema
from security import hash_password, verify_password
import os
import secrets
import string
from datetime import datetime, timedelta
import json
import zipfile
import io
import pandas as pd
from typing import Optional

router = APIRouter()

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
    
    # Ensure unique agent ID (very unlikely collision, but safety first)
    max_attempts = 10
    attempt = 0
    while db.query(Agent).filter(Agent.agent_id == agent_id).first() and attempt < max_attempts:
        agent_id, password = generate_agent_credentials()
        attempt += 1
    
    if attempt >= max_attempts:
        raise HTTPException(status_code=500, detail="Failed to generate unique agent ID")
    
    try:
        # Create new agent - use both password formats for compatibility
        hashed_pwd = hash_password(password)
        new_agent = Agent(
            agent_id=agent_id,
            name=name,
            email=email,
            mobile=mobile,
            dob=dob,
            country=country,
            gender=gender,
            password=password,        # Direct password storage (current system)
            hashed_password=hashed_pwd,  # Hashed password (future use)
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
    agents = db.query(Agent).all()
    result = []
    
    for agent in agents:
        # Get task completion count from new TaskProgress table
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent.agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent.agent_id,
            TaskProgress.status == 'completed'
        ).count()
        
        # Get login/logout times with proper error handling
        try:
            login_sessions = db.query(AgentSession).filter(
                AgentSession.agent_id == agent.agent_id
            ).order_by(AgentSession.login_time.desc()).limit(5).all()
        except Exception as e:
            print(f"Error querying sessions for agent {agent.agent_id}: {e}")
            login_sessions = []
        
        last_login = None
        last_logout = None
        current_session = None
        
        if login_sessions:
            # Find current active session
            current_session = next((s for s in login_sessions if s.logout_time is None), None)
            last_login = login_sessions[0].login_time.strftime('%Y-%m-%d %H:%M:%S') if login_sessions[0].login_time else None
            
            # Find last completed session
            completed_sessions = [s for s in login_sessions if s.logout_time is not None]
            if completed_sessions:
                last_logout = completed_sessions[0].logout_time.strftime('%Y-%m-%d %H:%M:%S')
        
        result.append({
            "id": agent.id,
            "agent_id": agent.agent_id,
            "name": agent.name,
            "email": agent.email,
            "password": agent.password,  # Show actual password for admin
            "status": agent.status,
            "tasks_completed": completed_tasks,
            "total_tasks": total_tasks,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "last_login": last_login,
            "last_logout": last_logout,
            "current_session_duration": None,  # We'll calculate this in frontend
            "is_currently_logged_in": current_session is not None,
            "recent_sessions": [
                {
                    "login_time": s.login_time.strftime('%Y-%m-%d %H:%M:%S') if s.login_time else None,
                    "logout_time": s.logout_time.strftime('%Y-%m-%d %H:%M:%S') if s.logout_time else None,
                    "duration_minutes": s.duration_minutes
                } for s in login_sessions
            ]
        })
    
    return result

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
    """Reset agent password and return new one"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Generate new password
    _, new_password = generate_agent_credentials()
    
    # Update both password formats
    agent.password = new_password
    agent.hashed_password = hash_password(new_password)
    db.commit()
    
    return {
        "agent_id": agent_id,
        "new_password": new_password,
        "message": "Password reset successfully"
    }

@router.patch("/api/agents/{agent_id}/status")
def update_agent_status(agent_id: str, status_data: AgentStatusUpdateSchema, db: Session = Depends(get_db)):
    # Find by agent_id string, not integer id
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.status = status_data.status
    db.commit()
    return {"message": "Agent status updated successfully"}

@router.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(agent)
    db.commit()
    return {"message": "Agent deleted successfully"}

# FIXED LOGIN ENDPOINT - This was the main issue!
@router.post("/api/agents/login")
async def login_agent(
    agent_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """FIXED Agent login - check both password formats for compatibility"""
    print(f"🔑 Login attempt for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check password - try direct password first, then hashed
    password_valid = False
    if agent.password and agent.password == password:
        password_valid = True
        print(f"✅ Direct password match for {agent_id}")
    elif agent.hashed_password and verify_password(password, agent.hashed_password):
        password_valid = True
        print(f"✅ Hashed password match for {agent_id}")
    
    if not password_valid:
        print(f"❌ Invalid password for {agent_id}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if agent.status != "active":
        print(f"❌ Agent {agent_id} is not active: {agent.status}")
        raise HTTPException(status_code=403, detail="Agent account is not active")
    
    try:
        # End any existing active sessions for this agent
        active_sessions = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).all()
        
        for session in active_sessions:
            session.logout_time = datetime.utcnow()
            duration = (session.logout_time - session.login_time).total_seconds() / 60
            session.duration_minutes = round(duration, 2)
        
        # Create new session
        new_session = AgentSession(
            agent_id=agent_id,
            login_time=datetime.utcnow(),
            ip_address="127.0.0.1",  # You can get this from request if needed
            user_agent="Web Browser"  # You can get this from request headers if needed
        )
        
        db.add(new_session)
        db.commit()
        
        print(f"✅ Login successful for {agent_id}")
        
        # FIXED: Return proper response structure that frontend expects
        return {
            "success": True,
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "agent": {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "status": agent.status
            },
            "name": agent.name,
            "session_id": new_session.id,
            "login_time": new_session.login_time.isoformat()
        }
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        # Even if session tracking fails, allow login
        return {
            "success": True,
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "name": agent.name,
            "agent": {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "status": agent.status
            }
        }

@router.post("/api/agents/{agent_id}/logout")
async def logout_agent(agent_id: str, db: Session = Depends(get_db)):
    """Handle agent logout and update session"""
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
            duration = (active_session.logout_time - active_session.login_time).total_seconds() / 60
            active_session.duration_minutes = round(duration, 2)
            db.commit()
            
            return {
                "message": "Logout successful",
                "session_duration": f"{active_session.duration_minutes} minutes"
            }
    except Exception as e:
        print(f"Logout session error: {e}")
    
    return {"message": "Logout successful"}

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
            duration = (active_session.logout_time - active_session.login_time).total_seconds() / 60
            active_session.duration_minutes = round(duration, 2)
            db.commit()
            
            return {
                "message": f"Agent {agent_id} has been forcefully logged out",
                "session_duration": f"{active_session.duration_minutes} minutes"
            }
    except Exception as e:
        print(f"Force logout error: {e}")
    
    return {"message": "Agent was not logged in"}

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

# ===== FIXED TASK ENDPOINTS =====

@router.get("/api/agents/{agent_id}/current-task")
def get_current_task(agent_id: str, db: Session = Depends(get_db)):
    """FIXED: Get current task for an agent"""
    print(f"📋 Getting current task for agent: {agent_id}")
    
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # First try to get any existing in_progress task
    current_task = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'in_progress'
    ).order_by(TaskProgress.assigned_at).first()
    
    # If no in_progress task, get the next pending task
    if not current_task:
        current_task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'pending'
        ).order_by(TaskProgress.assigned_at).first()
        
        # If we found a pending task, mark it as in_progress
        if current_task:
            current_task.status = 'in_progress'
            current_task.started_at = datetime.utcnow()
            db.commit()
            db.refresh(current_task)
            print(f"📋 Marked task {current_task.id} as in_progress")
    
    # If no tasks available, return completion status
    if not current_task:
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'completed'
        ).count()
        
        print(f"✅ All tasks completed for {agent_id}: {completed_tasks}/{total_tasks}")
        
        return {
            "completed": True,
            "message": "All tasks completed! Great job!",
            "total_completed": completed_tasks,
            "total_tasks": total_tasks,
            "task": None,
            "image_url": None,
            "image_name": None,
            "current_index": completed_tasks,
            "progress": f"{completed_tasks}/{total_tasks}"
        }
    
    # Calculate progress statistics
    total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
    completed_tasks = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'completed'
    ).count()
    
    print(f"📊 Task progress for {agent_id}: {completed_tasks + 1}/{total_tasks} (current task: {current_task.id})")
    
    return {
        "completed": False,
        "task": {
            "id": current_task.id,
            "agent_id": current_task.agent_id,
            "image_path": current_task.image_path,
            "image_filename": current_task.image_filename,
            "status": current_task.status,
            "assigned_at": current_task.assigned_at.isoformat() if current_task.assigned_at else None,
            "started_at": current_task.started_at.isoformat() if current_task.started_at else None
        },
        "image_url": current_task.image_path,
        "image_name": current_task.image_filename,
        "current_index": completed_tasks + 1,  # Current task index (1-based)
        "total_images": total_tasks,
        "progress": f"{completed_tasks + 1}/{total_tasks}",
        "completion_percentage": round(((completed_tasks + 1) / total_tasks) * 100, 1) if total_tasks > 0 else 0
    }

@router.get("/api/agents/{agent_id}/tasks")
def get_agent_tasks(agent_id: str, db: Session = Depends(get_db)):
    """Get all tasks for an agent"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get all tasks for this agent
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

# FIXED SUBMIT ENDPOINT - This was the other main issue!
@router.post("/api/agents/{agent_id}/submit")
async def submit_task_data(
    agent_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """FIXED: Submit task data - handles both JSON and form data properly"""
    print(f"📤 Submit request received for agent: {agent_id}")
    
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Find the current in-progress task
    current_task = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'in_progress'
    ).order_by(TaskProgress.assigned_at).first()
    
    if not current_task:
        # Fallback: try to find pending task and mark as in_progress
        current_task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'pending'
        ).order_by(TaskProgress.assigned_at).first()
        
        if current_task:
            current_task.status = 'in_progress'
            current_task.started_at = datetime.utcnow()
    
    if not current_task:
        print(f"❌ No active task found for {agent_id}")
        raise HTTPException(
            status_code=404, 
            detail="No active task found for submission. Please refresh and try again."
        )
    
    print(f"📋 Found active task: {current_task.id}")
    
    # Parse form data from request
    try:
        content_type = request.headers.get("content-type", "")
        print(f"📤 Content type: {content_type}")
        
        if content_type.startswith("application/json"):
            form_data = await request.json()
            print(f"📝 Received JSON data with {len(form_data)} fields")
        else:
            # Handle multipart form data
            form_data_raw = await request.form()
            form_data = {}
            for key, value in form_data_raw.items():
                if key not in ['agent_id', 'task_id']:  # Skip metadata fields
                    form_data[key] = value
            print(f"📝 Received form data with {len(form_data)} fields")
        
        # Log the received data (first few fields for debugging)
        if form_data:
            sample_fields = list(form_data.items())[:3]
            print(f"📄 Sample data: {sample_fields}")
        
    except Exception as parse_error:
        print(f"❌ Error parsing form data: {parse_error}")
        raise HTTPException(status_code=400, detail="Invalid form data format")
    
    # Create submission record
    try:
        submission = SubmittedForm(
            agent_id=agent_id,
            task_id=current_task.id,
            image_filename=current_task.image_filename,
            form_data=form_data,  # Store as JSON object
            submitted_at=datetime.utcnow()
        )
        
        db.add(submission)
        
        # Mark current task as completed
        current_task.status = 'completed'
        current_task.completed_at = datetime.utcnow()
        
        # Commit both changes
        db.commit()
        db.refresh(submission)
        db.refresh(current_task)
        
        print(f"✅ Task {current_task.id} completed by agent {agent_id}")
        
    except Exception as db_error:
        print(f"❌ Database error during submission: {db_error}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save submission to database")
    
    # Check if there are more tasks
    next_task = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'pending'
    ).order_by(TaskProgress.assigned_at).first()
    
    # Calculate final statistics
    total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
    completed_tasks = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'completed'
    ).count()
    
    response_data = {
        "success": True,
        "message": "Task submitted successfully!",
        "submission_id": submission.id,
        "task_id": current_task.id,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
        "has_next_task": next_task is not None,
        "progress": f"{completed_tasks}/{total_tasks}",
        "completion_percentage": round((completed_tasks / total_tasks) * 100, 1) if total_tasks > 0 else 0
    }
    
    if next_task:
        response_data["next_task_available"] = True
        response_data["message"] = f"Task submitted! {total_tasks - completed_tasks} tasks remaining."
    else:
        response_data["next_task_available"] = False
        response_data["message"] = "Congratulations! All tasks completed successfully!"
        response_data["all_completed"] = True
    
    print(f"📊 Response: {response_data['message']}")
    return response_data

# ===== ADMIN ROUTES =====

@router.get("/api/admin/statistics")
def get_admin_statistics(db: Session = Depends(get_db)):
    """Get system statistics for admin dashboard"""
    total_agents = db.query(Agent).count()
    active_agents = db.query(Agent).filter(Agent.status == "active").count()
    total_submissions = db.query(SubmittedForm).count()
    
    # Get task statistics from TaskProgress table
    total_tasks = db.query(TaskProgress).count()
    completed_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'completed').count()
    pending_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'pending').count()
    in_progress_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'in_progress').count()
    
    return {
        "total_agents": total_agents,
        "active_agents": active_agents,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "pending_tasks": pending_tasks,
        "in_progress_tasks": in_progress_tasks
    }

@router.get("/api/admin/export-excel")
def export_to_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export submitted data to Excel file"""
    
    try:
        # Build query
        query = db.query(SubmittedForm)
        
        # Apply filters
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        
        if date_from:
            try:
                from_date = datetime.fromisoformat(date_from)
                query = query.filter(SubmittedForm.submitted_at >= from_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                to_date = datetime.fromisoformat(date_to)
                # Add 23:59:59 to include the whole day
                to_date = to_date.replace(hour=23, minute=59, second=59)
                query = query.filter(SubmittedForm.submitted_at <= to_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        # Get all submissions
        submissions = query.order_by(SubmittedForm.submitted_at.desc()).all()
        
        if not submissions:
            raise HTTPException(status_code=404, detail="No data found with current filters")
        
        # Prepare data for Excel
        excel_data = []
        for submission in submissions:
            try:
                # Handle both JSON and string format
                if isinstance(submission.form_data, str):
                    form_data = json.loads(submission.form_data)
                else:
                    form_data = submission.form_data
                
                # Create row with metadata first
                row_data = {
                    'Submission_ID': submission.id,
                    'Agent_ID': submission.agent_id,
                    'Task_ID': submission.task_id,
                    'Submitted_At': submission.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'Image_Name': submission.image_filename or form_data.get('image_name', 'Unknown')
                }
                
                # Add all form fields
                form_fields = [
                    'DR_NO', 'Date_Rptd', 'DATE_OCC', 'TIME_OCC', 'Unique_Identifier',
                    'AREA_NAME', 'Rpt_Dist_No', 'VIN', 'Crm', 'Crm_Cd_Desc', 'Mocodes',
                    'Vict_Age', 'Geolocation', 'DEPARTMENT', 'Premis_Cd', 'Premis_Desc',
                    'ARREST_KEY', 'PD_DESC', 'CCD_LONCOD', 'Status_Desc', 'LAW_CODE',
                    'SubAgency', 'Charge', 'Race', 'LOCATION', 'SeqID', 'LAT', 'LON',
                    'Point', 'Shape__Area'
                ]
                
                for field in form_fields:
                    row_data[field] = form_data.get(field, '')
                
                excel_data.append(row_data)
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error for submission {submission.id}: {e}")
                continue
        
        if not excel_data:
            raise HTTPException(status_code=404, detail="No valid data found")
        
        # Import pandas here to catch import errors
        try:
            import pandas as pd
        except ImportError:
            raise HTTPException(status_code=500, detail="pandas not installed. Run: pip install pandas")
        
        # Create DataFrame
        df = pd.DataFrame(excel_data)
        
        # Create Excel file in memory
        output = io.BytesIO()
        
        try:
            # Use openpyxl engine
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Crime Records Data', index=False)
                
                # Auto-adjust column widths
                workbook = writer.book
                worksheet = writer.sheets['Crime Records Data']
                
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value and len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    # Set width with min 10, max 50
                    adjusted_width = min(max(max_length + 2, 10), 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating Excel file: {str(e)}")
        
        output.seek(0)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crime_records_export_{timestamp}.xlsx"
        
        # Return file as download
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in Excel export: {e}")
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

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
            if isinstance(submission.form_data, str):
                form_data = json.loads(submission.form_data)
            else:
                form_data = submission.form_data
                
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

@router.get("/api/admin/export-sessions")
async def export_sessions(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export session report to Excel"""
    try:
        # For now, return a simple response indicating the feature is available
        return JSONResponse(
            content={"message": "Session export feature available - implement based on your specific requirements"},
            status_code=501
        )
        
    except Exception as e:
        print(f"❌ Error in session export: {e}")
        raise HTTPException(status_code=500, detail=f"Session export failed: {str(e)}")

# ===== ADDITIONAL HELPER ENDPOINTS =====

@router.get("/api/agents/{agent_id}/next-task")
async def get_next_task(agent_id: str, db: Session = Depends(get_db)):
    """Get next available task - Alternative endpoint"""
    try:
        # This endpoint just redirects to current-task for consistency
        return get_current_task(agent_id, db)
        
    except Exception as e:
        print(f"❌ Error getting next task for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting next task: {str(e)}")

@router.post("/api/agents/{agent_id}/skip-task")
async def skip_current_task(agent_id: str, db: Session = Depends(get_db)):
    """Skip current task (mark as skipped) - Optional functionality"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Find current in-progress task
        current_task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'in_progress'
        ).order_by(TaskProgress.assigned_at).first()
        
        if not current_task:
            raise HTTPException(status_code=404, detail="No active task to skip")
        
        # Mark as skipped
        current_task.status = 'skipped'
        current_task.completed_at = datetime.utcnow()
        db.commit()
        
        print(f"⏭️ Task {current_task.id} skipped by agent {agent_id}")
        
        return {
            "success": True,
            "message": "Task skipped successfully",
            "task_id": current_task.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error skipping task for {agent_id}: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Error skipping task: {str(e)}")

@router.get("/api/agents/{agent_id}/progress")
async def get_agent_progress(agent_id: str, db: Session = Depends(get_db)):
    """Get detailed progress information for an agent"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'completed'
        ).count()
        pending_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'pending'
        ).count()
        in_progress_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'in_progress'
        ).count()
        skipped_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'skipped'
        ).count()
        
        completion_percentage = round((completed_tasks / total_tasks) * 100, 1) if total_tasks > 0 else 0
        
        return {
            "agent_id": agent_id,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks,
            "skipped_tasks": skipped_tasks,
            "completion_percentage": completion_percentage,
            "progress_text": f"{completed_tasks}/{total_tasks}",
            "is_completed": pending_tasks == 0 and in_progress_tasks == 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting progress for {agent_id}: {e}")
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0,
            "skipped_tasks": 0,
            "completion_percentage": 0
        }

# ===== BACKWARD COMPATIBILITY ENDPOINTS =====

@router.get("/api/agents/{agent_id}/statistics")
async def get_agent_statistics(agent_id: str, db: Session = Depends(get_db)):
    """Get statistics for a specific agent - Backward compatibility"""
    try:
        return await get_agent_progress(agent_id, db)
    except Exception as e:
        print(f"❌ Error getting statistics for {agent_id}: {e}")
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0
        }

# ===== SYSTEM MAINTENANCE ENDPOINTS =====

@router.post("/api/admin/cleanup-orphaned-tasks")
async def cleanup_orphaned_tasks(db: Session = Depends(get_db)):
    """Clean up orphaned tasks without valid agents"""
    try:
        # Find tasks with non-existent agents
        orphaned_tasks = db.query(TaskProgress).filter(
            ~TaskProgress.agent_id.in_(
                db.query(Agent.agent_id)
            )
        ).all()
        
        if not orphaned_tasks:
            return {
                "success": True,
                "message": "No orphaned tasks found",
                "cleaned_up": 0
            }
        
        # Delete orphaned tasks
        count = len(orphaned_tasks)
        for task in orphaned_tasks:
            db.delete(task)
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Successfully cleaned up {count} orphaned tasks",
            "cleaned_up": count
        }
        
    except Exception as e:
        print(f"❌ Error cleaning up orphaned tasks: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/api/admin/system-health")
async def get_system_health(db: Session = Depends(get_db)):
    """Get comprehensive system health information"""
    try:
        # Database connectivity test
        db_status = "connected"
        try:
            db.execute("SELECT 1")
        except Exception as e:
            db_status = f"error: {str(e)[:50]}"
        
        # Get table counts
        agent_count = db.query(Agent).count()
        task_count = db.query(TaskProgress).count()
        submission_count = db.query(SubmittedForm).count()
        session_count = db.query(AgentSession).count()
        
        # Check for active sessions
        active_sessions = db.query(AgentSession).filter(
            AgentSession.logout_time.is_(None)
        ).count()
        
        # Check for stuck tasks (in_progress for > 1 hour)
        from sqlalchemy import text
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        stuck_tasks = db.query(TaskProgress).filter(
            TaskProgress.status == 'in_progress',
            TaskProgress.started_at < one_hour_ago
        ).count()
        
        return {
            "status": "healthy" if db_status == "connected" else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "database": {
                "status": db_status,
                "agents": agent_count,
                "tasks": task_count,
                "submissions": submission_count,
                "sessions": session_count
            },
            "active_sessions": active_sessions,
            "stuck_tasks": stuck_tasks,
            "warnings": [
                f"{stuck_tasks} tasks stuck in progress" if stuck_tasks > 0 else None,
                f"Database issues: {db_status}" if db_status != "connected" else None
            ]
        }
        
    except Exception as e:
        print(f"❌ Error getting system health: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
