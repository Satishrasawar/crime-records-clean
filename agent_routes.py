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
from datetime import datetime
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

@router.post("/api/agents/login")
async def login_agent(
    agent_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Agent login - check both password formats for compatibility"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check password - try direct password first, then hashed
    password_valid = False
    if agent.password and agent.password == password:
        password_valid = True
    elif agent.hashed_password and verify_password(password, agent.hashed_password):
        password_valid = True
    
    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if agent.status != "active":
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
        
        return {
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "name": agent.name,
            "session_id": new_session.id,
            "login_time": new_session.login_time.isoformat()
        }
    except Exception as e:
        print(f"Session creation error: {e}")
        # Even if session tracking fails, allow login
        return {
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "name": agent.name
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

# ===== NEW COMPATIBLE TASK ENDPOINTS =====

@router.get("/api/agents/{agent_id}/current-task")
def get_current_task(agent_id: str, db: Session = Depends(get_db)):
    """Get current task for an agent - NEW SYSTEM"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Find next pending or in-progress task
    next_task = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status.in_(['pending', 'in_progress'])
    ).order_by(TaskProgress.assigned_at).first()
    
    if not next_task:
        # Check if there are any completed tasks to show progress
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'completed'
        ).count()
        
        return {
            "completed": True,
            "message": "All tasks completed",
            "total_completed": completed_tasks,
            "total_tasks": total_tasks
        }
    
    # Mark task as in_progress if it was pending
    if next_task.status == 'pending':
        next_task.status = 'in_progress'
        next_task.started_at = datetime.utcnow()
        db.commit()
    
    # Get task statistics
    total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
    completed_tasks = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'completed'
    ).count()
    current_index = completed_tasks  # Current task index
    
    return {
        "task": {
            "id": next_task.id,
            "agent_id": next_task.agent_id,
            "image_path": next_task.image_path,
            "image_filename": next_task.image_filename,
            "status": next_task.status,
            "assigned_at": next_task.assigned_at.isoformat()
        },
        "image_url": next_task.image_path,
        "image_name": next_task.image_filename,
        "current_index": current_index,
        "total_images": total_tasks,
        "progress": f"{current_index + 1}/{total_tasks}"
    }

@router.get("/api/agents/{agent_id}/tasks")
def get_agent_tasks(agent_id: str, db: Session = Depends(get_db)):
    """Get all tasks for an agent - NEW SYSTEM"""
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

@router.post("/api/agents/{agent_id}/submit")
async def submit_task_data(
    agent_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Submit task data - NEW SYSTEM compatible with form data"""
    # Verify agent exists
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get the current in-progress task for this agent
    current_task = db.query(TaskProgress).filter(
        TaskProgress.agent_id == agent_id,
        TaskProgress.status == 'in_progress'
    ).order_by(TaskProgress.assigned_at).first()
    
    if not current_task:
        # If no in-progress task, try to find a pending one
        current_task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'pending'
        ).order_by(TaskProgress.assigned_at).first()
    
    if not current_task:
        raise HTTPException(status_code=404, detail="No active task found for submission")
    
    # Get form data from request
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            form_data = await request.json()
        else:
            form_data_raw = await request.form()
            form_data = dict(form_data_raw)
            # Remove agent_id and task_id from form data
            form_data.pop('agent_id', None)
            form_data.pop('task_id', None)
    except Exception as e:
        print(f"Error parsing form data: {e}")
        form_data = {}
    
    # Create submitted form record
    submission = SubmittedForm(
        agent_id=agent_id,
        task_id=current_task.id,
        image_filename=current_task.image_filename,
        form_data=form_data,  # Store as JSON object
        submitted_at=datetime.utcnow()
    )
    db.add(submission)
    
    # Mark task as completed
    current_task.status = 'completed'
    current_task.completed_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "message": "Task submitted successfully", 
        "success": True,
        "submission_id": submission.id,
        "task_id": current_task.id
    }

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
