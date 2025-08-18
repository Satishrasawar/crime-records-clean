# agent_routes.py - Part 1: Imports and Setup
from fastapi import APIRouter, Form, Depends, HTTPException, UploadFile, File, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Agent, TaskProgress, SubmittedForm, AgentSession, Admin
from app.schemas import AgentStatusUpdateSchema
from app.security import hash_password, verify_password
import os
import secrets
import string
from datetime import datetime, timedelta
import json
import zipfile
import io
import pandas as pd
from typing import Optional
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def generate_agent_credentials():
    """Generate unique agent ID and secure password"""
    agent_id = "AGT" + "".join(secrets.choice(string.digits) for _ in range(6))
    password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
    return agent_id, password

def get_agent_image_files(agent_id: str):
    """Get all image files assigned to a specific agent"""
    agent_folder = f"static/task_images/agent_{agent_id}"
    if not os.path.exists(agent_folder):
        # Fallback to general folder
        agent_folder = "static/task_images/crime_records_wide"
    
    if not os.path.exists(agent_folder):
        return []
    
    return sorted([f for f in os.listdir(agent_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    # agent_routes.py - Part 2: Admin Login Functions

# ===================== ADMIN LOGIN ENDPOINTS =====================

@router.post("/api/admin/login")
@limiter.limit("5/minute")
async def admin_login(request: Request, db: Session = Depends(get_db)):
    """FIXED Admin login endpoint"""
    try:
        print("🔐 FIXED Admin login attempt received")
        
        # Parse request data
        try:
            data = await request.json()
            print(f"📨 Login data received: {list(data.keys()) if data else 'No data'}")
        except Exception as parse_error:
            print(f"❌ Failed to parse login data: {parse_error}")
            raise HTTPException(status_code=400, detail="Invalid JSON data")
        
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        
        print(f"🔍 Login attempt - Username: '{username}', Password provided: {bool(password)}")

        if not username or not password:
            print("❌ Missing username or password")
            raise HTTPException(status_code=400, detail="Username and password are required")

        print(f"🔍 Querying database for admin: {username}")
        admin = db.query(Admin).filter(Admin.username == username).first()
        
        if not admin:
            print(f"❌ Admin user '{username}' not found in database")
            # Check if any admins exist
            admin_count = db.query(Admin).count()
            print(f"📊 Total admins in database: {admin_count}")
            if admin_count == 0:
                print("⚠️ No admin users exist in database!")
                # Auto-create admin for testing
                try:
                    print("🔧 Auto-creating admin user...")
                    hashed_password = hash_password("admin123")
                    new_admin = Admin(
                        username="admin",
                        hashed_password=hashed_password,
                        email="admin@agent-task-system.com",
                        is_active=True,
                        created_at=datetime.utcnow()
                    )
                    db.add(new_admin)
                    db.commit()
                    db.refresh(new_admin)
                    admin = new_admin
                    print("✅ Auto-created admin user successfully!")
                except Exception as create_error:
                    print(f"❌ Failed to auto-create admin: {create_error}")
                    raise HTTPException(status_code=401, detail="Invalid credentials")
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        print(f"✅ Admin user found: {admin.username}, Active: {admin.is_active}")
        
        # Check if admin is active
        if not admin.is_active:
            print(f"❌ Admin {username} is not active")
            raise HTTPException(status_code=403, detail="Account is not active")

        # Verify password
        try:
            # Handle both hashed and plain passwords for debugging
            if admin.hashed_password.startswith('$2b$') or admin.hashed_password.startswith('$2a$'):
                # This is a bcrypt hash
                password_valid = verify_password(password, admin.hashed_password)
                print(f"🔐 Bcrypt password verification result: {password_valid}")
            else:
                # Might be plain text (for debugging) - re-hash it
                print("⚠️ Plain text password detected - converting to hash")
                if admin.hashed_password == password:
                    # Update to hashed password
                    admin.hashed_password = hash_password(password)
                    db.commit()
                    password_valid = True
                    print("✅ Password updated to hash and verified")
                else:
                    password_valid = False
                    print("❌ Plain text password doesn't match")
            
        except Exception as verify_error:
            print(f"❌ Password verification error: {verify_error}")
            # Fallback: try direct comparison for debugging
            if admin.hashed_password == password:
                password_valid = True
                print("✅ Fallback: Direct password match")
            else:
                password_valid = False
                print("❌ Fallback: Direct password mismatch")
        
        if not password_valid:
            print(f"❌ Invalid password for admin {username}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
            # agent_routes.py - Part 3: Admin Management and Auth Endpoints

        # Create access token
        try:
            access_token = create_access_token(data={"sub": username})
            print(f"✅ JWT token created successfully for {username}")
        except Exception as token_error:
            print(f"❌ Token creation error: {token_error}")
            # Return success without token for debugging
            access_token = "debug_token_" + secrets.token_urlsafe(32)
        
        print(f"🎉 Admin login successful for {username}")

        return {
            "success": True,
            "access_token": access_token,
            "token_type": "bearer",
            "message": "Login successful",
            "admin_info": {
                "username": admin.username,
                "email": admin.email,
                "is_active": admin.is_active
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected admin login error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# SIMPLE TEST LOGIN ENDPOINT (NO JWT)
@router.post("/api/admin/simple-login")
@limiter.limit("10/minute")
async def simple_admin_login(request: Request, db: Session = Depends(get_db)):
    """Simple admin login for testing without JWT complexity"""
    try:
        data = await request.json()
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        
        print(f"🧪 Simple login test - Username: '{username}'")
        
        if not username or not password:
            return {"success": False, "message": "Username and password required"}
        
        # Hardcoded test for debugging
        if username == "admin" and password == "admin123":
            print("✅ Hardcoded credentials matched!")
            return {
                "success": True,
                "message": "Login successful (hardcoded test)",
                "access_token": "test_token_123",
                "token_type": "bearer"
            }
        
        # Try database
        admin = db.query(Admin).filter(Admin.username == username).first()
        if admin:
            print(f"👤 Found admin in database: {admin.username}")
            # Try multiple password checks
            if (admin.hashed_password == password or 
                (admin.hashed_password.startswith('$') and verify_password(password, admin.hashed_password))):
                return {
                    "success": True,
                    "message": "Login successful (database)",
                    "access_token": "db_token_123",
                    "token_type": "bearer"
                }
        
        return {"success": False, "message": "Invalid credentials"}
        
    except Exception as e:
        print(f"❌ Simple login error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

# ADMIN STATUS CHECK
@router.get("/api/admin/status")
@limiter.limit("20/minute")
async def check_admin_status(request: Request, db: Session = Depends(get_db)):
    """Check admin system status"""
    try:
        admin_count = db.query(Admin).count()
        
        if admin_count == 0:
            # Auto-create admin
            try:
                hashed_password = hash_password("admin123")
                new_admin = Admin(
                    username="admin",
                    hashed_password=hashed_password,
                    email="admin@agent-task-system.com",
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.add(new_admin)
                db.commit()
                admin_count = 1
                print("✅ Auto-created admin user")
            except Exception as create_error:
                print(f"❌ Failed to create admin: {create_error}")
        
        admins = db.query(Admin).all()
        admin_list = []
        
        for admin in admins:
            admin_list.append({
                "username": admin.username,
                "email": admin.email,
                "is_active": admin.is_active,
                "has_hash": admin.hashed_password.startswith('$') if admin.hashed_password else False,
                "created_at": admin.created_at.isoformat() if admin.created_at else None
            })
        
        return {
            "admin_count": admin_count,
            "admins": admin_list,
            "test_credentials": {
                "username": "admin",
                "password": "admin123"
            },
            "status": "ready"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "error"}

# CORS PREFLIGHT HANDLER
@router.options("/api/admin/login")
async def admin_login_options(request: Request):
    """Handle CORS preflight for admin login"""
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )

# ===================== ADMIN DEBUG ENDPOINTS =====================
@router.post("/api/admin/debug-create")
@limiter.limit("1/minute")
async def debug_create_admin(request: Request, db: Session = Depends(get_db)):
    """Debug endpoint to create admin user"""
    try:
        print("🔧 Debug: Creating admin user")
        
        # Check if admin already exists
        existing_admin = db.query(Admin).filter(Admin.username == "admin").first()
        if existing_admin:
            print(f"⚠️ Admin already exists: {existing_admin.username}, Active: {existing_admin.is_active}")
            return {
                "message": "Admin already exists",
                "username": existing_admin.username,
                "email": existing_admin.email,
                "is_active": existing_admin.is_active,
                "created_at": existing_admin.created_at.isoformat() if existing_admin.created_at else None
            }
        
        # Create new admin
        print("🔧 Creating new admin user")
        hashed_password = hash_password("admin123")
        print(f"🔐 Generated hash: {hashed_password[:20]}...")
        
        new_admin = Admin(
            username="admin",
            hashed_password=hashed_password,
            email="admin@agent-task-system.com",
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)
        
        print(f"✅ Admin created successfully: ID {new_admin.id}")
        
        return {
            "success": True,
            "message": "Admin user created successfully!",
            "credentials": {
                "username": "admin",
                "password": "admin123"
            },
            "admin_info": {
                "id": new_admin.id,
                "username": new_admin.username,
                "email": new_admin.email,
                "is_active": new_admin.is_active
            }
        }
        
    except Exception as e:
        print(f"❌ Error creating admin: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create admin: {str(e)}")

@router.get("/api/admin/debug-check")
@limiter.limit("10/minute")
async def debug_check_admin(request: Request, db: Session = Depends(get_db)):
    """Debug endpoint to check admin status"""
    try:
        print("🔍 Debug: Checking admin status")
        
        # Get all admins
        admins = db.query(Admin).all()
        admin_count = len(admins)
        
        print(f"📊 Found {admin_count} admin users")
        
        if admin_count == 0:
            return {
                "admin_exists": False,
                "admin_count": 0,
                "message": "No admin users found. Use /api/admin/debug-create to create one."
            }
        
        admin_list = []
        for admin in admins:
            admin_info = {
                "id": admin.id,
                "username": admin.username,
                "email": admin.email,
                "is_active": admin.is_active,
                "created_at": admin.created_at.isoformat() if admin.created_at else None,
                "hash_preview": admin.hashed_password[:20] + "..." if admin.hashed_password else "No hash"
            }
            admin_list.append(admin_info)
            print(f"👤 Admin: {admin.username}, Active: {admin.is_active}")
        
        return {
            "admin_exists": True,
            "admin_count": admin_count,
            "admins": admin_list,
            "message": f"Found {admin_count} admin user(s). Use credentials: admin / admin123"
        }
        
    except Exception as e:
        print(f"❌ Error checking admin: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
        # agent_routes.py - Part 4: Agent Management Functions

# ===================== AGENT MANAGEMENT ENDPOINTS =====================

@router.get("/api/agents")
def get_all_agents(db: Session = Depends(get_db)):
    """Get all agents with their statistics"""
    agents = db.query(Agent).all()
    result = []
    
    for agent in agents:
        # Get task completion count
        completed_count = db.query(SubmittedForm).filter(SubmittedForm.agent_id == agent.agent_id).count()
        
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
            "password": agent.password,  # Show plain password for admin
            "status": agent.status,
            "tasks_completed": completed_count,
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
    """Get agent password for admin (this should be protected in production)"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent_id,
        "password": agent.password,  # Return plain password
        "message": f"Password for agent {agent_id}: {agent.password}",
        "can_reset": True
    }

@router.post("/api/admin/reset-password/{agent_id}")
def reset_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Reset agent password and return new one"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Generate new password
    _, new_password = generate_agent_credentials()
    
    # Update password (store as plain text for now)
    agent.password = new_password
    db.commit()
    
    return {
        "agent_id": agent_id,
        "new_password": new_password,
        "message": "Password reset successfully"
    }

@router.patch("/api/agents/{agent_id}/status")
def update_agent_status(agent_id: str, status_data: AgentStatusUpdateSchema, db: Session = Depends(get_db)):
    """Update agent status"""
    # Find by agent_id string, not integer id
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.status = status_data.status
    db.commit()
    return {"message": "Agent status updated successfully"}

@router.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete an agent"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(agent)
    db.commit()
    return {"message": "Agent deleted successfully"}

# ===================== AGENT LOGIN AND SESSION MANAGEMENT =====================

@router.post("/api/agents/login")
async def login_agent(
    agent_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """FIXED: Login agent using consistent password system"""
    print(f"🔑 Login attempt for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Use plain password comparison (since we store plain passwords now)
    if agent.password != password:
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
            ip_address="127.0.0.1",  # Should get real IP in production
            user_agent="Web Browser"  # Should get real UA in production
        )
        
        db.add(new_session)
        db.commit()
        
        print(f"✅ Login successful for {agent_id}")
        
        return {
            "success": True,
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "name": agent.name,
            "email": agent.email,
            "status": agent.status
        }
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return {
            "success": True,
            "message": "Login successful", 
            "agent_id": agent.agent_id, 
            "name": agent.name,
            "session_warning": "Session tracking unavailable"
        }

@router.post("/api/agents/{agent_id}/logout")
async def logout_agent(agent_id: str, db: Session = Depends(get_db)):
    """Logout an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
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
    """Force logout an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
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
    # agent_routes.py - Part 5: Task Management and Statistics

# ===================== TASK MANAGEMENT =====================

@router.get("/api/agents/{agent_id}/current-task")
def get_current_task(agent_id: str, db: Session = Depends(get_db)):
    """Get current task for an agent"""
    print(f"📋 Getting current task for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
    if not progress:
        progress = TaskProgress(agent_id=agent_id, current_index=0)
        db.add(progress)
        db.commit()
        db.refresh(progress)
        print(f"📋 Created new progress tracker for {agent_id}")

    image_files = get_agent_image_files(agent_id)
    
    if not image_files:
        print(f"📋 No images found for agent {agent_id}")
        return {"message": "No tasks assigned", "completed": True}
    
    if progress.current_index >= len(image_files):
        completed_count = db.query(SubmittedForm).filter(SubmittedForm.agent_id == agent_id).count()
        print(f"✅ All tasks completed for {agent_id}: {completed_count} submissions")
        return {
            "message": "All tasks completed", 
            "completed": True,
            "total_completed": completed_count
        }

    current_image = image_files[progress.current_index]
    
    agent_folder = f"static/task_images/agent_{agent_id}"
    if os.path.exists(agent_folder) and current_image in os.listdir(agent_folder):
        image_url = f"/static/task_images/agent_{agent_id}/{current_image}"
    else:
        image_url = f"/static/task_images/crime_records_wide/{current_image}"

    print(f"📊 Current task for {agent_id}: {progress.current_index + 1}/{len(image_files)} - {current_image}")

    return {
        "image_url": image_url,
        "image_name": current_image,
        "progress": f"{progress.current_index + 1}/{len(image_files)}",
        "current_index": progress.current_index,
        "total_images": len(image_files),
        "task_number": progress.current_index + 1,
        "completed": False
    }

# ===================== TASK SUBMISSION =====================

@router.post("/api/agents/{agent_id}/submit")
async def submit_task_data(
    agent_id: str,
    DR_NO: str = Form(...),
    Date_Rptd: str = Form(...),
    DATE_OCC: str = Form(...),
    TIME_OCC: str = Form(...),
    Unique_Identifier: str = Form(...),
    AREA_NAME: str = Form(...),
    Rpt_Dist_No: str = Form(...),
    VIN: str = Form(...),
    Crm: str = Form(...),
    Crm_Cd_Desc: str = Form(...),
    Mocodes: str = Form(...),
    Vict_Age: str = Form(...),
    Geolocation: str = Form(...),
    DEPARTMENT: str = Form(...),
    Premis_Cd: str = Form(...),
    Premis_Desc: str = Form(...),
    ARREST_KEY: str = Form(...),
    PD_DESC: str = Form(...),
    CCD_LONCOD: str = Form(...),
    Status_Desc: str = Form(...),
    LAW_CODE: str = Form(...),
    SubAgency: str = Form(...),
    Charge: str = Form(...),
    Race: str = Form(...),
    LOCATION: str = Form(...),
    SeqID: str = Form(...),
    LAT: str = Form(...),
    LON: str = Form(...),
    Point: str = Form(...),
    Shape__Area: str = Form(...),
    db: Session = Depends(get_db)
):
    """Submit task form data"""
    print(f"📤 Submit request received for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
    current_image_name = None
    
    if progress:
        image_files = get_agent_image_files(agent_id)
        if progress.current_index < len(image_files):
            current_image_name = image_files[progress.current_index]
        print(f"📋 Current task index: {progress.current_index}, Image: {current_image_name}")
    
    form_data = {
        "DR_NO": DR_NO, "Date_Rptd": Date_Rptd, "DATE_OCC": DATE_OCC, "TIME_OCC": TIME_OCC,
        "Unique_Identifier": Unique_Identifier, "AREA_NAME": AREA_NAME, "Rpt_Dist_No": Rpt_Dist_No,
        "VIN": VIN, "Crm": Crm, "Crm_Cd_Desc": Crm_Cd_Desc, "Mocodes": Mocodes,
        "Vict_Age": Vict_Age, "Geolocation": Geolocation, "DEPARTMENT": DEPARTMENT,
        "Premis_Cd": Premis_Cd, "Premis_Desc": Premis_Desc, "ARREST_KEY": ARREST_KEY,
        "PD_DESC": PD_DESC, "CCD_LONCOD": CCD_LONCOD, "Status_Desc": Status_Desc,
        "LAW_CODE": LAW_CODE, "SubAgency": SubAgency, "Charge": Charge, "Race": Race,
        "LOCATION": LOCATION, "SeqID": SeqID, "LAT": LAT, "LON": LON,
        "Point": Point, "Shape__Area": Shape__Area,
        "image_name": current_image_name
    }
    
    print(f"📝 Form data prepared with {len([k for k, v in form_data.items() if v and str(v).strip()])} filled fields")
    
    try:
        submission = SubmittedForm(
            agent_id=agent_id,
            form_data=json.dumps(form_data),
            submitted_at=datetime.utcnow()
        )
        db.add(submission)
        
        if progress:
            progress.current_index += 1
            progress.updated_at = datetime.utcnow()
        else:
            progress = TaskProgress(agent_id=agent_id, current_index=1)
            db.add(progress)
        
        db.commit()
        
        print(f"✅ Task submitted successfully for agent {agent_id}, new index: {progress.current_index}")
        
        return {
            "message": "Task submitted successfully", 
            "success": True,
            "submission_id": submission.id,
            "next_task_index": progress.current_index
        }
    except Exception as db_error:
        print(f"❌ Database error during submission: {db_error}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save submission: {str(db_error)}")

# ===================== STATISTICS AND REPORTING =====================

@router.get("/api/admin/statistics")
def get_admin_statistics(db: Session = Depends(get_db)):
    """Get admin dashboard statistics"""
    total_agents = db.query(Agent).count()
    active_agents = db.query(Agent).filter(Agent.status == "active").count()
    total_submissions = db.query(SubmittedForm).count()
    
    total_tasks = 0
    for agent in db.query(Agent).all():
        agent_images = get_agent_image_files(agent.agent_id)
        total_tasks += len(agent_images)
    
    pending_tasks = max(0, total_tasks - total_submissions)
    
    return {
        "total_agents": total_agents,
        "active_agents": active_agents,
        "total_tasks": total_tasks,
        "completed_tasks": total_submissions,
        "pending_tasks": pending_tasks
    }

@router.get("/api/admin/session-report")
async def get_session_report(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get session report"""
    try:
        query = db.query(AgentSession)
        
        if agent_id:
            query = query.filter(AgentSession.agent_id == agent_id)
        
        if date_from:
            query = query.filter(AgentSession.login_time >= datetime.strptime(date_from, '%Y-%m-%d'))
        
        if date_to:
            query = query.filter(AgentSession.login_time <= datetime.strptime(date_to, '%Y-%m-%d'))
        
        sessions = query.order_by(AgentSession.login_time.desc()).all()
        
        result = []
        for session in sessions:
            agent = db.query(Agent).filter(Agent.agent_id == session.agent_id).first()
            result.append({
                "session_id": session.id,
                "agent_id": session.agent_id,
                "agent_name": agent.name if agent else "Unknown",
                "login_time": session.login_time.isoformat() if session.login_time else None,
                "logout_time": session.logout_time.isoformat() if session.logout_time else None,
                "duration_minutes": session.duration_minutes,
            # agent_routes.py - Part 6: Data Export and Final Functions

                "is_active": session.logout_time is None
            })
        
        return result
    except Exception as e:
        print(f"❌ Error in session report: {e}")
        raise HTTPException(status_code=500, detail=f"Session report failed: {str(e)}")

# ===================== FILE UPLOAD AND TASK ASSIGNMENT =====================

@router.post("/api/admin/upload-tasks")
async def upload_task_images(
    agent_id: str = Form(...),
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload ZIP file with task images for an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if not zip_file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    try:
        agent_dir = f"static/task_images/agent_{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)
        
        zip_content = await zip_file.read()
        
        images_processed = 0
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue
                
                file_ext = os.path.splitext(file_info.filename.lower())[1]
                if file_ext not in supported_extensions:
                    continue
                
                with zip_ref.open(file_info) as source_file:
                    safe_filename = os.path.basename(file_info.filename)
                    if not safe_filename:
                        continue
                    
                    counter = 1
                    original_name = safe_filename
                    while os.path.exists(os.path.join(agent_dir, safe_filename)):
                        name, ext = os.path.splitext(original_name)
                        safe_filename = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    image_path = os.path.join(agent_dir, safe_filename)
                    with open(image_path, 'wb') as dest_file:
                        dest_file.write(source_file.read())
                    
                    images_processed += 1
                    
                    if images_processed >= 5000:
                        break
        
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        if progress:
            progress.current_index = 0
        else:
            progress = TaskProgress(agent_id=agent_id, current_index=0)
            db.add(progress)
        
        db.commit()
        
        return {
            "message": "Images uploaded successfully",
            "agent_id": agent_id,
            "images_processed": images_processed,
            "agent_directory": agent_dir
        }
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ===================== DATA EXPORT AND PREVIEW =====================

@router.get("/api/admin/export-excel")
def export_to_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export submitted data to Excel"""
    try:
        query = db.query(SubmittedForm)
        
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
                to_date = to_date.replace(hour=23, minute=59, second=59)
                query = query.filter(SubmittedForm.submitted_at <= to_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        submissions = query.order_by(SubmittedForm.submitted_at.desc()).all()
        
        if not submissions:
            raise HTTPException(status_code=404, detail="No data found with current filters")
        
        excel_data = []
        for submission in submissions:
            try:
                form_data = json.loads(submission.form_data)
                
                row_data = {
                    'Submission_ID': submission.id,
                    'Agent_ID': submission.agent_id,
                    'Submitted_At': submission.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'Image_Name': form_data.get('image_name', 'Unknown')
                }
                
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
        
        try:
            import pandas as pd
        except ImportError:
            raise HTTPException(status_code=500, detail="pandas not installed. Run: pip install pandas")
        
        df = pd.DataFrame(excel_data)
        
        output = io.BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Crime Records Data', index=False)
                
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
                    
                    adjusted_width = min(max(max_length + 2, 10), 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating Excel file: {str(e)}")
        
        output.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crime_records_export_{timestamp}.xlsx"
        
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
        submission_count = db.query(SubmittedForm).count()
        session_count = db.query(AgentSession).count()
        
        return {
            "success": True,
            "message": f"Data available - Agents: {agent_count}, Submissions: {submission_count}, Sessions: {session_count}",
            "counts": {
                "agents": agent_count,
                "submissions": submission_count,
                "sessions": session_count
            }
        }
        
    except Exception as e:
        print(f"❌ Error testing data: {e}")
        raise HTTPException(status_code=500, detail=f"Data test failed: {str(e)}")

# ===================== ADVANCED ADMIN FUNCTIONS =====================

@router.get("/api/admin/agent-details/{agent_id}")
async def get_agent_details(agent_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific agent"""
    try:
        # Get agent
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get task progress
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        
        # Get total assigned images
        assigned_images = get_agent_image_files(agent_id)
        
        # Get completed tasks count
        completed_count = db.query(SubmittedForm).filter(SubmittedForm.agent_id == agent_id).count()
        
        # Get recent submissions (last 10)
        recent_submissions = db.query(SubmittedForm).filter(
            SubmittedForm.agent_id == agent_id
        ).order_by(SubmittedForm.submitted_at.desc()).limit(10).all()
        
        # Get session information
        sessions = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id
        ).order_by(AgentSession.login_time.desc()).limit(10).all()
        
        # Calculate progress percentage
        total_images = len(assigned_images)
        progress_percentage = (completed_count / total_images * 100) if total_images > 0 else 0
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "email": agent.email,
            "mobile": agent.mobile,
            "status": agent.status,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "progress": {
                "current_index": progress.current_index if progress else 0,
                "total_images": total_images,
                "completed_count": completed_count,
                "progress_percentage": round(progress_percentage, 2),
                "remaining_tasks": max(0, total_images - completed_count)
            },
            "recent_submissions": [
                {
                    "id": sub.id,
                    "submitted_at": sub.submitted_at.isoformat(),
                    "image_name": json.loads(sub.form_data).get('image_name', 'Unknown')
                } for sub in recent_submissions
            ],
            "recent_sessions": [
                {
                    "login_time": sess.login_time.isoformat() if sess.login_time else None,
                    "logout_time": sess.logout_time.isoformat() if sess.logout_time else None,
                    "duration_minutes": sess.duration_minutes,
                    "is_active": sess.logout_time is None
                } for sess in sessions
            ]
        }
        
    except Exception as e:
        print(f"❌ Error getting agent details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent details: {str(e)}")

@router.get("/api/admin/system-health")
async def check_system_health(db: Session = Depends(get_db)):
    """Check system health and database connectivity"""
    try:
        # Test database connection
        agent_count = db.query(Agent).count()
        
        # Check static directory
        static_exists = os.path.exists("static/task_images")
        
        # Check for any active sessions
        active_sessions = db.query(AgentSession).filter(AgentSession.logout_time.is_(None)).count()
        
        return {
            "status": "healthy",
            "database_connected": True,
            "static_directory_exists": static_exists,
            "total_agents": agent_count,
            "active_sessions": active_sessions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "database_connected": False,
            "timestamp": datetime.utcnow().isoformat()
        }

