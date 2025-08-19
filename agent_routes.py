# agent_routes.py
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
from datetime import datetime, timedelta, date
import json
import zipfile
import io
import pandas as pd
from typing import Optional
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import re
import shutil

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

def generate_unique_agent_id(db: Session):
    """Generate a unique agent ID with increased attempts to reduce collision failures"""
    max_attempts = 100
    for attempt in range(max_attempts):
        agent_number = secrets.randbelow(900000) + 100000
        agent_id = f"AGT{agent_number}"
        existing = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not existing:
            print(f"✅ Generated unique agent ID: {agent_id}")
            return agent_id
    # Fallback to timestamp-based ID
    import time
    fallback_id = f"AGT{str(int(time.time()))[-6:]}"
    existing_fallback = db.query(Agent).filter(Agent.agent_id == fallback_id).first()
    if existing_fallback:
        raise HTTPException(status_code=500, detail="Failed to generate unique agent ID even with fallback")
    print(f"⚠️ Using fallback agent ID: {fallback_id}")
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

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number format (more permissive)"""
    clean_mobile = re.sub(r'[\s\-\(\)]', '', mobile)
    return re.match(r'^\+?\d{10,15}$', clean_mobile) is not None

def get_agent_image_files(agent_id: str):
    """Get all image files assigned to a specific agent"""
    agent_folder = f"static/task_images/agent_{agent_id}"
    if not os.path.exists(agent_folder):
        agent_folder = "static/task_images/crime_records_wide"
    if not os.path.exists(agent_folder):
        return []
    return sorted([f for f in os.listdir(agent_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# ===================== AGENT REGISTRATION ENDPOINTS =====================

@router.post("/api/agents/register")
@limiter.limit("3/minute")
async def register_agent(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(None),
    country: str = Form(None),
    gender: str = Form(None),
    db: Session = Depends(get_db)
):
    """Register a new agent (debugged: collect all validation errors, better ID/password generation)"""
    try:
        print(f"🆕 Agent registration attempt: {name}, {email}")
        
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
        
        # Create new agent
        agent_data = {
            'agent_id': agent_id,
            'name': name,
            'email': email,
            'mobile': mobile,
            'password': password,  # Store plain text for now, as per model
            'status': "pending",  # Align with original
            'created_at': datetime.utcnow()
        }
        
        if dob:
            agent_data['dob'] = dob_date
        if country:
            agent_data['country'] = country.strip()
        if gender:
            agent_data['gender'] = gender
        
        new_agent = Agent(**agent_data)
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        print(f"✅ Agent registered successfully: {agent_id}")
        
        # Build response
        response_data = {
            "success": True,
            "message": "Registration successful! Please save your credentials.",
            "agent_id": agent_id,
            "password": password,
            "name": name,
            "email": email,
            "mobile": mobile,
            "status": "pending",
            "instructions": [
                "Save your Agent ID and Password securely",
                "Your account is pending approval",
                "Contact admin to activate your account",
                "Use these credentials to log in once activated"
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
        print(f"❌ Registration error: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/api/agents/check-availability")
@limiter.limit("10/minute")
async def check_availability(
    request: Request,
    email: str = Form(None),
    mobile: str = Form(None),
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
        print(f"❌ Availability check error: {e}")
        return {"available": False, "message": "Check failed"}

@router.get("/api/agents/test-registration")
@limiter.limit("5/minute")
async def test_registration_system(request: Request, db: Session = Depends(get_db)):
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
            "optional_fields": ["dob", "country", "gender"],
            "validation_rules": {
                "name": "Minimum 2 characters",
                "email": "Valid email format",
                "mobile": "10-15 digits, international format supported"
            }
        }
        
    except Exception as e:
        return {
            "system_status": "error",
            "error": str(e),
            "database_connected": False
        }

# ===================== ADMIN LOGIN ENDPOINTS =====================

@router.post("/api/admin/login")
@limiter.limit("5/minute")
async def admin_login(request: Request, db: Session = Depends(get_db)):
    """Admin login endpoint (debugged: enforce hashed passwords)"""
    try:
        print("🔐 Admin login attempt received")
        
        data = await request.json()
        print(f"📨 Login data received: {list(data.keys()) if data else 'No data'}")
        
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
            admin_count = db.query(Admin).count()
            print(f"📊 Total admins in database: {admin_count}")
            if admin_count == 0:
                print("⚠️ No admin users exist in database!")
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
        
        if not admin.is_active:
            print(f"❌ Admin {username} is not active")
            raise HTTPException(status_code=403, detail="Account is not active")

        if not verify_password(password, admin.hashed_password):
            print("❌ Password verification failed")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        print("✅ Password verified successfully")
        
        access_token = create_access_token(data={"sub": admin.username})
        
        return {
            "success": True,
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "username": admin.username,
                "email": admin.email,
                "role": "admin"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# ===================== AGENT MANAGEMENT ENDPOINTS =====================

@router.get("/api/agents")
@limiter.limit("50/minute")
async def list_agents(request: Request, db: Session = Depends(get_db)):
    """List all agents with their statistics"""
    try:
        agents = db.query(Agent).all()
        agent_list = []
        
        for agent in agents:
            total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent.agent_id).count()
            completed_tasks = db.query(TaskProgress).filter(
                TaskProgress.agent_id == agent.agent_id,
                TaskProgress.status == 'completed'
            ).count()
            
            latest_session = db.query(AgentSession).filter(
                AgentSession.agent_id == agent.agent_id
            ).order_by(AgentSession.login_time.desc()).first()
            
            agent_list.append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "password": "********",
                "status": agent.status,
                "tasks_completed": completed_tasks,
                "total_tasks": total_tasks,
                "last_login": latest_session.login_time.isoformat() if latest_session else None,
                "is_currently_logged_in": latest_session.is_active if latest_session else False
            })
        
        return agent_list
        
    except Exception as e:
        print(f"❌ Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")

@router.post("/api/agents/{agent_id}/status")
@limiter.limit("10/minute")
async def update_agent_status(
    agent_id: str,
    status_update: AgentStatusUpdateSchema,
    db: Session = Depends(get_db)
):
    """Update agent status"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent.status = status_update.status
        agent.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "success": True,
            "message": f"Agent {agent_id} status updated to {status_update.status}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")

# ===================== TASK MANAGEMENT ENDPOINTS =====================

@router.get("/api/agents/{agent_id}/current-task")
@limiter.limit("50/minute")
async def get_current_task(agent_id: str, request: Request, db: Session = Depends(get_db)):
    """Get current task for an agent"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        if not progress:
            progress = TaskProgress(agent_id=agent_id, current_index=0)
            db.add(progress)
            db.commit()
            db.refresh(progress)
        
        image_files = get_agent_image_files(agent_id)
        if not image_files:
            return {
                "completed": True,
                "message": "No tasks available",
                "agent_id": agent_id,
                "current_index": progress.current_index,
                "progress": f"{progress.current_index}/0"
            }
        
        if progress.current_index >= len(image_files):
            return {
                "completed": True,
                "message": "All tasks completed",
                "agent_id": agent_id,
                "current_index": progress.current_index,
                "progress": f"{progress.current_index}/{len(image_files)}"
            }
        
        current_image = image_files[progress.current_index]
        image_url = f"/static/task_images/agent_{agent_id}/{current_image}"
        
        return {
            "completed": False,
            "message": "Task available",
            "agent_id": agent_id,
            "current_index": progress.current_index,
            "total_tasks": len(image_files),
            "progress": f"{progress.current_index}/{len(image_files)}",
            "image_url": image_url,
            "image_name": current_image
        }
        
    except Exception as e:
        print(f"❌ Error getting current task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task: {str(e)}")

@router.post("/api/agents/{agent_id}/submit")
@limiter.limit("50/minute")
async def submit_task_form(agent_id: str, request: Request, db: Session = Depends(get_db)):
    """Submit completed task form"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        form_data = await request.json()
        image_name = form_data.get("image_name")
        if not image_name:
            raise HTTPException(status_code=400, detail="Image name is required")
        
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        if not progress:
            progress = TaskProgress(agent_id=agent_id, current_index=0)
            db.add(progress)
        
        image_files = get_agent_image_files(agent_id)
        if progress.current_index >= len(image_files) or image_files[progress.current_index] != image_name:
            raise HTTPException(status_code=400, detail="Invalid task submission")
        
        new_submission = SubmittedForm(
            agent_id=agent_id,
            task_id=progress.id,
            image_filename=image_name,
            form_data=json.dumps(form_data),
            submitted_at=datetime.utcnow()
        )
        
        progress.current_index += 1
        progress.status = "completed" if progress.current_index >= len(image_files) else "in_progress"
        progress.updated_at = datetime.utcnow()
        
        db.add(new_submission)
        db.commit()
        
        return {
            "success": True,
            "message": "Task submitted successfully",
            "agent_id": agent_id,
            "image_name": image_name,
            "next_index": progress.current_index
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting task: {e}")
        raise HTTPException(status_code=500, detail=f"Task submission failed: {str(e)}")

# ===================== UPLOAD ENDPOINTS =====================

@router.post("/api/admin/upload-chunk")
@limiter.limit("20/minute")
async def upload_chunk(
    request: Request,
    file: UploadFile = File(...),
    upload_id: str = Form(None),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    agent_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Handle chunked file upload"""
    try:
        if not upload_id:
            upload_id = str(uuid.uuid4())
            upload_sessions[upload_id] = {
                "chunks": {},
                "total_chunks": total_chunks,
                "agent_id": agent_id,
                "created_at": datetime.now(),
                "temp_dir": os.path.join(CHUNK_UPLOAD_DIR, upload_id)
            }
            os.makedirs(upload_sessions[upload_id]["temp_dir"], exist_ok=True)
        
        session = upload_sessions.get(upload_id)
        if not session:
            raise HTTPException(status_code=400, detail="Invalid upload session")
        
        chunk_path = os.path.join(session["temp_dir"], f"chunk_{chunk_index}")
        async with aiofiles.open(chunk_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        session["chunks"][chunk_index] = chunk_path
        
        return {
            "success": True,
            "upload_id": upload_id,
            "chunk_index": chunk_index,
            "message": f"Chunk {chunk_index + 1}/{total_chunks} uploaded successfully"
        }
        
    except Exception as e:
        print(f"❌ Error uploading chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk upload failed: {str(e)}")

@router.post("/api/admin/finalize-upload")
@limiter.limit("10/minute")
async def finalize_upload(
    request: Request,
    upload_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Finalize chunked upload and process ZIP file"""
    try:
        session = upload_sessions.get(upload_id)
        if not session:
            raise HTTPException(status_code=400, detail="Invalid upload session")
        
        agent_id = session["agent_id"]
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent_dir = f"static/task_images/agent_{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)
        
        combined_zip_path = os.path.join(session["temp_dir"], "combined.zip")
        with open(combined_zip_path, 'wb') as combined_zip:
            for i in range(session["total_chunks"]):
                chunk_path = session["chunks"].get(i)
                if not chunk_path or not os.path.exists(chunk_path):
                    raise HTTPException(status_code=400, detail=f"Missing chunk {i}")
                with open(chunk_path, 'rb') as chunk_file:
                    combined_zip.write(chunk_file.read())
        
        images_processed = 0
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        with zipfile.ZipFile(combined_zip_path, 'r') as zip_ref:
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
            progress.updated_at = datetime.utcnow()
        else:
            progress = TaskProgress(agent_id=agent_id, current_index=0)
            db.add(progress)
        
        db.commit()
        
        try:
            shutil.rmtree(session["temp_dir"])
        except Exception as cleanup_error:
            print(f"⚠️ Error cleaning up temp files: {cleanup_error}")
        
        del upload_sessions[upload_id]
        
        return {
            "message": "Chunked upload completed successfully",
            "agent_id": agent_id,
            "images_processed": images_processed,
            "agent_directory": agent_dir
        }
        
    except Exception as e:
        print(f"❌ Error finalizing chunked upload: {e}")
        try:
            if upload_id in upload_sessions:
                session = upload_sessions[upload_id]
                if os.path.exists(session["temp_dir"]):
                    shutil.rmtree(session["temp_dir"])
                del upload_sessions[upload_id]
        except Exception as cleanup_error:
            print(f"⚠️ Error during error cleanup: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"Failed to finalize upload: {str(e)}")

@router.post("/api/admin/upload-bulk")
@limiter.limit("5/minute")
async def upload_bulk_images(
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload bulk images to general directory"""
    try:
        general_dir = "static/task_images/crime_records_wide"
        os.makedirs(general_dir, exist_ok=True)
        
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
                    while os.path.exists(os.path.join(general_dir, safe_filename)):
                        name, ext = os.path.splitext(original_name)
                        safe_filename = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    image_path = os.path.join(general_dir, safe_filename)
                    with open(image_path, 'wb') as dest_file:
                        dest_file.write(source_file.read())
                    
                    images_processed += 1
                    
                    if images_processed >= 10000:
                        break
        
        return {
            "message": "Bulk images uploaded successfully",
            "images_processed": images_processed,
            "directory": general_dir,
            "note": "These images are available to all agents who don't have specific task assignments"
        }
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        print(f"❌ Bulk upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")

# ===================== DATA EXPORT AND PREVIEW =====================

@router.get("/api/admin/export-excel")
@limiter.limit("5/minute")
async def export_to_excel(
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
        
        df = pd.DataFrame(excel_data)
        
        output = io.BytesIO()
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
        
    except Exception as e:
        print(f"❌ Excel export error: {e}")
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

@router.get("/api/admin/preview-data")
@limiter.limit("10/minute")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Preview submitted data with pagination"""
    try:
        query = db.query(SubmittedForm)
        
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        
        if date_from:
            try:
                from_date = datetime.strptime(date_from, '%Y-%m-%d')
                query = query.filter(SubmittedForm.submitted_at >= from_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                to_date = datetime.strptime(date_to, '%Y-%m-%d')
                to_date = to_date.replace(hour=23, minute=59, second=59)
                query = query.filter(SubmittedForm.submitted_at <= to_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        total_count = query.count()
        submissions = query.order_by(SubmittedForm.submitted_at.desc()).limit(min(limit, 1000)).all()
        
        result = []
        for submission in submissions:
            try:
                form_data = json.loads(submission.form_data)
                
                agent = db.query(Agent).filter(Agent.agent_id == submission.agent_id).first()
                agent_name = agent.name if agent else "Unknown"
                
                result.append({
                    "id": submission.id,
                    "agent_id": submission.agent_id,
                    "agent_name": agent_name,
                    "submitted_at": submission.submitted_at.isoformat(),
                    "image_name": form_data.get('image_name', 'Unknown'),
                    "form_data": form_data,
                    "data_preview": {
                        "DR_NO": form_data.get('DR_NO', ''),
                        "DATE_OCC": form_data.get('DATE_OCC', ''),
                        "AREA_NAME": form_data.get('AREA_NAME', ''),
                        "Crm_Cd_Desc": form_data.get('Crm_Cd_Desc', ''),
                        "LOCATION": form_data.get('LOCATION', '')
                    }
                })
            except Exception as parse_error:
                print(f"Error parsing submission {submission.id}: {parse_error}")
                continue
        
        return {
            "success": True,
            "data": result,
            "total_count": total_count,
            "returned_count": len(result),
            "filters_applied": {
                "agent_id": agent_id,
                "date_from": date_from,
                "date_to": date_to,
                "limit": limit
            }
        }
        
    except Exception as e:
        print(f"❌ Error in data preview: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@router.get("/api/admin/test-data")
@limiter.limit("10/minute")
async def test_data_availability(db: Session = Depends(get_db)):
    """Test data availability and system health"""
    try:
        agent_count = db.query(Agent).count()
        submission_count = db.query(SubmittedForm).count()
        session_count = db.query(AgentSession).count()
        progress_count = db.query(TaskProgress).count()
        
        recent_submissions = db.query(SubmittedForm).filter(
            SubmittedForm.submitted_at >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        active_sessions = db.query(AgentSession).filter(
            AgentSession.logout_time.is_(None)
        ).count()
        
        return {
            "success": True,
            "message": f"Data available - Agents: {agent_count}, Submissions: {submission_count}, Sessions: {session_count}",
            "counts": {
                "agents": agent_count,
                "submissions": submission_count,
                "sessions": session_count,
                "task_progress": progress_count,
                "recent_submissions_24h": recent_submissions,
                "active_sessions": active_sessions
            },
            "system_health": {
                "database_connected": True,
                "timestamp": datetime.utcnow().isoformat(),
                "tables_accessible": True
            }
        }
        
    except Exception as e:
        print(f"❌ Error testing data: {e}")
        return {
            "success": False,
            "error": str(e),
            "system_health": {
                "database_connected": False,
                "timestamp": datetime.utcnow().isoformat(),
                "tables_accessible": False
            }
        }

# ===================== DASHBOARD AND MAINTENANCE ENDPOINTS =====================

@router.get("/api/admin/dashboard-summary")
@limiter.limit("10/minute")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get comprehensive dashboard summary"""
    try:
        total_agents = db.query(Agent).count()
        active_agents = db.query(Agent).filter(Agent.status == "active").count()
        pending_agents = db.query(Agent).filter(Agent.status == "pending").count()
        total_submissions = db.query(SubmittedForm).count()
        
        last_24h = datetime.utcnow() - timedelta(hours=24)
        recent_submissions = db.query(SubmittedForm).filter(
            SubmittedForm.submitted_at >= last_24h
        ).count()
        
        recent_logins = db.query(AgentSession).filter(
            AgentSession.login_time >= last_24h
        ).count()
        
        active_sessions = db.query(AgentSession).filter(
            AgentSession.logout_time.is_(None)
        ).count()
        
        total_tasks = 0
        for agent in db.query(Agent).all():
            agent_images = get_agent_image_files(agent.agent_id)
            total_tasks += len(agent_images)
        
        last_week = datetime.utcnow() - timedelta(days=7)
        top_performers_query = db.query(
            SubmittedForm.agent_id,
            db.func.count(SubmittedForm.id).label('submission_count')
        ).filter(
            SubmittedForm.submitted_at >= last_week
        ).group_by(SubmittedForm.agent_id).order_by(
            db.func.count(SubmittedForm.id).desc()
        ).limit(5).all()
        
        top_performers = []
        for agent_id, count in top_performers_query:
            agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
            if agent:
                top_performers.append({
                    "agent_id": agent_id,
                    "name": agent.name,
                    "submissions_last_week": count
                })
        
        recent_submissions_details = db.query(SubmittedForm).order_by(
            SubmittedForm.submitted_at.desc()
        ).limit(10).all()
        
        recent_activity = []
        for sub in recent_submissions_details:
            agent = db.query(Agent).filter(Agent.agent_id == sub.agent_id).first()
            recent_activity.append({
                "agent_id": sub.agent_id,
                "agent_name": agent.name if agent else "Unknown",
                "submitted_at": sub.submitted_at.isoformat(),
                "image_name": json.loads(sub.form_data).get('image_name', 'Unknown') if sub.form_data else 'Unknown'
            })
        
        return {
            "success": True,
            "counts": {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "pending_agents": pending_agents,
                "total_submissions": total_submissions,
                "total_tasks": total_tasks,
                "pending_tasks": max(0, total_tasks - total_submissions)
            },
            "recent_activity": {
                "submissions_24h": recent_submissions,
                "logins_24h": recent_logins,
                "active_sessions": active_sessions
            },
            "performance": {
                "completion_rate": round((total_submissions / total_tasks * 100), 2) if total_tasks > 0 else 0,
                "avg_submissions_per_agent": round(total_submissions / active_agents, 2) if active_agents > 0 else 0
            },
            "top_performers": top_performers,
            "recent_submissions": recent_activity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error generating dashboard summary: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/api/admin/maintenance")
@limiter.limit("5/minute")
async def perform_maintenance(
    request: Request,
    action: str = Form(...),
    db: Session = Depends(get_db)
):
    """Perform maintenance actions"""
    try:
        if action == "cleanup_old_sessions":
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            old_sessions = db.query(AgentSession).filter(
                AgentSession.login_time < cutoff_time,
                AgentSession.logout_time.is_(None)
            ).all()
            
            for session in old_sessions:
                session.logout_time = session.login_time + timedelta(hours=8)
                session.duration_minutes = 480
            
            db.commit()
            
            return {
                "success": True,
                "action": action,
                "message": f"Cleaned up {len(old_sessions)} old sessions"
            }
            
        elif action == "update_progress":
            agents = db.query(Agent).all()
            updated_count = 0
            
            for agent in agents:
                completed_count = db.query(SubmittedForm).filter(
                    SubmittedForm.agent_id == agent.agent_id
                ).count()
                
                progress = db.query(TaskProgress).filter(
                    TaskProgress.agent_id == agent.agent_id
                ).first()
                
                if progress:
                    progress.current_index = completed_count
                    progress.updated_at = datetime.utcnow()
                    updated_count += 1
                else:
                    new_progress = TaskProgress(
                        agent_id=agent.agent_id,
                        current_index=completed_count
                    )
                    db.add(new_progress)
                    updated_count += 1
            
            db.commit()
            
            return {
                "success": True,
                "action": action,
                "message": f"Updated progress for {updated_count} agents"
            }
            
        elif action == "check_data_integrity":
            issues = []
            
            agents_without_progress = db.query(Agent).outerjoin(TaskProgress).filter(
                TaskProgress.agent_id.is_(None)
            ).count()
            
            if agents_without_progress > 0:
                issues.append(f"{agents_without_progress} agents without progress records")
            
            orphaned_submissions = db.query(SubmittedForm).outerjoin(Agent).filter(
                Agent.agent_id.is_(None)
            ).count()
            
            if orphaned_submissions > 0:
                issues.append(f"{orphaned_submissions} submissions without valid agents")
            
            return {
                "success": True,
                "action": action,
                "issues_found": len(issues),
                "issues": issues,
                "data_integrity": "Good" if len(issues) == 0 else "Issues Found"
            }
            
        else:
            raise HTTPException(status_code=400, detail="Invalid maintenance action")
            
    except Exception as e:
        print(f"❌ Maintenance error: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Maintenance failed: {str(e)}")

# ===================== HEALTH CHECK ENDPOINT =====================

@router.get("/api/health")
@limiter.limit("100/minute")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "Agent Task Management System",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
