import os
import sys
import uuid
import shutil
import zipfile
import asyncio
import aiofiles
import secrets
import string
import hashlib
import tempfile
import json
import re
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager
from io import BytesIO
import logging

# FastAPI and related imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Additional imports
import pandas as pd
from PIL import Image
from jose import jwt, JWTError
import bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration - Railway compatible
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    logger.info("✅ Using PostgreSQL database")
else:
    # Fallback to SQLite for local development
    DATABASE_URL = "sqlite:///./crime_records.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    logger.info("✅ Using SQLite database (local)")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# Database Models
class Admin(Base):
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    mobile = Column(String)
    hashed_password = Column(String)
    dob = Column(String, nullable=True)
    country = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_currently_logged_in = Column(Boolean, default=False)
    tasks_completed = Column(Integer, default=0)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)

class TaskProgress(Base):
    __tablename__ = "task_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    agent_id = Column(String)
    image_path = Column(String)
    image_name = Column(String)
    status = Column(String, default="pending")  # pending, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class SubmittedForm(Base):
    __tablename__ = "submitted_forms"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String)
    task_id = Column(String)
    image_name = Column(String)
    
    # Crime Report Fields
    crime_type = Column(String, nullable=True)
    location = Column(String, nullable=True)
    date_time = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    suspect_info = Column(Text, nullable=True)
    witness_info = Column(Text, nullable=True)
    evidence_details = Column(Text, nullable=True)
    priority_level = Column(String, nullable=True)
    
    # Additional Fields
    reporter_name = Column(String, nullable=True)
    reporter_contact = Column(String, nullable=True)
    case_number = Column(String, nullable=True)
    investigating_officer = Column(String, nullable=True)
    
    submitted_at = Column(DateTime, default=datetime.utcnow)

class AgentSession(Base):
    __tablename__ = "agent_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String)
    session_token = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

class ChunkedUpload(Base):
    __tablename__ = "chunked_uploads"
    
    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(String, unique=True, index=True)
    filename = Column(String)
    filesize = Column(Integer)
    total_chunks = Column(Integer)
    chunks_received = Column(Integer, default=0)
    agent_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed = Column(Boolean, default=False)

# Global variables
database_ready = False
upload_sessions: Dict[str, Dict[str, Any]] = {}

# Directories
UPLOAD_DIR = Path("uploads")
TASKS_DIR = Path("static/task_images")
TEMP_DIR = Path("temp")
CHUNK_UPLOAD_DIR = Path("temp_chunks")

# Create directories
for directory in [UPLOAD_DIR, TASKS_DIR, TEMP_DIR, CHUNK_UPLOAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Security functions
def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash"""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def generate_password(length: int = 8) -> str:
    """Generate secure random password"""
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

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

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Database setup
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create database tables"""
    try:
        logger.info("🔧 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        global database_ready
        database_ready = True
        return True
    except Exception as e:
        logger.error(f"❌ Database table creation failed: {e}")
        return False

def create_default_admin():
    """Create default admin user"""
    try:
        if not database_ready:
            return False
            
        db = SessionLocal()
        try:
            # Check if admin exists
            existing_admin = db.query(Admin).filter(Admin.username == "admin").first()
            
            if not existing_admin:
                # Create new admin
                admin = Admin(
                    username="admin",
                    hashed_password=hash_password("admin123"),
                    email="admin@agent-task-system.com",
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.add(admin)
                db.commit()
                logger.info("✅ Default admin created: admin/admin123")
            else:
                # Update existing admin
                existing_admin.hashed_password = hash_password("admin123")
                existing_admin.is_active = True
                db.commit()
                logger.info("✅ Admin credentials updated: admin/admin123")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Admin creation error: {e}")
            db.rollback()
            return False
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"❌ Admin setup failed: {e}")
        return False

# Enhanced CORS Origins
ALLOWED_ORIGINS = [
    "https://agent-task-system.com",
    "https://www.agent-task-system.com", 
    "https://web-railwaybuilderherokupython.up.railway.app",
    "https://web-production-b3ef2.up.railway.app",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173"
]

# Add environment-specific origins
if os.environ.get("RAILWAY_STATIC_URL"):
    ALLOWED_ORIGINS.append(f"https://{os.environ.get('RAILWAY_STATIC_URL')}")
if os.environ.get("RAILWAY_PUBLIC_DOMAIN"):
    ALLOWED_ORIGINS.append(f"https://{os.environ.get('RAILWAY_PUBLIC_DOMAIN')}")

# Security
security = HTTPBearer()

async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    admin = db.query(Admin).filter(Admin.username == username).first()
    if admin is None or not admin.is_active:
        raise credentials_exception
    
    return admin

# Cleanup function
async def periodic_cleanup():
    """Clean up old upload sessions and temporary files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            now = datetime.utcnow()
            expired_sessions = []
            
            # Clean up upload sessions
            for upload_id, session in upload_sessions.items():
                if (now - session.get("created_at", now)).total_seconds() > 7200:  # 2 hours
                    expired_sessions.append(upload_id)
            
            for upload_id in expired_sessions:
                if upload_id in upload_sessions:
                    del upload_sessions[upload_id]
                    logger.info(f"🧹 Cleaned expired upload session: {upload_id}")
            
            # Clean up temporary files
            if TEMP_DIR.exists():
                for temp_file in TEMP_DIR.glob("*"):
                    if temp_file.is_file():
                        file_age = (now.timestamp() - temp_file.stat().st_mtime)
                        if file_age > 7200:  # 2 hours
                            temp_file.unlink()
                            logger.info(f"🧹 Cleaned temp file: {temp_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown"""
    logger.info("🚀 Starting Client Records System...")
    
    # Initialize database
    tables_created = create_tables()
    if tables_created:
        create_default_admin()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# FastAPI App
app = FastAPI(
    title="Client Records Data Entry System",
    version="2.0.0", 
    description="Enhanced system for crime records management with Railway deployment",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# Security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted")
except Exception as e:
    logger.warning(f"⚠️ Static files mount failed: {e}")

# Import agent routes
try:
    from agent_routes import router as agent_router
    app.include_router(agent_router, prefix="/api")
    logger.info("✅ Agent routes imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import agent routes: {e}")
    # Create basic agent routes if import fails
    @app.post("/api/agents/register")
    async def register_agent_fallback():
        return {"error": "Agent routes not properly configured"}

# Root routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Client Records Data Entry System API v2.0",
        "status": "operational",
        "database": "ready" if database_ready else "initializing",
        "health": "/health",
        "admin": "/admin.html",
        "agent": "/agent.html"
    }

@app.get("/health")
async def health_check():
    """Health check for Railway"""
    return {
        "status": "healthy",
        "database": database_ready,
        "timestamp": datetime.utcnow().isoformat()
    }

# Admin Authentication Routes
@app.post("/api/admin/login")
async def admin_login(credentials: dict, db: Session = Depends(get_db)):
    """Admin login"""
    try:
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        admin = db.query(Admin).filter(Admin.username == username).first()
        
        if admin and admin.is_active and verify_password(password, admin.hashed_password):
            admin.last_login = datetime.utcnow()
            db.commit()
            
            access_token = create_access_token(data={"sub": username})
            
            return {
                "success": True,
                "message": "Login successful",
                "access_token": access_token,
                "token_type": "bearer",
                "username": username
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Admin login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/admin/simple-login")
async def simple_admin_login(credentials: dict):
    """Simple admin login for testing"""
    username = credentials.get("username")
    password = credentials.get("password")
    
    if username == "admin" and password == "admin123":
        return {"success": True, "message": "Login successful"}
    
    return {"success": False, "message": "Invalid credentials"}

@app.post("/api/admin/test-login")
async def test_admin_login(credentials: dict, db: Session = Depends(get_db)):
    """Test admin login endpoint"""
    return await admin_login(credentials, db)

@app.get("/api/admin/check-admin")
async def check_admin_status(db: Session = Depends(get_db)):
    """Check admin status"""
    try:
        admin_count = db.query(Admin).count()
        admin = db.query(Admin).filter(Admin.username == "admin").first()
        
        return {
            "admin_exists": admin is not None,
            "admin_count": admin_count,
            "admin_active": admin.is_active if admin else False,
            "last_login": admin.last_login.isoformat() if admin and admin.last_login else None,
            "database_ready": database_ready
        }
    except Exception as e:
        return {"error": str(e), "database_ready": database_ready}

@app.post("/api/admin/create-admin")
async def create_admin_endpoint(db: Session = Depends(get_db)):
    """Create admin user endpoint"""
    success = create_default_admin()
    return {
        "success": success,
        "message": "Admin user setup completed" if success else "Admin setup failed"
    }

# System Status Routes
@app.get("/api/admin/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        total_agents = db.query(Agent).count()
        total_tasks = db.query(TaskProgress).count()
        completed_tasks = db.query(TaskProgress).filter(TaskProgress.status == "completed").count()
        pending_tasks = total_tasks - completed_tasks
        
        return {
            "total_agents": total_agents,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks
        }
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return {
            "total_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0
        }

# Task Management Routes
@app.post("/api/admin/upload-tasks")
async def upload_tasks(
    zip_file: UploadFile = File(...),
    agent_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload ZIP file and create tasks"""
    try:
        # Validate agent exists and is active
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        if agent.status != "active":
            raise HTTPException(status_code=400, detail="Agent is not active")
        
        # Validate file type
        if not zip_file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
        
        # Create agent directory
        agent_dir = TASKS_DIR / agent_id
        agent_dir.mkdir(exist_ok=True)
        
        # Save ZIP file temporarily
        zip_path = TEMP_DIR / f"{uuid.uuid4()}.zip"
        
        with open(zip_path, "wb") as buffer:
            content = await zip_file.read()
            buffer.write(content)
        
        images_processed = 0
        supported_extensions = ('.jpg', '.jpeg', '.png', '.gif')
        
        # Extract and process images
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.lower().endswith(supported_extensions):
                    try:
                        # Extract image
                        extracted_path = zip_ref.extract(file_info, TEMP_DIR)
                        
                        # Validate image file
                        try:
                            with Image.open(extracted_path) as img:
                                img.verify()  # Verify it's a valid image
                        except (IOError, SyntaxError):
                            logger.warning(f"Skipping invalid image: {file_info.filename}")
                            continue
                        
                        # Move to agent directory
                        image_name = os.path.basename(file_info.filename)
                        final_path = agent_dir / image_name
                        
                        # Ensure unique filename
                        counter = 1
                        original_name = final_path.stem
                        original_ext = final_path.suffix
                        while final_path.exists():
                            final_path = agent_dir / f"{original_name}_{counter}{original_ext}"
                            counter += 1
                        
                        shutil.move(extracted_path, final_path)
                        
                        # Create task
                        task_id = f"TASK_{agent_id}_{uuid.uuid4().hex[:8].upper()}"
                        task = TaskProgress(
                            task_id=task_id,
                            agent_id=agent_id,
                            image_path=str(final_path),
                            image_name=final_path.name,
                            status="pending"
                        )
                        db.add(task)
                        images_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing image {file_info.filename}: {e}")
                        continue
        
        # Clean up
        try:
            os.remove(zip_path)
        except:
            pass
        
        # Remove extracted directories in temp
        for item in TEMP_DIR.iterdir():
            if item.is_dir() and str(item.name).startswith('tmp'):
                try:
                    shutil.rmtree(item, ignore_errors=True)
                except:
                    pass
        
        db.commit()
        
        logger.info(f"✅ Processed {images_processed} images for agent {agent_id}")
        
        return {
            "success": True,
            "images_processed": images_processed,
            "agent_id": agent_id,
            "message": f"Successfully assigned {images_processed} tasks to agent"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Agent Task Routes
@app.get("/api/agents/{agent_id}/tasks")
async def get_agent_tasks(
    agent_id: str,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get tasks for agent with proper validation"""
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
            "total": len(tasks),
            "agent_id": agent_id,
            "agent_name": agent.name
        }
        
    except Exception as e:
        logger.error(f"Tasks fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tasks")

# Form Submission Routes
@app.post("/api/agents/{agent_id}/tasks/{task_id}/submit")
async def submit_task_form(
    agent_id: str,
    task_id: str,
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
    """Submit form data for task"""
    try:
        # Verify task
        task = db.query(TaskProgress).filter(
            TaskProgress.task_id == task_id,
            TaskProgress.agent_id == agent_id
        ).first()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Check if already submitted
        existing_submission = db.query(SubmittedForm).filter(
            SubmittedForm.task_id == task_id,
            SubmittedForm.agent_id == agent_id
        ).first()
        
        if existing_submission:
            # Update existing
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
            
            # Update task and agent
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            
            agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
            if agent:
                agent.tasks_completed += 1
            
            message = "Form submitted successfully"
        
        db.commit()
        
        return {
            "success": True,
            "message": message,
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Form submission error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to submit form")

# Data Export Routes
@app.get("/api/admin/export-excel")
async def export_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export data to Excel"""
    try:
        query = db.query(SubmittedForm)
        
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        
        if date_from:
            date_from_obj = datetime.fromisoformat(date_from)
            query = query.filter(SubmittedForm.submitted_at >= date_from_obj)
        
        if date_to:
            date_to_obj = datetime.fromisoformat(date_to + "T23:59:59")
            query = query.filter(SubmittedForm.submitted_at <= date_to_obj)
        
        submissions = query.all()
        
        if not submissions:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Create DataFrame
        data = []
        for sub in submissions:
            data.append({
                "Agent ID": sub.agent_id,
                "Task ID": sub.task_id,
                "Image Name": sub.image_name,
                "Crime Type": sub.crime_type,
                "Location": sub.location,
                "Date Time": sub.date_time,
                "Description": sub.description,
                "Suspect Info": sub.suspect_info,
                "Witness Info": sub.witness_info,
                "Evidence Details": sub.evidence_details,
                "Priority Level": sub.priority_level,
                "Reporter Name": sub.reporter_name,
                "Reporter Contact": sub.reporter_contact,
                "Case Number": sub.case_number,
                "Investigating Officer": sub.investigating_officer,
                "Submitted At": sub.submitted_at.isoformat() if sub.submitted_at else None
            })
        
        df = pd.DataFrame(data)
        
        # Create Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Crime Records', index=False)
        
        output.seek(0)
        
        return StreamingResponse(
            BytesIO(output.getvalue()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=crime_records_export.xlsx"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Excel export error: {e}")
        raise HTTPException(status_code=500, detail="Export failed")

# Static file serving
@app.get("/admin.html")
async def serve_admin():
    """Serve admin panel"""
    if os.path.exists("admin.html"):
        return FileResponse("admin.html")
    return JSONResponse({"error": "Admin panel not found"}, status_code=404)

@app.get("/agent.html")
async def serve_agent():
    """Serve agent panel"""
    if os.path.exists("agent.html"):
        return FileResponse("agent.html")
    return JSONResponse({"error": "Agent panel not found"}, status_code=404)

# Debug and testing routes
@app.get("/debug")
async def debug_info(db: Session = Depends(get_db)):
    """Debug information"""
    try:
        agent_count = db.query(Agent).count()
        task_count = db.query(TaskProgress).count()
        form_count = db.query(SubmittedForm).count()
        
        return {
            "system": {
                "database_ready": database_ready,
                "python_version": sys.version,
                "files": os.listdir("."),
                "directories": {
                    "uploads": UPLOAD_DIR.exists(),
                    "tasks": TASKS_DIR.exists(),
                    "temp": TEMP_DIR.exists(),
                    "chunks": CHUNK_UPLOAD_DIR.exists()
                }
            },
            "database": {
                "agents": agent_count,
                "tasks": task_count,
                "forms": form_count
            },
            "upload_sessions": len(upload_sessions),
            "cors_origins": len(ALLOWED_ORIGINS)
        }
    except Exception as e:
        return {"error": str(e), "database_ready": database_ready}

# Additional admin routes
@app.get("/api/admin/agent-password/{agent_id}")
async def get_agent_password_info(agent_id: str, db: Session = Depends(get_db)):
    """Get agent password information"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "message": f"Password information for {agent.name} ({agent_id}). Contact admin for password reset.",
        "agent_id": agent_id,
        "name": agent.name,
        "email": agent.email
    }

@app.post("/api/admin/reset-password/{agent_id}")
async def reset_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Reset agent password"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Generate new password
    new_password = generate_secure_password()
    agent.hashed_password = hash_password(new_password)
    agent.login_attempts = 0
    agent.locked_until = None
    
    db.commit()
    
    return {
        "success": True,
        "agent_id": agent_id,
        "new_password": new_password,
        "message": f"Password reset for {agent.name}"
    }

@app.patch("/api/agents/{agent_id}/status")
async def update_agent_status(
    agent_id: str,
    status_data: dict,
    db: Session = Depends(get_db)
):
    """Update agent status"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    new_status = status_data.get("status")
    if new_status not in ["active", "inactive"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    agent.status = new_status
    db.commit()
    
    return {"success": True, "message": f"Agent status updated to {new_status}"}

# Preview data route
@app.get("/api/admin/preview-data")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Preview data for export"""
    try:
        query = db.query(SubmittedForm)
        
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        
        if date_from:
            date_from_obj = datetime.fromisoformat(date_from)
            query = query.filter(SubmittedForm.submitted_at >= date_from_obj)
        
        if date_to:
            date_to_obj = datetime.fromisoformat(date_to + "T23:59:59")
            query = query.filter(SubmittedForm.submitted_at <= date_to_obj)
        
        submissions = query.limit(50).all()
        
        data = []
        for sub in submissions:
            data.append({
                "agent_id": sub.agent_id,
                "task_id": sub.task_id,
                "image_name": sub.image_name,
                "crime_type": sub.crime_type,
                "location": sub.location,
                "date_time": sub.date_time,
                "description": sub.description,
                "suspect_info": sub.suspect_info,
                "witness_info": sub.witness_info,
                "evidence_details": sub.evidence_details,
                "priority_level": sub.priority_level,
                "reporter_name": sub.reporter_name,
                "reporter_contact": sub.reporter_contact,
                "case_number": sub.case_number,
                "investigating_officer": sub.investigating_officer,
                "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None
            })
        
        return {"data": data, "total_count": query.count()}
        
    except Exception as e:
        logger.error(f"Admin data preview error: {e}")
        raise HTTPException(status_code=500, detail="Preview failed")

@app.get("/api/admin/test-data")
async def test_data_availability(db: Session = Depends(get_db)):
    """Test data availability"""
    try:
        submission_count = db.query(SubmittedForm).count()
        agent_count = db.query(Agent).count()
        task_count = db.query(TaskProgress).count()
        completed_task_count = db.query(TaskProgress).filter(TaskProgress.status == "completed").count()
        
        return {
            "success": True,
            "message": f"Data available: {submission_count} submissions, {agent_count} agents, {task_count} tasks",
            "submission_count": submission_count,
            "agent_count": agent_count,
            "task_count": task_count,
            "completed_task_count": completed_task_count
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Chunked upload routes for large files
@app.post("/api/admin/init-chunked-upload")
async def init_chunked_upload(
    filename: str = Form(...),
    filesize: int = Form(...),
    total_chunks: int = Form(...),
    agent_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Initialize chunked upload"""
    try:
        upload_id = str(uuid.uuid4())
        
        chunked_upload = ChunkedUpload(
            upload_id=upload_id,
            filename=filename,
            filesize=filesize,
            total_chunks=total_chunks,
            agent_id=agent_id
        )
        
        db.add(chunked_upload)
        db.commit()
        
        # Create directory for chunks
        chunk_dir = CHUNK_UPLOAD_DIR / upload_id
        chunk_dir.mkdir(exist_ok=True)
        
        return {"upload_id": upload_id}
        
    except Exception as e:
        logger.error(f"Chunked upload init error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize upload")

@app.post("/api/admin/upload-chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload individual chunk"""
    try:
        chunked_upload = db.query(ChunkedUpload).filter(
            ChunkedUpload.upload_id == upload_id
        ).first()
        
        if not chunked_upload:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        # Save chunk
        chunk_dir = CHUNK_UPLOAD_DIR / upload_id
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}"
        
        async with aiofiles.open(chunk_path, 'wb') as buffer:
            content = await chunk.read()
            await buffer.write(content)
        
        # Update progress
        chunked_upload.chunks_received += 1
        db.commit()
        
        return {"success": True, "chunk_index": chunk_index}
        
    except Exception as e:
        logger.error(f"Chunk upload error: {e}")
        raise HTTPException(status_code=500, detail="Chunk upload failed")

@app.post("/api/admin/finalize-chunked-upload")
async def finalize_chunked_upload(
    upload_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Finalize chunked upload and process file"""
    try:
        chunked_upload = db.query(ChunkedUpload).filter(
            ChunkedUpload.upload_id == upload_id
        ).first()
        
        if not chunked_upload:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        # Reassemble file
        chunk_dir = CHUNK_UPLOAD_DIR / upload_id
        final_path = TEMP_DIR / chunked_upload.filename
        
        with open(final_path, "wb") as outfile:
            for i in range(chunked_upload.total_chunks):
                chunk_path = chunk_dir / f"chunk_{i:04d}"
                if chunk_path.exists():
                    with open(chunk_path, "rb") as chunk_file:
                        outfile.write(chunk_file.read())
        
        # Process as normal ZIP file
        agent = db.query(Agent).filter(Agent.agent_id == chunked_upload.agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent_dir = TASKS_DIR / chunked_upload.agent_id
        agent_dir.mkdir(exist_ok=True)
        
        images_processed = 0
        
        # Extract and process images
        with zipfile.ZipFile(final_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    extracted_path = zip_ref.extract(file_info, TEMP_DIR)
                    
                    image_name = os.path.basename(file_info.filename)
                    final_image_path = agent_dir / image_name
                    shutil.move(extracted_path, final_image_path)
                    
                    task_id = f"TASK_{chunked_upload.agent_id}_{uuid.uuid4().hex[:8].upper()}"
                    task = TaskProgress(
                        task_id=task_id,
                        agent_id=chunked_upload.agent_id,
                        image_path=str(final_image_path),
                        image_name=image_name,
                        status="pending"
                    )
                    db.add(task)
                    images_processed += 1
        
        # Mark upload as completed
        chunked_upload.completed = True
        db.commit()
        
        # Clean up
        shutil.rmtree(chunk_dir, ignore_errors=True)
        if final_path.exists():
            os.remove(final_path)
        
        logger.info(f"✅ Chunked upload completed: {images_processed} images for {chunked_upload.agent_id}")
        
        return {
            "success": True,
            "images_processed": images_processed,
            "agent_id": chunked_upload.agent_id
        }
        
    except Exception as e:
        logger.error(f"Finalize chunked upload error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to finalize upload")

# Agent dashboard and profile routes
@app.get("/api/agents/{agent_id}/dashboard")
async def get_agent_dashboard(agent_id: str, db: Session = Depends(get_db)):
    """Get agent dashboard data"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == "completed"
        ).count()
        
        recent_submissions = db.query(SubmittedForm).filter(
            SubmittedForm.agent_id == agent_id
        ).order_by(SubmittedForm.submitted_at.desc()).limit(5).all()
        
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
                "pending_tasks": total_tasks - completed_tasks
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dashboard")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    logger.error(f"Internal error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 60)
    print("🚀 CLIENT RECORDS DATA ENTRY SYSTEM v2.0")
    print("=" * 60)
    print(f"🌍 Environment: {'Production' if DATABASE_URL and DATABASE_URL.startswith('postgresql') else 'Development'}")
    print(f"🔗 CORS Origins: {len(ALLOWED_ORIGINS)} configured")
    print(f"💾 Database: {'PostgreSQL' if DATABASE_URL and DATABASE_URL.startswith('postgresql') else 'SQLite'}")
    print(f"📊 Database Ready: {database_ready}")
    print(f"🏃 Starting server on port {port}")
    print("=" * 60)
    print("🔐 DEFAULT CREDENTIALS:")
    print("Admin - Username: admin, Password: admin123")
    print("=" * 60)
    print("📱 ACCESS POINTS:")
    print(f"- Admin Panel: http://localhost:{port}/admin.html")
    print(f"- Agent Panel: http://localhost:{port}/agent.html")
    print(f"- Health Check: http://localhost:{port}/health")
    print(f"- API Documentation: http://localhost:{port}/docs")
    print(f"- Debug Info: http://localhost:{port}/debug")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
