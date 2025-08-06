import os
import sys
import uuid
import shutil
import zipfile
import asyncio
import aiofiles
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import logging
import secrets
import string
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
import pandas as pd
import io
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Console only for Railway
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Client Records Data Entry System",
    version="2.0.1",
    description="Enhanced system for agent-task-system.com with chunked upload support"
)

# CORS configuration
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://agent-task-system.com,https://www.agent-task-system.com").split(",")
if os.environ.get("DOMAIN") == "web-railwaybuilderherokupython.up.railway.app":
    ALLOWED_ORIGINS.append("https://web-railwaybuilderherokupython.up.railway.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Session-Token", "X-Requested-With", "Accept", "Origin"],
    expose_headers=["Content-Disposition"]
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/login")

# File system configuration
CHUNK_UPLOAD_DIR = "/app/temp_chunks"
STATIC_DIR = "/app/static"
STATIC_TASKS_DIR = "/app/static/task_images"
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(STATIC_TASKS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global state
database_ready = False
routes_ready = False

# Mock database for fallback
class MockDB:
    def query(self, *args, **kwargs):
        logger.warning("Using MockDB: Database not available")
        return MockQuery()
    
    def add(self, *args, **kwargs):
        pass
    
    def commit(self):
        pass
    
    def rollback(self):
        pass
    
    def close(self):
        pass

class MockQuery:
    def filter(self, *args, **kwargs):
        return self
    
    def order_by(self, *args, **kwargs):
        return self
    
    def first(self):
        return None
    
    def all(self):
        return []
    
    def count(self):
        return 0
    
    def limit(self, *args):
        return self
    
    def join(self, *args):
        return self

def get_mock_db():
    return MockDB()

# Database setup
try:
    logger.info("📦 Importing database modules...")
    from database import Base, engine, get_db
    from models import Agent, TaskProgress, SubmittedForm, AgentSession, UploadSession
    
    logger.info("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created successfully!")
    database_ready = True
    db_dependency = get_db
except Exception as e:
    logger.error(f"❌ Database setup failed: {e}", exc_info=True)
    database_ready = False
    db_dependency = get_mock_db

# Agent routes
try:
    logger.info("📦 Importing agent routes...")
    from agent_routes import router as agent_router
    app.include_router(agent_router)
    logger.info("✅ Agent routes included successfully!")
    routes_ready = True
except Exception as e:
    logger.error(f"❌ Agent routes failed: {e}", exc_info=True)
    routes_ready = False

# JWT authentication
def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_admin(db: Session = Depends(db_dependency), token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        agent_id = payload.get("sub")
        if not agent_id:
            raise HTTPException(status_code=401, detail="Invalid token: No agent_id")
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent or agent.status != "active":
            raise HTTPException(status_code=403, detail="Invalid or inactive admin")
        if not agent.is_admin:  # Assuming Agent model has is_admin
            raise HTTPException(status_code=403, detail="Admin access required")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/admin/login")
async def admin_login(email: str = Form(...), password: str = Form(...), db: Session = Depends(db_dependency)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    try:
        agent = db.query(Agent).filter(Agent.email == email).first()
        if not agent or not pwd_context.verify(password, agent.hashed_password) or not agent.is_admin:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if agent.status != "active":
            raise HTTPException(status_code=403, detail="Account is not active")
        token = create_access_token({"sub": agent.agent_id, "role": "admin"})
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Admin login failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed")

# Middleware for security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# Health endpoint
@app.get("/health")
async def health_check(db: Session = Depends(db_dependency)):
    health_status = {
        "status": "healthy",
        "platform": "Railway",
        "message": "Service is running",
        "timestamp": datetime.utcnow().isoformat(),
        "domain": os.environ.get("DOMAIN", "agent-task-system.com"),
        "database": "unknown",
        "imports_loaded": database_ready,
        "chunked_upload": "enabled",
        "version": "2.0.1"
    }
    
    if database_ready:
        try:
            result = db.execute(text("SELECT 1")).scalar()
            health_status["database"] = "connected" if result == 1 else "query_failed"
        except Exception as e:
            health_status["database"] = f"query_error: {str(e)[:50]}"
            health_status["status"] = "degraded"
    else:
        health_status["database"] = "not_ready"
        health_status["status"] = "degraded"
    
    health_status["static_storage"] = "ready" if os.path.exists(STATIC_TASKS_DIR) else "missing"
    health_status["upload_storage"] = "ready" if os.path.exists(CHUNK_UPLOAD_DIR) else "missing"
    if database_ready:
        health_status["active_uploads"] = db.query(UploadSession).count()
    else:
        health_status["active_uploads"] = 0
    
    return health_status

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Client Records Data Entry System API v2.0.1",
        "status": "running",
        "platform": "Railway",
        "domain": os.environ.get("DOMAIN", "agent-task-system.com"),
        "health_check": "/health",
        "admin_panel": "/static/admin.html",
        "agent_panel": "/static/agent.html",
        "features": ["chunked_upload", "large_file_support", "custom_domain_support", "ssl_enabled"]
    }

# Static file serving
@app.get("/static/admin.html")
async def serve_admin_panel():
    try:
        file_path = os.path.join(STATIC_DIR, "admin.html")
        if os.path.exists(file_path):
            return FileResponse(file_path, headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            })
        return JSONResponse({"error": "Admin panel not found"}, status_code=404)
    except Exception as e:
        logger.error(f"❌ Error serving admin panel: {e}", exc_info=True)
        return JSONResponse({"error": f"Could not serve admin panel: {str(e)}"}, status_code=500)

@app.get("/static/agent.html")
async def serve_agent_panel():
    try:
        file_path = os.path.join(STATIC_DIR, "agent.html")
        if os.path.exists(file_path):
            return FileResponse(file_path, headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            })
        return JSONResponse({"error": "Agent panel not found"}, status_code=404)
    except Exception as e:
        logger.error(f"❌ Error serving agent panel: {e}", exc_info=True)
        return JSONResponse({"error": f"Could not serve agent panel: {str(e)}"}, status_code=500)

# Debug and status endpoints
@app.get("/debug")
async def debug_info(db: Session = Depends(db_dependency)):
    return {
        "environment": {
            "domain": os.environ.get("DOMAIN", "agent-task-system.com"),
            "port": os.environ.get("PORT", "not_set"),
            "database_url_set": bool(os.environ.get("DATABASE_URL")),
            "allowed_origins": ALLOWED_ORIGINS
        },
        "system": {
            "files": os.listdir(STATIC_DIR) if os.path.exists(STATIC_DIR) else [],
            "python_version": sys.version,
            "database_ready": database_ready,
            "routes_ready": routes_ready
        },
        "features": {
            "upload_sessions": db.query(UploadSession).count() if database_ready else 0,
            "chunk_upload_dir_exists": os.path.exists(CHUNK_UPLOAD_DIR),
            "static_dir_exists": os.path.exists(STATIC_TASKS_DIR)
        }
    }

@app.get("/status")
async def system_status(db: Session = Depends(db_dependency)):
    return {
        "status": "operational",
        "database": "ready" if database_ready else "failed",
        "routes": "ready" if routes_ready else "failed",
        "domain": os.environ.get("DOMAIN", "agent-task-system.com"),
        "health": "ok",
        "chunked_upload": "enabled",
        "active_uploads": db.query(UploadSession).count() if database_ready else 0,
        "cors_origins": len(ALLOWED_ORIGINS)
    }

# Admin statistics
@app.get("/api/admin/statistics")
async def get_admin_statistics(db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        return {
            "total_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0,
            "error": "Database not ready"
        }
    
    try:
        total_agents = db.query(Agent).count()
        total_tasks = db.query(TaskProgress).count()
        completed_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'completed').count()
        pending_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'pending').count()
        in_progress_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'in_progress').count()
        
        return {
            "total_agents": total_agents,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks
        }
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

# Agent endpoints
@app.get("/api/agents")
async def list_agents(db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        return {"agents": [], "error": "Database not ready"}
    
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
            
            agent_data = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "status": agent.status,
                "is_admin": agent.is_admin,
                "tasks_completed": completed_tasks,
                "total_tasks": total_tasks,
                "last_login": latest_session.login_time.isoformat() if latest_session and latest_session.login_time else None,
                "last_logout": latest_session.logout_time.isoformat() if latest_session and latest_session.logout_time else None,
                "is_currently_logged_in": latest_session.logout_time is None if latest_session else False
            }
            agent_list.append(agent_data)
        
        return {"agents": agent_list}
    except Exception as e:
        logger.error(f"❌ Error listing agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@app.get("/api/agents/{agent_id}/tasks/current")
async def get_current_task(agent_id: str, db: Session = Depends(db_dependency)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        next_task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status.in_(['pending', 'in_progress'])
        ).order_by(TaskProgress.assigned_at).first()
        
        if not next_task:
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
        
        if next_task.status == 'pending':
            next_task.status = 'in_progress'
            next_task.started_at = datetime.utcnow()
            db.commit()
        
        total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
        completed_tasks = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'completed'
        ).count()
        current_index = completed_tasks
        
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting current task for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting current task: {str(e)}")

@app.get("/api/agents/{agent_id}/tasks")
async def get_agent_tasks(agent_id: str, db: Session = Depends(db_dependency)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
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
        
        return {"tasks": task_list}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting tasks for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting tasks: {str(e)}")

@app.get("/api/agents/{agent_id}/statistics")
async def get_agent_statistics(agent_id: str, db: Session = Depends(db_dependency)):
    if not database_ready:
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0,
            "error": "Database not ready"
        }
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
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
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting statistics for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

# Agent registration
@app.post("/api/agents/register")
async def register_agent(
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(...),
    country: str = Form(...),
    gender: str = Form(...),
    is_admin: bool = Form(False),
    db: Session = Depends(db_dependency)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        try:
            datetime.strptime(dob, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        existing_agent = db.query(Agent).filter(Agent.email == email).first()
        if existing_agent:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        agent_id = f"AG{datetime.utcnow().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"
        password = ''.join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(12))
        
        new_agent = Agent(
            agent_id=agent_id,
            name=name,
            email=email,
            mobile=mobile,
            dob=dob,
            country=country,
            gender=gender,
            hashed_password=pwd_context.hash(password),
            status="active",
            is_admin=is_admin,
            created_at=datetime.utcnow()
        )
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        logger.info(f"✅ New agent registered: {agent_id} (admin: {is_admin})")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "password": password,
            "message": "Agent registered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error registering agent: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# Form submission
@app.post("/api/agents/{agent_id}/submit")
async def submit_task_form(
    agent_id: str,
    request: Request,
    db: Session = Depends(db_dependency)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            data = await request.json()
        else:
            form_data = await request.form()
            data = {k: v for k, v in form_data.items() if k not in ('agent_id', 'task_id')}
        
        current_task = db.query(TaskProgress).filter(
            TaskProgress.agent_id == agent_id,
            TaskProgress.status == 'in_progress'
        ).order_by(TaskProgress.assigned_at).first()
        
        if not current_task:
            current_task = db.query(TaskProgress).filter(
                TaskProgress.agent_id == agent_id,
                TaskProgress.status == 'pending'
            ).order_by(TaskProgress.assigned_at).first()
        
        if not current_task:
            raise HTTPException(status_code=404, detail="No active task found for submission")
        
        submitted_form = SubmittedForm(
            agent_id=agent_id,
            task_id=current_task.id,
            image_filename=current_task.image_filename,
            form_data=data,
            submitted_at=datetime.utcnow()
        )
        
        db.add(submitted_form)
        current_task.status = 'completed'
        current_task.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"✅ Task {current_task.id} completed by agent {agent_id}")
        
        return {
            "success": True,
            "message": "Task submitted successfully",
            "task_id": current_task.id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error submitting task for {agent_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")

# Standard upload
@app.post("/api/admin/upload-tasks")
async def upload_tasks_standard(
    zip_file: UploadFile = File(...),
    agent_id: str = Form(...),
    db: Session = Depends(db_dependency),
    admin: dict = Depends(get_current_admin)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        if agent.status != "active":
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not active")
        
        temp_file_path = os.path.join(CHUNK_UPLOAD_DIR, f"temp_{uuid.uuid4().hex}_{zip_file.filename}")
        
        async with aiofiles.open(temp_file_path, 'wb') as buffer:
            content = await zip_file.read()
            await buffer.write(content)
        
        result = await process_uploaded_zip(temp_file_path, agent_id, db)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Chunked upload endpoints
@app.post("/api/admin/init-chunked-upload")
async def init_chunked_upload(
    filename: str = Form(...),
    filesize: int = Form(...),
    total_chunks: int = Form(...),
    agent_id: str = Form(...),
    db: Session = Depends(db_dependency),
    admin: dict = Depends(get_current_admin)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        if agent.status != "active":
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not active")
        
        upload_id = str(uuid.uuid4())
        upload_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        session = UploadSession(
            id=upload_id,
            filename=filename,
            filesize=filesize,
            total_chunks=total_chunks,
            agent_id=agent_id,
            created_at=datetime.utcnow(),
            chunks_received=[]  # Empty list for JSON column
        )
        db.add(session)
        db.commit()
        
        logger.info(f"🚀 Initialized chunked upload: {upload_id} for {filename} ({filesize} bytes, {total_chunks} chunks)")
        
        return {
            "upload_id": upload_id,
            "status": "initialized",
            "message": f"Ready to receive {total_chunks} chunks"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to initialize chunked upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize upload: {str(e)}")

@app.post("/api/admin/upload-chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
    db: Session = Depends(db_dependency)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        session = db.query(UploadSession).filter(UploadSession.id == upload_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        if chunk_index >= session.total_chunks or chunk_index < 0:
            raise HTTPException(status_code=400, detail=f"Invalid chunk index: {chunk_index}")
        
        chunks_received = session.chunks_received or []
        if chunk_index in chunks_received:
            return {
                "status": "chunk_already_exists",
                "chunk_index": chunk_index,
                "received_chunks": len(chunks_received),
                "total_chunks": session.total_chunks
            }
        
        chunk_path = os.path.join(CHUNK_UPLOAD_DIR, upload_id, f"chunk_{chunk_index:06d}")
        
        async with aiofiles.open(chunk_path, 'wb') as f:
            content = await chunk.read()
            await f.write(content)
        
        chunks_received.append(chunk_index)
        session.chunks_received = chunks_received
        db.commit()
        
        logger.info(f"📦 Received chunk {chunk_index + 1}/{session.total_chunks} for upload {upload_id}")
        
        return {
            "status": "chunk_uploaded",
            "chunk_index": chunk_index,
            "received_chunks": len(chunks_received),
            "total_chunks": session.total_chunks,
            "progress_percentage": (len(chunks_received) / session.total_chunks) * 100
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to upload chunk {chunk_index}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload chunk: {str(e)}")

@app.post("/api/admin/finalize-chunked-upload")
async def finalize_chunked_upload(upload_id: str = Form(...), db: Session = Depends(db_dependency)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        session = db.query(UploadSession).filter(UploadSession.id == upload_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        chunks_received = session.chunks_received or []
        if len(chunks_received) != session.total_chunks:
            missing_chunks = set(range(session.total_chunks)) - set(chunks_received)
            raise HTTPException(
                status_code=400,
                detail=f"Missing chunks: {sorted(list(missing_chunks))[:10]}{'...' if len(missing_chunks) > 10 else ''}"
            )
        
        logger.info(f"🔄 Combining {session.total_chunks} chunks for upload {upload_id}")
        
        final_file_path = os.path.join(CHUNK_UPLOAD_DIR, upload_id, session.filename)
        
        with open(final_file_path, 'wb') as final_file:
            for chunk_index in range(session.total_chunks):
                chunk_path = os.path.join(CHUNK_UPLOAD_DIR, upload_id, f"chunk_{chunk_index:06d}")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as chunk_file:
                        final_file.write(chunk_file.read())
                    os.remove(chunk_path)
                else:
                    raise HTTPException(status_code=500, detail=f"Chunk {chunk_index} file not found")
        
        logger.info(f"✅ Successfully combined all chunks into {final_file_path}")
        
        result = await process_uploaded_zip(final_file_path, session.agent_id, db)
        
        db.delete(session)
        db.commit()
        cleanup_upload_session(upload_id)
        
        return result
    except HTTPException:
        cleanup_upload_session(upload_id)
        raise
    except Exception as e:
        logger.error(f"❌ Failed to finalize upload {upload_id}: {e}", exc_info=True)
        cleanup_upload_session(upload_id)
        raise HTTPException(status_code=500, detail=f"Failed to finalize upload: {str(e)}")

def cleanup_upload_session(upload_id: str):
    try:
        upload_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            logger.info(f"🧹 Cleaned up upload directory: {upload_dir}")
    except Exception as e:
        logger.error(f"❌ Error cleaning up upload session {upload_id}: {e}", exc_info=True)

async def process_uploaded_zip(file_path: str, agent_id: str, db: Session):
    temp_files_created = []
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        logger.info(f"🔄 Processing ZIP file: {file_path} for agent: {agent_id}")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"ZIP file not found: {file_path}")
        
        if not zipfile.is_zipfile(file_path):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive")
        
        images_processed = 0
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            image_files = [
                f for f in zip_ref.namelist()
                if f.lower().endswith(image_extensions) and
                not f.startswith(('__MACOSX/', '.', 'thumbs.db')) and
                '/' not in f.split('/')[-1]
            ]
            
            if not image_files:
                raise HTTPException(status_code=400, detail="No valid image files found in ZIP archive")
            
            logger.info(f"📸 Found {len(image_files)} valid images in ZIP file")
            
            tasks_to_add = []
            for idx, image_file in enumerate(image_files):
                try:
                    image_data = zip_ref.read(image_file)
                    if len(image_data) == 0:
                        logger.warning(f"⚠️ Skipping empty image: {image_file}")
                        continue
                    
                    original_name = os.path.basename(image_file)
                    file_extension = os.path.splitext(original_name)[1].lower() or '.jpg'
                    unique_filename = f"task_{agent_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{idx:04d}_{uuid.uuid4().hex[:8]}{file_extension}"
                    image_path = os.path.join(STATIC_TASKS_DIR, unique_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    temp_files_created.append(image_path)
                    
                    task_progress = TaskProgress(
                        agent_id=agent_id,
                        image_filename=unique_filename,
                        image_path=f"/static/task_images/{unique_filename}",
                        status="pending",
                        assigned_at=datetime.utcnow()
                    )
                    tasks_to_add.append(task_progress)
                    images_processed += 1
                    
                    logger.info(f"✅ Processed image {idx + 1}/{len(image_files)}: {unique_filename}")
                except Exception as e:
                    logger.error(f"❌ Error processing image {image_file}: {e}", exc_info=True)
                    continue
            
            if not tasks_to_add:
                raise HTTPException(status_code=400, detail="No images could be processed successfully")
            
            for task in tasks_to_add:
                db.add(task)
            db.commit()
            temp_files_created.clear()
        
        logger.info(f"✅ Successfully created {images_processed} tasks for agent {agent_id}")
        
        return {
            "status": "success",
            "images_processed": images_processed,
            "message": f"Successfully processed {images_processed} images and assigned tasks to agent {agent_id}",
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing ZIP file: {e}", exc_info=True)
        db.rollback()
        for temp_file in temp_files_created:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"🧹 Cleaned up failed upload file: {temp_file}")
            except Exception as cleanup_error:
                logger.error(f"❌ Error cleaning up file {temp_file}: {cleanup_error}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ZIP processing failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"🧹 Cleaned up ZIP file: {file_path}")
            except Exception as e:
                logger.error(f"❌ Error cleaning up ZIP file: {e}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting periodic cleanup task...")
    logger.info(f"🌍 Domain: {os.environ.get('DOMAIN', 'agent-task-system.com')}")
    logger.info(f"🔗 Allowed Origins: {len(ALLOWED_ORIGINS)} configured")
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    while True:
        try:
            if database_ready:
                db = next(db_dependency())
                try:
                    now = datetime.utcnow()
                    expired_sessions = db.query(UploadSession).filter(
                        (now - UploadSession.created_at).total_seconds() > 7200
                    ).all()
                    
                    for session in expired_sessions:
                        logger.info(f"🧹 Cleaning up expired upload session: {session.id}")
                        cleanup_upload_session(session.id)
                        db.delete(session)
                    
                    if expired_sessions:
                        db.commit()
                        logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired upload sessions")
                except Exception as e:
                    logger.error(f"❌ Error in periodic cleanup: {e}", exc_info=True)
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"❌ Error in periodic cleanup setup: {e}", exc_info=True)
        
        await asyncio.sleep(3600)

@app.get("/api/admin/upload-sessions")
async def get_upload_sessions(db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        return {"upload_sessions": {}, "error": "Database not ready"}
    
    try:
        sessions = db.query(UploadSession).all()
        sessions_info = {}
        for session in sessions:
            chunks_received = session.chunks_received or []
            sessions_info[session.id] = {
                "filename": session.filename,
                "filesize": session.filesize,
                "total_chunks": session.total_chunks,
                "received_chunks": len(chunks_received),
                "progress": (len(chunks_received) / session.total_chunks) * 100,
                "created_at": session.created_at.isoformat(),
                "age_minutes": (datetime.utcnow() - session.created_at).total_seconds() / 60
            }
        return {"upload_sessions": sessions_info}
    except Exception as e:
        logger.error(f"❌ Error getting upload sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving upload sessions: {str(e)}")

@app.post("/api/admin/reset-password/{agent_id}")
async def reset_agent_password(agent_id: str, db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        new_password = ''.join(secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(12))
        agent.hashed_password = pwd_context.hash(new_password)
        db.commit()
        
        return {
            "success": True,
            "new_password": new_password,
            "message": f"Password reset successfully for agent {agent_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error resetting password for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Password reset failed: {str(e)}")

@app.get("/api/admin/agent-password/{agent_id}")
async def get_agent_password(agent_id: str, db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return {
            "message": f"Password for agent {agent_id} cannot be retrieved directly due to hashing",
            "agent_id": agent_id,
            "reset_endpoint": f"/api/admin/reset-password/{agent_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting password for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving password: {str(e)}")

@app.patch("/api/agents/{agent_id}/status")
async def update_agent_status(agent_id: str, status_data: dict, db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        new_status = status_data.get("status")
        if new_status not in ["active", "inactive"]:
            raise HTTPException(status_code=400, detail="Status must be 'active' or 'inactive'")
        
        agent.status = new_status
        db.commit()
        
        return {
            "success": True,
            "message": f"Agent {agent_id} status updated to {new_status}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating status for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")

@app.post("/api/admin/force-logout/{agent_id}")
async def force_logout_agent(agent_id: str, db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        active_session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).first()
        
        if active_session:
            active_session.logout_time = datetime.utcnow()
            active_session.duration_minutes = (active_session.logout_time - active_session.login_time).total_seconds() / 60
            db.commit()
            return {"success": True, "message": f"Agent {agent_id} logged out successfully"}
        else:
            return {"success": True, "message": f"Agent {agent_id} was not logged in"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error forcing logout for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Force logout failed: {str(e)}")

@app.get("/api/admin/preview-data")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(db_dependency),
    admin: dict = Depends(get_current_admin)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if agent_id and not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        if date_from:
            try:
                datetime.strptime(date_from, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                datetime.strptime(date_to, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
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
            result.append({
                "id": submission.id,
                "agent_id": submission.agent_id,
                "task_id": submission.task_id,
                "image_filename": submission.image_filename,
                "submitted_at": submission.submitted_at.isoformat(),
                "form_data": submission.form_data
            })
        
        return {"submissions": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in data preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@app.get("/api/admin/test-data")
async def test_data_availability(db: Session = Depends(db_dependency), admin: dict = Depends(get_current_admin)):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
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
        logger.error(f"❌ Error testing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data test failed: {str(e)}")

@app.get("/api/admin/session-report")
async def get_session_report(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(db_dependency),
    admin: dict = Depends(get_current_admin)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if agent_id and not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        if date_from:
            try:
                datetime.strptime(date_from, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                datetime.strptime(date_to, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        query = db.query(AgentSession).join(Agent)
        if agent_id:
            query = query.filter(AgentSession.agent_id == agent_id)
        if date_from:
            query = query.filter(AgentSession.login_time >= datetime.strptime(date_from, '%Y-%m-%d'))
        if date_to:
            query = query.filter(AgentSession.login_time <= datetime.strptime(date_to, '%Y-%m-%d'))
        
        sessions = query.order_by(AgentSession.login_time.desc()).limit(100).all()
        result = []
        for session in sessions:
            duration_minutes = None
            if session.logout_time and session.login_time:
                duration = session.logout_time - session.login_time
                duration_minutes = int(duration.total_seconds() / 60)
            
            result.append({
                "agent_id": session.agent_id,
                "agent_name": session.agent.name if session.agent else "Unknown",
                "login_time": session.login_time.isoformat() if session.login_time else None,
                "logout_time": session.logout_time.isoformat() if session.logout_time else None,
                "duration_minutes": duration_minutes
            })
        
        return {"sessions": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in session report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Session report failed: {str(e)}")

@app.get("/api/admin/export-excel")
async def export_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(db_dependency),
    admin: dict = Depends(get_current_admin)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if agent_id and not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        if date_from:
            try:
                datetime.strptime(date_from, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                datetime.strptime(date_to, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        query = db.query(SubmittedForm)
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        if date_from:
            query = query.filter(SubmittedForm.submitted_at >= datetime.strptime(date_from, '%Y-%m-%d'))
        if date_to:
            query = query.filter(SubmittedForm.submitted_at <= datetime.strptime(date_to, '%Y-%m-%d'))
        
        submissions = query.all()
        if not submissions:
            raise HTTPException(status_code=404, detail="No data found")
        
        excel_data = []
        for submission in submissions:
            row = {
                "Submission_ID": submission.id,
                "Agent_ID": submission.agent_id,
                "Task_ID": submission.task_id,
                "Image_Filename": submission.image_filename,
                "Submitted_At": submission.submitted_at.isoformat()
            }
            form_data = submission.form_data or {}
            for key, value in form_data.items():
                row[f"Form_{key}"] = str(value)  # Flatten complex data
            excel_data.append(row)
        
        df = pd.DataFrame(excel_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Submissions", index=False)
            for column in writer.sheets["Submissions"].columns:
                max_length = max(len(str(cell.value or "")) for cell in column) + 2
                writer.sheets["Submissions"].column_dimensions[column[0].column_letter].width = min(max_length, 50)
        
        output.seek(0)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=submissions_{timestamp}.xlsx"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in Excel export: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/admin/export-sessions")
async def export_sessions(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(db_dependency),
    admin: dict = Depends(get_current_admin)
):
    if not database_ready:
        raise HTTPException(status_code=503, detail="Database not ready")
    
    try:
        if agent_id and not agent_id.startswith("AG"):
            raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        if date_from:
            try:
                datetime.strptime(date_from, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                datetime.strptime(date_to, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        query = db.query(AgentSession).join(Agent)
        if agent_id:
            query = query.filter(AgentSession.agent_id == agent_id)
        if date_from:
            query = query.filter(AgentSession.login_time >= datetime.strptime(date_from, '%Y-%m-%d'))
        if date_to:
            query = query.filter(AgentSession.login_time <= datetime.strptime(date_to, '%Y-%m-%d'))
        
        sessions = query.all()
        if not sessions:
            raise HTTPException(status_code=404, detail="No session data found")
        
        excel_data = []
        for session in sessions:
            duration_minutes = None
            if session.logout_time and session.login_time:
                duration = session.logout_time - session.login_time
                duration_minutes = int(duration.total_seconds() / 60)
            
            excel_data.append({
                "Agent_ID": session.agent_id,
                "Agent_Name": session.agent.name if session.agent else "Unknown",
                "Login_Time": session.login_time.isoformat() if session.login_time else None,
                "Logout_Time": session.logout_time.isoformat() if session.logout_time else None,
                "Duration_Minutes": duration_minutes,
                "IP_Address": session.ip_address,
                "User_Agent": session.user_agent
            })
        
        df = pd.DataFrame(excel_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sessions", index=False)
            for column in writer.sheets["Sessions"].columns:
                max_length = max(len(str(cell.value or "")) for cell in column) + 2
                writer.sheets["Sessions"].column_dimensions[column[0].column_letter].width = min(max_length, 50)
        
        output.seek(0)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=sessions_{timestamp}.xlsx"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in session export: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Session export failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("=" * 60)
    logger.info("🚀 CLIENT RECORDS DATA ENTRY SYSTEM v2.0.1")
    logger.info("=" * 60)
    logger.info(f"🌍 Domain: {os.environ.get('DOMAIN', 'agent-task-system.com')}")
    logger.info(f"🔗 CORS Origins: {len(ALLOWED_ORIGINS)} configured")
    logger.info(f"📁 Chunk upload directory: {CHUNK_UPLOAD_DIR}")
    logger.info(f"💾 Database ready: {database_ready}")
    logger.info(f"🛣️ Routes ready: {routes_ready}")
    logger.info(f"🏃 Starting server on port {port}")
    logger.info("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
