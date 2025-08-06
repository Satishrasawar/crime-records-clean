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
from sqlalchemy import func
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status
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

# Initialize FastAPI app
app = FastAPI(
    title="Client Records Data Entry System",
    version="2.0.0",
    description="Enhanced system for agent-task-system.com with chunked upload support"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://agent-task-system.com", "https://www.agent-task-system.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/login")

# Chunked upload configuration
CHUNK_UPLOAD_DIR = "/app/temp_chunks"  # Use Railway volume
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

# Database setup
try:
    logger.info("📦 Importing database modules...")
    from database import Base, engine, get_db
    from models import Agent, TaskProgress, SubmittedForm, AgentSession, UploadSession
    
    logger.info("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created successfully!")
    database_ready = True
except Exception as e:
    logger.error(f"❌ Database setup failed: {e}", exc_info=True)
    database_ready = False

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

# Static files
try:
    static_dir = "/app/static/task_images"  # Use Railway volume
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info("✅ Static files configured")
except Exception as e:
    logger.error(f"❌ Static files setup failed: {e}", exc_info=True)

# JWT authentication
def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_admin(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/admin/login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    # Replace with proper admin credentials check (e.g., database table)
    if username == "admin" and password == "secure_password":
        token = create_access_token({"sub": username, "role": "admin"})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

# Health endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "platform": "Railway",
        "message": "Service is running",
        "imports_loaded": database_ready,
        "chunked_upload": "enabled"
    }

# Root endpoint
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Client Records Data Entry System API v2.0",
        "status": "running",
        "platform": "Railway",
        "health_check": "/health",
        "features": ["chunked_upload", "large_file_support"]
    }

# Statistics endpoint
@app.get("/api/admin/statistics")
async def get_admin_statistics(db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """Get admin dashboard statistics"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

# Agent endpoints
@app.get("/api/agents")
async def list_agents(db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """List all agents with their statistics"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
                "tasks_completed": completed_tasks,
                "total_tasks": total_tasks,
                "last_login": latest_session.login_time.isoformat() if latest_session and latest_session.login_time else None,
                "last_logout": latest_session.logout_time.isoformat() if latest_session and latest_session.logout_time else None,
                "is_currently_logged_in": latest_session.logout_time is None if latest_session else False
            }
            agent_list.append(agent_data)
        
        return agent_list
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error listing agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@app.get("/api/agents/{agent_id}/tasks/current")
async def get_current_task(agent_id: str, db: Session = Depends(get_db)):
    """Get current task for an agent"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
async def get_agent_tasks(agent_id: str, db: Session = Depends(get_db)):
    """Get all tasks for an agent"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
        
        return task_list
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting tasks for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting tasks: {str(e)}")

@app.get("/api/agents/{agent_id}/statistics")
async def get_agent_statistics(agent_id: str, db: Session = Depends(get_db)):
    """Get statistics for a specific agent"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
    db: Session = Depends(get_db)
):
    """Register a new agent"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
            created_at=datetime.utcnow()
        )
        
        db.add(new_agent)
        db.commit()
        
        logger.info(f"✅ New agent registered: {agent_id}")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "password": password,
            "message": "Agent registered successfully"
        }
    except Exception as e:
        logger.error(f"❌ Error registering agent: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# Form submission
@app.post("/api/agents/{agent_id}/submit")
async def submit_task_form(
    agent_id: str,
    data: dict,
    db: Session = Depends(get_db)
):
    """Submit completed task form"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
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
    db: Session = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Standard upload endpoint for smaller files"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        if agent.status != "active":
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not active")
        
        temp_file_path = os.path.join(CHUNK_UPLOAD_DIR, f"temp_{uuid.uuid4().hex}_{zip_file.filename}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await zip_file.read()
            buffer.write(content)
        
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
    db: Session = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Initialize a chunked upload session for large files"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
            chunks_received=[]
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
    db: Session = Depends(get_db)
):
    """Upload a single chunk of a large file"""
    try:
        session = db.query(UploadSession).filter(UploadSession.id == upload_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        if chunk_index >= session.total_chunks or chunk_index < 0:
            raise HTTPException(status_code=400, detail=f"Invalid chunk index: {chunk_index}")
        
        if chunk_index in session.chunks_received:
            return {
                "status": "chunk_already_exists",
                "chunk_index": chunk_index,
                "received_chunks": len(session.chunks_received),
                "total_chunks": session.total_chunks
            }
        
        chunk_path = os.path.join(CHUNK_UPLOAD_DIR, upload_id, f"chunk_{chunk_index:06d}")
        
        async with aiofiles.open(chunk_path, 'wb') as f:
            content = await chunk.read()
            await f.write(content)
        
        session.chunks_received.append(chunk_index)
        db.commit()
        
        logger.info(f"📦 Received chunk {chunk_index + 1}/{session.total_chunks} for upload {upload_id}")
        
        return {
            "status": "chunk_uploaded",
            "chunk_index": chunk_index,
            "received_chunks": len(session.chunks_received),
            "total_chunks": session.total_chunks,
            "progress_percentage": (len(session.chunks_received) / session.total_chunks) * 100
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to upload chunk {chunk_index}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload chunk: {str(e)}")

@app.post("/api/admin/finalize-chunked-upload")
async def finalize_chunked_upload(upload_id: str = Form(...), db: Session = Depends(get_db)):
    """Combine all chunks and process the complete file"""
    try:
        session = db.query(UploadSession).filter(UploadSession.id == upload_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        if len(session.chunks_received) != session.total_chunks:
            missing_chunks = set(range(session.total_chunks)) - set(session.chunks_received)
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
    """Clean up upload session and temporary files"""
    try:
        upload_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            logger.info(f"🧹 Cleaned up upload directory: {upload_dir}")
    except Exception as e:
        logger.error(f"❌ Error cleaning up upload session {upload_id}: {e}", exc_info=True)

async def process_uploaded_zip(file_path: str, agent_id: str, db: Session):
    """Process the uploaded ZIP file and create tasks"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        logger.info(f"🔄 Processing ZIP file: {file_path} for agent: {agent_id}")
        
        images_processed = 0
        
        if not zipfile.is_zipfile(file_path):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive")
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist()
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
                         and not f.startswith('__MACOSX/')
                         and not f.startswith('.')]
            
            if not image_files:
                raise HTTPException(status_code=400, detail="No valid image files found in ZIP archive")
            
            logger.info(f"📸 Found {len(image_files)} images in ZIP file")
            
            static_dir = "/app/static/task_images"
            os.makedirs(static_dir, exist_ok=True)
            
            for idx, image_file in enumerate(image_files):
                try:
                    image_data = zip_ref.read(image_file)
                    file_extension = os.path.splitext(image_file)[1].lower() or '.jpg'
                    unique_filename = f"task_{agent_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{idx:04d}{file_extension}"
                    image_path = os.path.join(static_dir, unique_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    task_progress = TaskProgress(
                        agent_id=agent_id,
                        image_filename=unique_filename,
                        image_path=f"/static/task_images/{unique_filename}",
                        status="pending",
                        assigned_at=datetime.utcnow()
                    )
                    db.add(task_progress)
                    images_processed += 1
                    
                    logger.info(f"✅ Processed image {idx + 1}/{len(image_files)}: {unique_filename}")
                except Exception as e:
                    logger.error(f"❌ Error processing image {image_file}: {e}", exc_info=True)
                    continue
            
            db.commit()
            logger.info(f"✅ Successfully created {images_processed} tasks for agent {agent_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing ZIP file: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🧹 Cleaned up ZIP file: {file_path}")
    
    return {
        "status": "success",
        "images_processed": images_processed,
        "message": f"Successfully processed {images_processed} images and assigned tasks to agent {agent_id}"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize cleanup tasks on startup"""
    logger.info("🚀 Starting periodic cleanup task...")
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Clean up old upload sessions every hour"""
    while True:
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
        
        await asyncio.sleep(3600)

@app.get("/admin.html")
async def serve_admin_panel():
    """Serve admin dashboard"""
    try:
        if os.path.exists("admin.html"):
            return FileResponse("admin.html")
        return {"error": "Admin panel not found", "files": os.listdir(".")}
    except Exception as e:
        logger.error(f"❌ Error serving admin panel: {e}", exc_info=True)
        return {"error": f"Could not serve admin panel: {str(e)}"}

@app.get("/agent.html")
async def serve_agent_panel():
    """Serve agent interface"""
    try:
        if os.path.exists("agent.html"):
            return FileResponse("agent.html")
        return {"error": "Agent panel not found", "files": os.listdir(".")}
    except Exception as e:
        logger.error(f"❌ Error serving agent panel: {e}", exc_info=True)
        return {"error": f"Could not serve agent panel: {str(e)}"}

@app.get("/debug")
def debug_info():
    """Debug endpoint to check system status"""
    return {
        "files": os.listdir("."),
        "python_path": sys.path,
        "modules": list(sys.modules.keys()),
        "database_ready": database_ready,
        "routes_ready": routes_ready,
        "port": os.environ.get("PORT", "not set"),
        "upload_sessions": db.query(UploadSession).count(),
        "chunk_upload_dir": os.path.exists(CHUNK_UPLOAD_DIR)
    }

@app.get("/status")
def system_status():
    """System status endpoint"""
    return {
        "database": "ready" if database_ready else "failed",
        "routes": "ready" if routes_ready else "failed",
        "health": "ok",
        "chunked_upload": "enabled",
        "active_uploads": db.query(UploadSession).count()
    }

@app.get("/api/admin/upload-sessions")
def get_upload_sessions(db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """Get current upload sessions (admin only)"""
    sessions = db.query(UploadSession).all()
    sessions_info = {}
    for session in sessions:
        sessions_info[session.id] = {
            "filename": session.filename,
            "filesize": session.filesize,
            "total_chunks": session.total_chunks,
            "received_chunks": len(session.chunks_received),
            "progress": (len(session.chunks_received) / session.total_chunks) * 100,
            "created_at": session.created_at.isoformat(),
            "age_minutes": (datetime.utcnow() - session.created_at).total_seconds() / 60
        }
    return {"upload_sessions": sessions_info}

@app.post("/api/admin/reset-password/{agent_id}")
async def reset_agent_password(agent_id: str, db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """Reset agent password"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
async def get_agent_password(agent_id: str, db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """Get agent password information"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
async def update_agent_status(agent_id: str, status_data: dict, db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """Update agent status"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
async def force_logout_agent(agent_id: str, db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """Force logout an agent"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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

@app.get("/api/admin/export-excel")
async def export_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Export submitted data to Excel"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
            row.update(submission.form_data)
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

@app.get("/api/admin/preview-data")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Preview submitted data"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in data preview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@app.get("/api/admin/test-data")
async def test_data_availability(db: Session = Depends(get_db), admin: dict = Depends(get_current_admin)):
    """Test data availability"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error testing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data test failed: {str(e)}")

@app.get("/api/admin/session-report")
async def get_session_report(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Get session report"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in session report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Session report failed: {str(e)}")

@app.get("/api/admin/export-sessions")
async def export_sessions(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
    admin: dict = Depends(get_current_admin)
):
    """Export session report to Excel"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"📁 Chunk upload directory: {CHUNK_UPLOAD_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)
