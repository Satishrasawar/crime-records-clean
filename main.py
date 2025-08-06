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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Initialize FastAPI app first
app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0",
    description="Enhanced system for agent-task-system.com with chunked upload support and custom domain"
)

# Enhanced CORS Origins handling
ALLOWED_ORIGINS = []
if os.environ.get("ALLOWED_ORIGINS"):
    ALLOWED_ORIGINS = [origin.strip() for origin in os.environ.get("ALLOWED_ORIGINS").split(",") if origin.strip()]
else:
    ALLOWED_ORIGINS = [
        "https://agent-task-system.com",
        "https://www.agent-task-system.com", 
        "https://web-railwaybuilderherokupython.up.railway.app",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

# Enhanced CORS middleware with custom domain support
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type", 
        "X-Session-Token",
        "X-Requested-With",
        "Accept",
        "Origin",
        "User-Agent",
        "DNT",
        "Cache-Control",
        "X-Mx-ReqToken",
        "Keep-Alive",
        "If-Modified-Since"
    ],
    expose_headers=["Content-Disposition"]
)

# Enhanced request middleware for domain detection, logging, and security
@app.middleware("http")
async def enhanced_request_middleware(request, call_next):
    """Enhanced middleware for domain detection, logging, and security"""
    host = request.headers.get("host", "unknown")
    origin = request.headers.get("origin", "unknown")
    
    # Log domain information for debugging (exclude health checks to reduce noise)
    if not request.url.path.startswith("/health") and not host.startswith(("127.0.0.1", "localhost")):
        print(f"🌍 Request - Host: {host}, Origin: {origin}, Path: {request.url.path}")
    
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Add domain-specific headers
    if "agent-task-system.com" in host:
        response.headers["X-Domain-Status"] = "production"
        response.headers["X-Environment"] = "production"
    elif "railway.app" in host:
        response.headers["X-Domain-Status"] = "railway"
        response.headers["X-Environment"] = "staging"
    else:
        response.headers["X-Domain-Status"] = "development"
        response.headers["X-Environment"] = "development"
    
    return response

# Chunked upload configuration
CHUNK_UPLOAD_DIR = "temp_chunks"
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

# In-memory storage for upload sessions
upload_sessions: Dict[str, Dict[str, Any]] = {}

# Try to import and setup database
try:
    print("📦 Importing database modules...")
    from database import Base, engine, get_db
    from models import Agent, TaskProgress, SubmittedForm, AgentSession
    
    print("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    # Enhanced logging for domain-aware debugging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    database_ready = True
except Exception as e:
    print(f"❌ Database setup failed: {e}")
    import traceback
    traceback.print_exc()
    database_ready = False

# Try to import and include agent routes
try:
    print("📦 Importing agent routes...")
    from agent_routes import router as agent_router
    app.include_router(agent_router)
    print("✅ Agent routes included successfully!")
    routes_ready = True
except Exception as e:
    print(f"❌ Agent routes failed: {e}")
    routes_ready = False

# Create directories and mount static files
try:
    os.makedirs("static/task_images", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Static files configured")
except Exception as e:
    print(f"❌ Static files setup failed: {e}")

# ===================== ENHANCED HEALTH CHECK - FIXED =====================
@app.get("/health")
def health_check():
    """Enhanced health check with proper database connectivity testing"""
    health_status = {
        "status": "healthy",
        "platform": "Railway",
        "message": "Service is running",
        "timestamp": datetime.now().isoformat(),
        "domain": os.environ.get("DOMAIN", "not_set"),
        "database": "unknown",
        "imports_loaded": "database" in sys.modules,
        "chunked_upload": "enabled",
        "version": "2.0.0"
    }
    
    # Test database connectivity with proper session handling
    if database_ready:
        try:
            db_gen = get_db()
            db = next(db_gen)
            try:
                # Use proper SQLAlchemy 2.0 syntax with text()
                result = db.execute(text("SELECT 1")).scalar()
                if result == 1:
                    health_status["database"] = "connected"
                else:
                    health_status["database"] = "query_failed"
                    health_status["status"] = "degraded"
            except Exception as query_error:
                health_status["database"] = f"query_error: {str(query_error)[:50]}"
                health_status["status"] = "degraded"
            finally:
                db.close()
        except Exception as conn_error:
            health_status["database"] = f"connection_error: {str(conn_error)[:50]}"
            health_status["status"] = "degraded"
    else:
        health_status["database"] = "not_ready"
        health_status["status"] = "degraded"
    
    # Check static directory
    if os.path.exists("static/task_images"):
        health_status["static_storage"] = "ready"
    else:
        health_status["static_storage"] = "missing"
        
    # Check temp directory for uploads
    if os.path.exists(CHUNK_UPLOAD_DIR):
        health_status["upload_storage"] = "ready"
        health_status["active_uploads"] = len(upload_sessions)
    else:
        health_status["upload_storage"] = "missing"
    
    return health_status

# Enhanced root endpoint
@app.get("/")
def root():
    """Root endpoint with domain information"""
    return {
        "message": "Client Records Data Entry System API v2.0",
        "status": "running",
        "platform": "Railway",
        "domain": os.environ.get("DOMAIN", "railway"),
        "health_check": "/health",
        "admin_panel": "/admin.html",
        "agent_panel": "/agent.html",
        "features": [
            "chunked_upload", 
            "large_file_support", 
            "custom_domain_support",
            "ssl_enabled",
            "enhanced_security"
        ]
    }

# ===================== DOMAIN-AWARE STATIC FILE SERVING =====================

@app.get("/admin.html")
async def serve_admin_panel():
    """Serve admin dashboard with domain detection"""
    try:
        if os.path.exists("admin.html"):
            return FileResponse("admin.html", headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            })
        return JSONResponse({"error": "Admin panel not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Could not serve admin panel: {e}"}, status_code=500)

@app.get("/agent.html") 
async def serve_agent_panel():
    """Serve agent interface with domain detection"""
    try:
        if os.path.exists("agent.html"):
            return FileResponse("agent.html", headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache", 
                "Expires": "0"
            })
        return JSONResponse({"error": "Agent panel not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Could not serve agent panel: {e}"}, status_code=500)

# ===================== ENHANCED DEBUG ENDPOINTS =====================

@app.get("/debug")
def debug_info():
    """Enhanced debug endpoint with domain information"""
    return {
        "environment": {
            "domain": os.environ.get("DOMAIN", "not_set"),
            "port": os.environ.get("PORT", "not_set"),
            "database_url_set": bool(os.environ.get("DATABASE_URL")),
            "allowed_origins": ALLOWED_ORIGINS,
            "allowed_origins_count": len(ALLOWED_ORIGINS)
        },
        "system": {
            "files": os.listdir("."),
            "python_version": sys.version,
            "database_ready": database_ready,
            "routes_ready": routes_ready
        },
        "features": {
            "upload_sessions": len(upload_sessions),
            "chunk_upload_dir_exists": os.path.exists(CHUNK_UPLOAD_DIR),
            "static_dir_exists": os.path.exists("static"),
            "static_images_dir_exists": os.path.exists("static/task_images")
        }
    }

@app.get("/status")
def system_status():
    """Enhanced system status endpoint"""
    return {
        "status": "operational",
        "database": "ready" if database_ready else "failed",
        "routes": "ready" if routes_ready else "failed", 
        "domain": os.environ.get("DOMAIN", "railway"),
        "health": "ok",
        "chunked_upload": "enabled",
        "active_uploads": len(upload_sessions),
        "cors_origins": len(ALLOWED_ORIGINS)
    }

# ===================== STATISTICS ENDPOINT =====================
@app.get("/api/admin/statistics")
async def get_admin_statistics(db: Session = Depends(get_db)):
    """Get admin dashboard statistics"""
    try:
        if not database_ready:
            return {
                "total_agents": 0,
                "total_tasks": 0,
                "completed_tasks": 0,
                "pending_tasks": 0,
                "in_progress_tasks": 0
            }
        
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
        print(f"❌ Error getting statistics: {e}")
        return {
            "total_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0
        }

# ===================== AGENTS ENDPOINTS =====================
@app.get("/api/agents")
async def list_agents(db: Session = Depends(get_db)):
    """List all agents with their statistics"""
    try:
        if not database_ready:
            return []
        
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
                "password": agent.password,
                "status": agent.status,
                "tasks_completed": completed_tasks,
                "total_tasks": total_tasks,
                "last_login": latest_session.login_time.isoformat() if latest_session and latest_session.login_time else None,
                "last_logout": latest_session.logout_time.isoformat() if latest_session and latest_session.logout_time else None,
                "is_currently_logged_in": latest_session.logout_time is None if latest_session else False
            }
            agent_list.append(agent_data)
        
        return agent_list
    except Exception as e:
        print(f"❌ Error listing agents: {e}")
        return []

# ===================== TASK ENDPOINTS FOR AGENTS =====================
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
            next_task.started_at = datetime.now()
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
        print(f"❌ Error getting current task for {agent_id}: {e}")
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
        print(f"❌ Error getting tasks for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting tasks: {str(e)}")

@app.get("/api/agents/{agent_id}/statistics")
async def get_agent_statistics(agent_id: str, db: Session = Depends(get_db)):
    """Get statistics for a specific agent"""
    try:
        if not database_ready:
            return {
                "total_tasks": 0,
                "completed_tasks": 0,
                "pending_tasks": 0,
                "in_progress_tasks": 0
            }
        
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
        print(f"❌ Error getting statistics for {agent_id}: {e}")
        return {
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "in_progress_tasks": 0
        }

# ===================== AGENT REGISTRATION ENDPOINT - FIXED =====================
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
    """Register a new agent with proper date handling"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        # Validate date format
        try:
            datetime.strptime(dob, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Check if email already exists
        existing_agent = db.query(Agent).filter(Agent.email == email).first()
        if existing_agent:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Generate unique agent ID
        agent_id = f"AG{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4())[:4].upper()}"
        
        # Generate secure password
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(12))
        
        # Create agent record
        new_agent = Agent(
            agent_id=agent_id,
            name=name,
            email=email,
            mobile=mobile,
            dob=dob,  # Store as string for compatibility
            country=country,
            gender=gender,
            password=password,
            status="active",
            created_at=datetime.now()
        )
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        print(f"✅ New agent registered: {agent_id}")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "password": password,
            "message": "Agent registered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error registering agent: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# ===================== FORM SUBMISSION ENDPOINT - FIXED =====================
@app.post("/api/agents/{agent_id}/submit")
async def submit_task_form(
    agent_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Submit completed task form - handles both JSON and form data"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Handle both JSON and form data from frontend
        content_type = request.headers.get("content-type", "")
        
        if content_type.startswith("application/json"):
            data = await request.json()
        else:
            # Handle form data from frontend
            form_data = await request.form()
            data = dict(form_data)
            # Remove metadata fields
            data.pop('agent_id', None)
            data.pop('task_id', None)
        
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
        
        # Create submitted form record
        submitted_form = SubmittedForm(
            agent_id=agent_id,
            task_id=current_task.id,
            image_filename=current_task.image_filename,
            form_data=data,  # Store as JSON
            submitted_at=datetime.now()
        )
        
        db.add(submitted_form)
        
        # Mark task as completed
        current_task.status = 'completed'
        current_task.completed_at = datetime.now()
        
        # Commit changes
        db.commit()
        
        print(f"✅ Task {current_task.id} completed by agent {agent_id}")
        
        return {
            "success": True,
            "message": "Task submitted successfully",
            "task_id": current_task.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting task for {agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")

# ===================== STANDARD UPLOAD ENDPOINT =====================
@app.post("/api/admin/upload-tasks")
async def upload_tasks_standard(
    zip_file: UploadFile = File(...),
    agent_id: str = Form(...),
    db: Session = Depends(get_db)
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
        
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uuid.uuid4().hex}_{zip_file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await zip_file.read()
            buffer.write(content)
        
        # Process the ZIP file
        result = await process_uploaded_zip(temp_file_path, agent_id, db)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# ===================== CHUNKED UPLOAD ENDPOINTS =====================

@app.post("/api/admin/init-chunked-upload")
async def init_chunked_upload(
    filename: str = Form(...),
    filesize: int = Form(...),
    total_chunks: int = Form(...),
    agent_id: str = Form(...)
):
    """Initialize a chunked upload session for large files"""
    try:
        # Validate agent exists (if database is ready)
        if database_ready:
            db_gen = get_db()
            db = next(db_gen)
            try:
                agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                if agent.status != "active":
                    raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not active")
            finally:
                db.close()
        
        # Create unique upload ID
        upload_id = str(uuid.uuid4())
        upload_dir = os.path.join(CHUNK_UPLOAD_DIR, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Store upload session info
        upload_sessions[upload_id] = {
            "filename": filename,
            "filesize": filesize,
            "total_chunks": total_chunks,
            "agent_id": agent_id,
            "received_chunks": set(),
            "upload_dir": upload_dir,
            "created_at": datetime.now()
        }
        
        print(f"🚀 Initialized chunked upload: {upload_id} for {filename} ({filesize} bytes, {total_chunks} chunks)")
        
        return {
            "upload_id": upload_id, 
            "status": "initialized",
            "message": f"Ready to receive {total_chunks} chunks"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to initialize chunked upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize upload: {str(e)}")

@app.post("/api/admin/upload-chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...)
):
    """Upload a single chunk of a large file"""
    try:
        if upload_id not in upload_sessions:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        session = upload_sessions[upload_id]
        
        # Validate chunk index
        if chunk_index >= session["total_chunks"] or chunk_index < 0:
            raise HTTPException(status_code=400, detail=f"Invalid chunk index: {chunk_index}")
        
        # Check if chunk already uploaded
        if chunk_index in session["received_chunks"]:
            return {
                "status": "chunk_already_exists",
                "chunk_index": chunk_index,
                "received_chunks": len(session["received_chunks"]),
                "total_chunks": session["total_chunks"]
            }
        
        chunk_path = os.path.join(session["upload_dir"], f"chunk_{chunk_index:06d}")
        
        # Save chunk to disk
        async with aiofiles.open(chunk_path, 'wb') as f:
            content = await chunk.read()
            await f.write(content)
        
        # Mark chunk as received
        session["received_chunks"].add(chunk_index)
        
        print(f"📦 Received chunk {chunk_index + 1}/{session['total_chunks']} for upload {upload_id}")
        
        return {
            "status": "chunk_uploaded",
            "chunk_index": chunk_index,
            "received_chunks": len(session["received_chunks"]),
            "total_chunks": session["total_chunks"],
            "progress_percentage": (len(session["received_chunks"]) / session["total_chunks"]) * 100
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to upload chunk {chunk_index}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload chunk: {str(e)}")

@app.post("/api/admin/finalize-chunked-upload")
async def finalize_chunked_upload(upload_id: str = Form(...), db: Session = Depends(get_db)):
    """Combine all chunks and process the complete file"""
    try:
        if upload_id not in upload_sessions:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        session = upload_sessions[upload_id]
        
        # Verify all chunks received
        if len(session["received_chunks"]) != session["total_chunks"]:
            missing_chunks = set(range(session["total_chunks"])) - session["received_chunks"]
            raise HTTPException(
                status_code=400, 
                detail=f"Missing chunks: {sorted(list(missing_chunks))[:10]}{'...' if len(missing_chunks) > 10 else ''}"
            )
        
        print(f"🔄 Combining {session['total_chunks']} chunks for upload {upload_id}")
        
        # Combine chunks into final file
        final_file_path = os.path.join(session["upload_dir"], session["filename"])
        
        with open(final_file_path, 'wb') as final_file:
            for chunk_index in range(session["total_chunks"]):
                chunk_path = os.path.join(session["upload_dir"], f"chunk_{chunk_index:06d}")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as chunk_file:
                        final_file.write(chunk_file.read())
                    # Clean up chunk file immediately
                    os.remove(chunk_path)
                else:
                    raise HTTPException(status_code=500, detail=f"Chunk {chunk_index} file not found")
        
        print(f"✅ Successfully combined all chunks into {final_file_path}")
        
        # Process the complete ZIP file
        result = await process_uploaded_zip(final_file_path, session["agent_id"], db)
        
        # Clean up upload session
        cleanup_upload_session(upload_id)
        
        return result
        
    except HTTPException:
        cleanup_upload_session(upload_id)
        raise
    except Exception as e:
        print(f"❌ Failed to finalize upload {upload_id}: {e}")
        cleanup_upload_session(upload_id)
        raise HTTPException(status_code=500, detail=f"Failed to finalize upload: {str(e)}")

def cleanup_upload_session(upload_id: str):
    """Clean up upload session and temporary files"""
    try:
        if upload_id in upload_sessions:
            session = upload_sessions[upload_id]
            upload_dir = session["upload_dir"]
            
            # Remove temporary directory and all contents
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
                print(f"🧹 Cleaned up upload directory: {upload_dir}")
            
            # Remove session from memory
            del upload_sessions[upload_id]
            print(f"🧹 Cleaned up upload session: {upload_id}")
            
    except Exception as e:
        print(f"❌ Error cleaning up upload session {upload_id}: {e}")

# ===================== ENHANCED ZIP PROCESSING FUNCTION =====================
async def process_uploaded_zip(file_path: str, agent_id: str, db: Session):
    """Enhanced ZIP file processing with comprehensive error handling and cleanup"""
    temp_files_created = []
    
    try:
        if not database_ready:
            raise Exception("Database not ready")
        
        print(f"🔄 Processing ZIP file: {file_path} for agent: {agent_id}")
        
        images_processed = 0
        
        # Verify ZIP file exists and is readable
        if not os.path.exists(file_path):
            raise Exception(f"ZIP file not found: {file_path}")
            
        if not zipfile.is_zipfile(file_path):
            raise Exception("Uploaded file is not a valid ZIP archive")
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Get image files from ZIP with better filtering
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
            image_files = []
            
            for file_info in zip_ref.filelist:
                filename = file_info.filename
                # Skip directories, hidden files, and system files
                if (not file_info.is_dir() and 
                    filename.lower().endswith(image_extensions) and
                    not filename.startswith(('__MACOSX/', '.', 'thumbs.db')) and
                    '/' not in filename.split('/')[-1]):  # Only files in root or simple subdirs
                    image_files.append(filename)
            
            if not image_files:
                raise Exception("No valid image files found in ZIP archive. Supported formats: JPG, JPEG, PNG, GIF, BMP, WEBP")
            
            print(f"📸 Found {len(image_files)} valid images in ZIP file")
            
            # Create static directory with proper permissions
            static_dir = "static/task_images"
            os.makedirs(static_dir, exist_ok=True)
            
            # Process images with transaction safety
            tasks_to_add = []
            
            for idx, image_file in enumerate(image_files):
                try:
                    # Extract image data
                    image_data = zip_ref.read(image_file)
                    
                    if len(image_data) == 0:
                        print(f"⚠️ Skipping empty image: {image_file}")
                        continue
                    
                    # Create unique filename with timestamp and random component
                    original_name = os.path.basename(image_file)
                    file_extension = os.path.splitext(original_name)[1].lower()
                    if not file_extension:
                        file_extension = '.jpg'  # Default extension
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_id = str(uuid.uuid4())[:8]
                    unique_filename = f"task_{agent_id}_{timestamp}_{idx:04d}_{unique_id}{file_extension}"
                    
                    # Save to static directory
                    image_path = os.path.join(static_dir, unique_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    temp_files_created.append(image_path)  # Track for cleanup if needed
                    
                    # Create task object (don't add to DB yet)
                    task_progress = TaskProgress(
                        agent_id=agent_id,
                        image_filename=unique_filename,
                        image_path=f"/static/task_images/{unique_filename}",
                        status="pending",
                        assigned_at=datetime.now()
                    )
                    tasks_to_add.append(task_progress)
                    images_processed += 1
                    
                    print(f"✅ Processed image {idx + 1}/{len(image_files)}: {unique_filename}")
                    
                except Exception as image_error:
                    print(f"❌ Error processing image {image_file}: {image_error}")
                    continue
            
            if not tasks_to_add:
                raise Exception("No images could be processed successfully")
            
            # Add all tasks to database in a single transaction
            try:
                for task in tasks_to_add:
                    db.add(task)
                db.commit()
                print(f"✅ Successfully created {len(tasks_to_add)} tasks for agent {agent_id}")
                temp_files_created.clear()  # Success - don't cleanup files
            except Exception as db_error:
                db.rollback()
                raise Exception(f"Database error while saving tasks: {str(db_error)}")
    
    except Exception as e:
        print(f"❌ Error processing ZIP file: {e}")
        db.rollback()
        
        # Cleanup any files created before the error
        for temp_file in temp_files_created:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🧹 Cleaned up failed upload file: {temp_file}")
            except Exception as cleanup_error:
                print(f"❌ Error cleaning up file {temp_file}: {cleanup_error}")
        
        raise Exception(f"ZIP processing failed: {str(e)}")
    
    finally:
        # Always clean up the original ZIP file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🧹 Cleaned up ZIP file: {file_path}")
            except Exception as cleanup_error:
                print(f"❌ Error cleaning up ZIP file: {cleanup_error}")
    
    return {
        "status": "success",
        "images_processed": images_processed,
        "message": f"Successfully processed {images_processed} images and assigned tasks to agent {agent_id}",
        "agent_id": agent_id,
        "timestamp": datetime.now().isoformat()
    }

# ===================== CLEANUP AND MAINTENANCE =====================

@app.on_event("startup")
async def startup_event():
    """Initialize cleanup tasks on startup"""
    print("🚀 Starting periodic cleanup task...")
    print(f"🌍 Domain: {os.environ.get('DOMAIN', 'railway')}")
    print(f"🔗 Allowed Origins: {len(ALLOWED_ORIGINS)} configured")
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Clean up old upload sessions every hour"""
    while True:
        try:
            now = datetime.now()
            expired_sessions = []
            
            for upload_id, session in upload_sessions.items():
                # Remove sessions older than 2 hours
                if (now - session["created_at"]).total_seconds() > 7200:
                    expired_sessions.append(upload_id)
            
            for upload_id in expired_sessions:
                print(f"🧹 Cleaning up expired upload session: {upload_id}")
                cleanup_upload_session(upload_id)
                
            if expired_sessions:
                print(f"🧹 Cleaned up {len(expired_sessions)} expired upload sessions")
                
        except Exception as e:
            print(f"❌ Error in periodic cleanup: {e}")
        
        # Wait 1 hour before next cleanup
        await asyncio.sleep(3600)

# ===================== UPLOAD SESSIONS MANAGEMENT =====================

@app.get("/api/admin/upload-sessions")
def get_upload_sessions():
    """Get current upload sessions (admin only)"""
    sessions_info = {}
    for upload_id, session in upload_sessions.items():
        sessions_info[upload_id] = {
            "filename": session["filename"],
            "filesize": session["filesize"],
            "total_chunks": session["total_chunks"],
            "received_chunks": len(session["received_chunks"]),
            "progress": (len(session["received_chunks"]) / session["total_chunks"]) * 100,
            "created_at": session["created_at"].isoformat(),
            "age_minutes": (datetime.now() - session["created_at"]).total_seconds() / 60
        }
    return {"upload_sessions": sessions_info}

# ===================== ADDITIONAL ADMIN ENDPOINTS =====================

@app.post("/api/admin/reset-password/{agent_id}")
async def reset_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Reset agent password"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Generate new password
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        new_password = ''.join(secrets.choice(alphabet) for _ in range(12))
        
        # Update password
        agent.password = new_password
        db.commit()
        
        return {
            "success": True,
            "new_password": new_password,
            "message": f"Password reset successfully for agent {agent_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error resetting password for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Password reset failed: {str(e)}")

@app.get("/api/admin/agent-password/{agent_id}")
async def get_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Get agent password information"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return {
            "message": f"Password for agent {agent_id} is: {agent.password}",
            "agent_id": agent_id,
            "password": agent.password
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting password for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving password: {str(e)}")

@app.patch("/api/agents/{agent_id}/status")
async def update_agent_status(agent_id: str, status_data: dict, db: Session = Depends(get_db)):
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
        print(f"❌ Error updating status for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Status update failed: {str(e)}")

@app.post("/api/admin/force-logout/{agent_id}")
async def force_logout_agent(agent_id: str, db: Session = Depends(get_db)):
    """Force logout an agent"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Find active session and close it
        active_session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).first()
        
        if active_session:
            active_session.logout_time = datetime.now()
            db.commit()
            return {"success": True, "message": f"Agent {agent_id} logged out successfully"}
        else:
            return {"success": True, "message": f"Agent {agent_id} was not logged in"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error forcing logout for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Force logout failed: {str(e)}")

@app.get("/api/admin/preview-data")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
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
        
    except Exception as e:
        print(f"❌ Error in data preview: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@app.get("/api/admin/test-data")
async def test_data_availability(db: Session = Depends(get_db)):
    """Test data availability"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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

@app.get("/api/admin/session-report")
async def get_session_report(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
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
        
    except Exception as e:
        print(f"❌ Error in session report: {e}")
        raise HTTPException(status_code=500, detail=f"Session report failed: {str(e)}")

# ===================== EXPORT ENDPOINTS (PLACEHOLDERS) =====================

@app.get("/api/admin/export-excel")
async def export_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export submitted data to Excel - ready for implementation"""
    return JSONResponse(
        content={
            "message": "Excel export feature - ready for implementation with pandas/openpyxl",
            "note": "Add pandas and openpyxl implementation here for full Excel export functionality"
        },
        status_code=501
    )

@app.get("/api/admin/export-sessions")
async def export_sessions(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export session report to Excel - ready for implementation"""
    return JSONResponse(
        content={
            "message": "Session export feature - ready for implementation", 
            "note": "Add pandas and openpyxl implementation here for full session export functionality"
        },
        status_code=501
    )

# ===================== MAIN ENTRY POINT =====================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("=" * 60)
    print("🚀 CLIENT RECORDS DATA ENTRY SYSTEM v2.0")
    print("=" * 60)
    print(f"🌍 Domain: {os.environ.get('DOMAIN', 'railway')}")
    print(f"🔗 CORS Origins: {len(ALLOWED_ORIGINS)} configured")
    print(f"📁 Chunk upload directory: {CHUNK_UPLOAD_DIR}")
    print(f"💾 Database ready: {database_ready}")
    print(f"🛣️ Routes ready: {routes_ready}")
    print(f"🏃 Starting server on port {port}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
            "
