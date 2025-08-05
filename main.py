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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

# Initialize FastAPI app first
app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0",
    description="Enhanced system for agent-task-system.com with chunked upload support"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chunked upload configuration
CHUNK_UPLOAD_DIR = "temp_chunks"
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

# In-memory storage for upload sessions (use Redis in production)
upload_sessions: Dict[str, Dict[str, Any]] = {}

# CRITICAL: Health endpoint (must work first)
@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy", 
        "platform": "Railway",
        "message": "Service is running",
        "imports_loaded": "database" in sys.modules,
        "chunked_upload": "enabled"
    }

# ===================== FRONTEND ROUTES =====================
@app.get("/", response_class=HTMLResponse)
async def serve_agent_homepage():
    """Serve agent.html as the homepage"""
    try:
        if os.path.exists("agent.html"):
            with open("agent.html", "r", encoding="utf-8") as f:
                content = f.read()
            # Update the BASE_URL to use the current domain
            content = content.replace(
                "https://web-production-b3ef2.up.railway.app", 
                ""  # Use relative URLs
            )
            return HTMLResponse(content=content)
        else:
            return HTMLResponse("<h1>Agent interface not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading agent interface: {e}</h1>", status_code=500)

@app.get("/admin", response_class=HTMLResponse)
@app.get("/admin.html", response_class=HTMLResponse)
async def serve_admin_dashboard():
    """Serve admin.html with updated configuration"""
    try:
        if os.path.exists("admin.html"):
            with open("admin.html", "r", encoding="utf-8") as f:
                content = f.read()
            # Update the BASE_URL to use the current domain
            content = content.replace(
                "https://web-production-b3ef2.up.railway.app", 
                ""  # Use relative URLs
            )
            return HTMLResponse(content=content)
        else:
            return HTMLResponse("<h1>Admin dashboard not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading admin dashboard: {e}</h1>", status_code=500)

@app.get("/agent", response_class=HTMLResponse)
@app.get("/agent.html", response_class=HTMLResponse)
async def serve_agent_interface():
    """Serve agent.html with updated configuration"""
    try:
        if os.path.exists("agent.html"):
            with open("agent.html", "r", encoding="utf-8") as f:
                content = f.read()
            # Update the BASE_URL to use the current domain
            content = content.replace(
                "https://web-production-b3ef2.up.railway.app", 
                ""  # Use relative URLs
            )
            return HTMLResponse(content=content)
        else:
            return HTMLResponse("<h1>Agent interface not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading agent interface: {e}</h1>", status_code=500)

# API info endpoint
@app.get("/api")
def api_info():
    """API information endpoint"""
    return {
        "message": "Client Records Data Entry System API v2.0",
        "status": "running",
        "platform": "Railway",
        "health_check": "/health",
        "features": ["chunked_upload", "large_file_support"],
        "endpoints": {
            "admin": "/admin or /admin.html",
            "agent": "/agent or /agent.html or /",
            "health": "/health",
            "api_docs": "/docs"
        }
    }

# Try to import and setup database
try:
    print("📦 Importing database modules...")
    from database import Base, engine, get_db
    from models import Agent, TaskProgress, SubmittedForm, AgentSession
    
    print("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    # Enhanced logging for upload debugging
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

# Create directories
try:
    os.makedirs("static/task_images", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Static files configured")
except Exception as e:
    print(f"❌ Static files setup failed: {e}")

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
        
        # Count agents
        total_agents = db.query(Agent).count()
        
        # Count tasks from TaskProgress table
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
            # Count tasks for this agent
            total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent.agent_id).count()
            completed_tasks = db.query(TaskProgress).filter(
                TaskProgress.agent_id == agent.agent_id,
                TaskProgress.status == 'completed'
            ).count()
            
            # Get latest session info
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
        
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Find next pending task
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
            next_task.started_at = datetime.now()
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
        
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting tasks for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting tasks: {str(e)}")

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
        
        # Verify agent exists
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
            db = next(get_db())
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
    """Process the uploaded ZIP file and create tasks"""
    try:
        if not database_ready:
            raise Exception("Database not ready")
        
        print(f"🔄 Processing ZIP file: {file_path} for agent: {agent_id}")
        
        images_processed = 0
        
        # Verify ZIP file
        if not zipfile.is_zipfile(file_path):
            raise Exception("Uploaded file is not a valid ZIP archive")
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Get image files from ZIP
            image_files = [f for f in zip_ref.namelist() 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) 
                         and not f.startswith('__MACOSX/') 
                         and not f.startswith('.')]
            
            if not image_files:
                raise Exception("No valid image files found in ZIP archive")
            
            print(f"📸 Found {len(image_files)} images in ZIP file")
            
            # Create static directory if it doesn't exist
            static_dir = "static/task_images"
            os.makedirs(static_dir, exist_ok=True)
            
            # Extract images and create tasks
            for idx, image_file in enumerate(image_files):
                try:
                    # Extract image
                    image_data = zip_ref.read(image_file)
                    
                    # Create unique filename
                    file_extension = os.path.splitext(image_file)[1].lower()
                    if not file_extension:
                        file_extension = '.jpg'  # Default extension
                    
                    unique_filename = f"task_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:04d}{file_extension}"
                    
                    # Save to static directory
                    image_path = os.path.join(static_dir, unique_filename)
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    # Create task in database
                    task_progress = TaskProgress(
                        agent_id=agent_id,
                        image_filename=unique_filename,
                        image_path=f"/static/task_images/{unique_filename}",
                        status="pending",
                        assigned_at=datetime.now()
                    )
                    db.add(task_progress)
                    images_processed += 1
                    
                    print(f"✅ Processed image {idx + 1}/{len(image_files)}: {unique_filename}")
                    
                except Exception as e:
                    print(f"❌ Error processing image {image_file}: {e}")
                    continue
            
            # Commit all tasks to database
            db.commit()
            print(f"✅ Successfully created {images_processed} tasks for agent {agent_id}")
    
    except Exception as e:
        print(f"❌ Error processing ZIP file: {e}")
        db.rollback()
        raise Exception(f"Error processing ZIP file: {str(e)}")
    
    finally:
        # Clean up the ZIP file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🧹 Cleaned up ZIP file: {file_path}")
    
    return {
        "status": "success",
        "images_processed": images_processed,
        "message": f"Successfully processed {images_processed} images and assigned tasks to agent {agent_id}"
    }

# ===================== CLEANUP AND MAINTENANCE =====================

@app.on_event("startup")
async def startup_event():
    """Initialize cleanup tasks on startup"""
    print("🚀 Starting periodic cleanup task...")
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

# ===================== DEBUG AND STATUS ENDPOINTS =====================

@app.get("/debug")
def debug_info():
    """Debug endpoint to check system status"""
    import sys
    return {
        "files": os.listdir("."),
        "python_path": sys.path,
        "modules": list(sys.modules.keys()),
        "database_ready": database_ready,
        "routes_ready": routes_ready,
        "port": os.environ.get("PORT", "not set"),
        "upload_sessions": len(upload_sessions),
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
        "active_uploads": len(upload_sessions)
    }

# Upload sessions management (for debugging)
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    print(f"📁 Chunk upload directory: {CHUNK_UPLOAD_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)
