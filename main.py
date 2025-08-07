# THIS IS ALMOST YOUR COMPLETE ORIGINAL CODE
# Adding the HTML serving endpoints that might be causing the issue

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
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Chunked upload configuration
CHUNK_UPLOAD_DIR = "temp_chunks"
os.makedirs(CHUNK_UPLOAD_DIR, exist_ok=True)

# In-memory storage for upload sessions
upload_sessions: Dict[str, Dict[str, Any]] = {}

# Global variables to track system state
database_ready = False
routes_ready = False

# Mock database classes for when database is not available
class MockDB:
    def query(self, *args, **kwargs):
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
    """Mock database dependency when database is not available"""
    return MockDB()

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
    db_dependency = get_db
    
except Exception as e:
    print(f"❌ Database setup failed: {e}")
    import traceback
    traceback.print_exc()
    database_ready = False
    db_dependency = get_mock_db

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

# ===================== CLEANUP FUNCTIONS =====================
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

# ===================== ZIP PROCESSING FUNCTION =====================
async def process_uploaded_zip(file_path: str, agent_id: str, db):
    """ZIP file processing function"""
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
                if hasattr(db, 'rollback'):
                    db.rollback()
                raise Exception(f"Database error while saving tasks: {str(db_error)}")
    
    except Exception as e:
        print(f"❌ Error processing ZIP file: {e}")
        if hasattr(db, 'rollback'):
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

# Lifespan context manager WITH BACKGROUND TASKS
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
    # Startup
    print("🚀 Starting periodic cleanup task...")
    print(f"🌍 Domain: {os.environ.get('DOMAIN', 'railway')}")
    print(f"🔗 Allowed Origins: {len(ALLOWED_ORIGINS)} configured")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    print("🛑 Shutting down application...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("✅ Application shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0",
    description="Enhanced system for agent-task-system.com with chunked upload support and custom domain",
    lifespan=lifespan
)

# Enhanced CORS middleware with custom domain support
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
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

# TRY TO IMPORT AGENT ROUTES
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

# ===================== ENHANCED STATIC FILE SERVING - THE SUSPECT =====================

@app.get("/admin")
async def serve_admin_panel_redirect():
    """Redirect /admin to /admin.html - MIGHT CAUSE ISSUES"""
    return FileResponse("admin.html") if os.path.exists("admin.html") else JSONResponse({"error": "Admin panel not found"}, status_code=404)

@app.get("/admin.html")
async def serve_admin_panel():
    """Serve admin dashboard - VERY COMPLEX HTML GENERATION"""
    try:
        if os.path.exists("admin.html"):
            return FileResponse("admin.html", headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Type": "text/html"
            })
        # If admin.html doesn't exist, create a basic one - THIS MIGHT BE THE ISSUE
        basic_admin_html = """<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel - Agent Task System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .status { background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #28a745; }
        .api-links { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px; }
        .api-link { background: #f8f9fa; padding: 20px; border-radius: 5px; border-left: 4px solid #007bff; }
        .api-link h3 { margin: 0 0 10px 0; color: #007bff; }
        .api-link a { color: #007bff; text-decoration: none; font-family: monospace; }
        .api-link a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Agent Task System - Admin Panel</h1>
        
        <div class="status">
            <strong>✅ System Status:</strong> Online and Ready<br>
            <strong>🌍 Platform:</strong> Railway<br>
            <strong>⏰ Last Updated:</strong> <span id="timestamp"></span>
        </div>

        <div class="api-links">
            <div class="api-link">
                <h3>📊 System Health</h3>
                <a href="/health" target="_blank">/health</a>
                <p>Check system health and database status</p>
            </div>
            
            <div class="api-link">
                <h3>👥 Agents Management</h3>
                <a href="/api/agents" target="_blank">/api/agents</a>
                <p>View all registered agents and their statistics</p>
            </div>
            
            <div class="api-link">
                <h3>📈 Statistics</h3>
                <a href="/api/admin/statistics" target="_blank">/api/admin/statistics</a>
                <p>Get overall system statistics</p>
            </div>
            
            <div class="api-link">
                <h3>🔧 Debug Info</h3>
                <a href="/debug" target="_blank">/debug</a>
                <p>System debug information</p>
            </div>
            
            <div class="api-link">
                <h3>📤 Upload Sessions</h3>
                <a href="/api/admin/upload-sessions" target="_blank">/api/admin/upload-sessions</a>
                <p>View active file upload sessions</p>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
    </script>
</body>
</html>"""
        
        # Create admin.html file if it doesn't exist - FILE CREATION MIGHT BE THE ISSUE
        with open("admin.html", "w") as f:
            f.write(basic_admin_html)
            
        return FileResponse("admin.html", headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Content-Type": "text/html"
        })
        
    except Exception as e:
        return JSONResponse({"error": f"Could not serve admin panel: {e}"}, status_code=500)

@app.get("/agent")
async def serve_agent_panel_redirect():
    """Redirect /agent to /agent.html"""
    return FileResponse("agent.html") if os.path.exists("agent.html") else JSONResponse({"error": "Agent panel not found"}, status_code=404)

@app.get("/agent.html") 
async def serve_agent_panel():
    """Serve agent interface - ANOTHER COMPLEX HTML GENERATOR"""
    try:
        if os.path.exists("agent.html"):
            return FileResponse("agent.html", headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache", 
                "Expires": "0",
                "Content-Type": "text/html"
            })
            
        # If agent.html doesn't exist, create a basic one - THIS MIGHT BE THE ISSUE
        basic_agent_html = """<!DOCTYPE html>
<html>
<head>
    <title>Agent Portal - Agent Task System</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #28a745; padding-bottom: 10px; }
        .login-form { background: #f8f9fa; padding: 30px; border-radius: 8px; margin: 20px 0; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #333; }
        .form-group input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 4px; font-size: 16px; }
        .form-group input:focus { border-color: #28a745; outline: none; }
        .btn { background: #28a745; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #218838; }
        .info-box { background: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #17a2b8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 Agent Portal - Task Management System</h1>
        
        <div class="info-box">
            <strong>Welcome to the Agent Portal!</strong><br>
            Enter your Agent ID and Password to access your assigned tasks.
        </div>

        <div class="login-form">
            <h3>Agent Login</h3>
            <form id="loginForm">
                <div class="form-group">
                    <label for="agentId">Agent ID:</label>
                    <input type="text" id="agentId" name="agentId" placeholder="Enter your Agent ID (e.g., AG20240101ABCD)" required>
                </div>
                
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" placeholder="Enter your password" required>
                </div>
                
                <button type="submit" class="btn">🚀 Login & View Tasks</button>
            </form>
        </div>
    </div>

    <script>
        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const agentId = document.getElementById('agentId').value;
            const password = document.getElementById('password').value;
            
            if (agentId && password) {
                // Redirect to current task API endpoint for now
                window.location.href = `/api/agents/${agentId}/tasks/current`;
            } else {
                alert('Please enter both Agent ID and Password');
            }
        });
    </script>
</body>
</html>"""
        
        # Create agent.html file if it doesn't exist - FILE CREATION MIGHT BE THE ISSUE
        with open("agent.html", "w") as f:
            f.write(basic_agent_html)
            
        return FileResponse("agent.html", headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Content-Type": "text/html"
        })
        
    except Exception as e:
        return JSONResponse({"error": f"Could not serve agent panel: {e}"}, status_code=500)

# ===================== ALL OTHER ENDPOINTS (REST OF YOUR CODE) =====================

@app.get("/health")
def health_check():
    """Enhanced health check"""
    return {
        "status": "healthy",
        "platform": "Railway",
        "message": "Service is running",
        "timestamp": datetime.now().isoformat(),
        "database": "ready" if database_ready else "not_ready",
        "routes": "ready" if routes_ready else "failed",
        "active_uploads": len(upload_sessions),
        "version": "2.0.0"
    }

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
        "routes_ready": routes_ready,
        "active_uploads": len(upload_sessions),
        "features": [
            "chunked_upload", 
            "large_file_support", 
            "custom_domain_support",
            "ssl_enabled",
            "enhanced_security"
        ]
    }

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
