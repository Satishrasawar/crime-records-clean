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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

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
    from app.database import Base, engine, get_db
    from app.models import Agent, TaskProgress, SubmittedForm, AgentSession, Admin, ImageAssignment
    
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
        "http://127.0.0.1:8000",
        "https://web-production-b3ef2.up.railway.app"
    ]

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
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

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

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

# ===================== ENHANCED HEALTH CHECK =====================
@app.get("/health")
@limiter.limit("100/minute")
async def health_check(request: Request):
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
            db_gen = db_dependency()
            if hasattr(db_gen, '__next__'):
                db = next(db_gen)
            else:
                db = db_gen
            
            try:
                # Simple test for database connectivity
                if hasattr(db, 'execute'):
                    from sqlalchemy import text
                    result = db.execute(text("SELECT 1")).scalar()
                    if result == 1:
                        health_status["database"] = "connected"
                    else:
                        health_status["database"] = "query_failed"
                        health_status["status"] = "degraded"
                else:
                    health_status["database"] = "mock_mode"
                    health_status["status"] = "degraded"
            except Exception as query_error:
                health_status["database"] = f"query_error: {str(query_error)[:50]}"
                health_status["status"] = "degraded"
            finally:
                if hasattr(db, 'close'):
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

# Add simple health endpoints for Railway
@app.get("/healthz")
@limiter.limit("100/minute")
async def railway_health(request: Request):
    """Simple health check for Railway"""
    return {"status": "ok"}

@app.get("/ping")
@limiter.limit("100/minute")
async def ping(request: Request):
    """Minimal ping"""
    return "pong"

# Enhanced root endpoint
@app.get("/")
@limiter.limit("100/minute")
async def root(request: Request):
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

# ===================== ENHANCED STATIC FILE SERVING =====================
@app.get("/admin")
@limiter.limit("50/minute")
async def serve_admin_panel_redirect(request: Request):
    """Redirect /admin to /admin.html"""
    return FileResponse("admin.html") if os.path.exists("admin.html") else JSONResponse({"error": "Admin panel not found"}, status_code=404)

@app.get("/admin.html")
@limiter.limit("50/minute")
async def serve_admin_panel(request: Request):
    """Serve admin dashboard"""
    try:
        if os.path.exists("admin.html"):
            return FileResponse("admin.html", headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Type": "text/html"
            })
        # If admin.html doesn't exist, create a basic one
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
            
            <div class="api-link">
                <h3>📋 Preview Data</h3>
                <a href="/api/admin/preview-data" target="_blank">/api/admin/preview-data</a>
                <p>Preview submitted form data</p>
            </div>
        </div>

        <div style="margin-top: 40px; padding: 20px; background: #e9ecef; border-radius: 5px;">
            <h3>🚀 Agent Registration</h3>
            <p>Register new agents via POST to: <code>/api/agents/register</code></p>
            <p>Upload tasks via POST to: <code>/api/admin/upload-tasks</code></p>
        </div>
    </div>

    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
    </script>
</body>
</html>"""
        
        # Create admin.html file if it doesn't exist
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
@limiter.limit("50/minute")
async def serve_agent_panel_redirect(request: Request):
    """Redirect /agent to /agent.html"""
    return FileResponse("agent.html") if os.path.exists("agent.html") else JSONResponse({"error": "Agent panel not found"}, status_code=404)

@app.get("/agent.html") 
@limiter.limit("50/minute")
async def serve_agent_panel(request: Request):
    """Serve agent interface"""
    try:
        if os.path.exists("agent.html"):
            return FileResponse("agent.html", headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache", 
                "Expires": "0",
                "Content-Type": "text/html"
            })
            
        # If agent.html doesn't exist, create a basic one
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

        <div style="margin-top: 30px; padding: 20px; background: #e9ecef; border-radius: 5px;">
            <h3>📋 Quick Links</h3>
            <p><strong>Get Current Task:</strong> <code>GET /api/agents/{agent_id}/current-task</code></p>
            <p><strong>Submit Task:</strong> <code>POST /api/agents/{agent_id}/submit</code></p>
            <p><strong>View Statistics:</strong> <code>GET /api/agents/{agent_id}/statistics</code></p>
        </div>
    </div>

    <script>
        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const agentId = document.getElementById('agentId').value;
            const password = document.getElementById('password').value;
            
            if (agentId && password) {
                // Redirect to current task API endpoint for now
                window.location.href = `/api/agents/${agentId}/current-task`;
            } else {
                alert('Please enter both Agent ID and Password');
            }
        });
    </script>
</body>
</html>"""
        
        # Create agent.html file if it doesn't exist
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

# ===================== ENHANCED DEBUG ENDPOINTS =====================
@app.get("/debug")
@limiter.limit("50/minute")
async def debug_info(request: Request):
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
@limiter.limit("50/minute")
async def system_status(request: Request):
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
@limiter.limit("50/minute")
async def get_admin_statistics(db = Depends(db_dependency), request: Request):
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
@limiter.limit("50/minute")
async def list_agents(db = Depends(db_dependency), request: Request):
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
@app.get("/api/agents/{agent_id}/current-task")
@limiter.limit("50/minute")
async def get_current_task(agent_id: str, db = Depends(db_dependency), request: Request):
    """Get current task for an agent - FIXED VERSION"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        # Validate agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
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
        
        # If no tasks available, return completion status
        if not current_task:
            total_tasks = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).count()
            completed_tasks = db.query(TaskProgress).filter(
                TaskProgress.agent_id == agent_id,
                TaskProgress.status == 'completed'
            ).count()
            
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
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting current task for {agent_id}: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Error getting current task: {str(e)}")

@app.post("/api/agents/{agent_id}/submit")
@limiter.limit("50/minute")
async def submit_task_form(
    agent_id: str,
    request: Request,
    db = Depends(db_dependency)
):
    """Submit completed task form - FIXED VERSION"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        # Validate agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Parse form data - handle both JSON and form submissions
        try:
            content_type = request.headers.get("content-type", "")
            
            if content_type.startswith("application/json"):
                form_data = await request.json()
            else:
                # Handle multipart form data
                form = await request.form()
                form_data = {}
                for key, value in form.items():
                    if key not in ['agent_id', 'task_id']:  # Skip metadata fields
                        form_data[key] = value
            
            print(f"📝 Received form data for {agent_id}: {list(form_data.keys())}")
            
        except Exception as parse_error:
            print(f"❌ Error parsing form data: {parse_error}")
            raise HTTPException(status_code=400, detail="Invalid form data format")
        
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
            raise HTTPException(
                status_code=404, 
                detail="No active task found for submission. Please refresh and try again."
            )
        
        # Create submission record
        try:
            submitted_form = SubmittedForm(
                agent_id=agent_id,
                task_id=current_task.id,
                image_filename=current_task.image_filename,
                form_data=form_data,
                submitted_at=datetime.utcnow()
            )
            
            db.add(submitted_form)
            
            # Mark current task as completed
            current_task.status = 'completed'
            current_task.completed_at = datetime.utcnow()
            
            # Commit both changes
            db.commit()
            db.refresh(submitted_form)
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
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting task for {agent_id}: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")

# ===================== ADDITIONAL HELPER ENDPOINTS =====================
@app.get("/api/agents/{agent_id}/next-task")
@limiter.limit("50/minute")
async def get_next_task(agent_id: str, db = Depends(db_dependency), request: Request):
    """Get next available task - Alternative endpoint"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        # This endpoint just redirects to current-task for consistency
        return await get_current_task(agent_id, db)
        
    except Exception as e:
        print(f"❌ Error getting next task for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting next task: {str(e)}")

@app.post("/api/agents/{agent_id}/skip-task")
@limiter.limit("50/minute")
async def skip_current_task(agent_id: str, db = Depends(db_dependency), request: Request):
    """Skip current task (mark as skipped) - Optional functionality"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
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

@app.get("/api/agents/{agent_id}/progress")
@limiter.limit("50/minute")
async def get_agent_progress(agent_id: str, db = Depends(db_dependency), request: Request):
    """Get detailed progress information for an agent"""
    try:
        if not database_ready:
            return {
                "total_tasks": 0,
                "completed_tasks": 0,
                "pending_tasks": 0,
                "in_progress_tasks": 0,
                "skipped_tasks": 0,
                "completion_percentage": 0
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

# ===================== STANDARD UPLOAD ENDPOINT =====================
@app.post("/api/admin/upload-tasks")
@limiter.limit("10/minute")
async def upload_tasks_standard(
    zip_file: UploadFile = File(...),
    agent_id: str = Form(...),
    db = Depends(db_dependency),
    request: Request
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
@limiter.limit("10/minute")
async def init_chunked_upload(
    filename: str = Form(...),
    filesize: int = Form(...),
    total_chunks: int = Form(...),
    agent_id: str = Form(...),
    request: Request
):
    """Initialize a chunked upload session for large files"""
    try:
        # Validate agent exists (if database is ready)
        if database_ready:
            db_gen = db_dependency()
            if hasattr(db_gen, '__next__'):
                db = next(db_gen)
            else:
                db = db_gen
                
            try:
                agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
                if not agent:
                    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
                if agent.status != "active":
                    raise HTTPException(status_code=400, detail=f"Agent {agent_id} is not active")
            finally:
                if hasattr(db, 'close'):
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
@limiter.limit("50/minute")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
    request: Request
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
@limiter.limit("10/minute")
async def finalize_chunked_upload(upload_id: str = Form(...), db = Depends(db_dependency), request: Request):
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

# ===================== UPLOAD SESSIONS MANAGEMENT =====================
@app.get("/api/admin/upload-sessions")
@limiter.limit("50/minute")
async def get_upload_sessions(request: Request):
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
@limiter.limit("10/minute")
async def reset_agent_password(agent_id: str, db = Depends(db_dependency), request: Request):
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
@limiter.limit("50/minute")
async def get_agent_password(agent_id: str, db = Depends(db_dependency), request: Request):
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
@limiter.limit("10/minute")
async def update_agent_status(agent_id: str, status_data: dict, db = Depends(db_dependency), request: Request):
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
@limiter.limit("10/minute")
async def force_logout_agent(agent_id: str, db = Depends(db_dependency), request: Request):
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
@limiter.limit("50/minute")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db = Depends(db_dependency),
    request: Request
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
@limiter.limit("50/minute")
async def test_data_availability(db = Depends(db_dependency), request: Request):
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
@limiter.limit("50/minute")
async def get_session_report(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db = Depends(db_dependency),
    request: Request
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
@limiter.limit("10/minute")
async def export_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db = Depends(db_dependency),
    request: Request
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
@limiter.limit("10/minute")
async def export_sessions(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db = Depends(db_dependency),
    request: Request
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
    # Railway requires binding to 0.0.0.0 and the PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)
