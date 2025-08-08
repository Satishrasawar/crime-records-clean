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

def create_default_admin():
    """Create default admin user with proper error handling"""
    try:
        print("🔧 Setting up admin user...")
        
        if not database_ready:
            print("⚠️ Database not ready, skipping admin creation")
            return
        
        # Import after database is ready
        from app.models import Admin
        from app.security import hash_password
        
        # Get database session
        db_gen = db_dependency()
        if hasattr(db_gen, '__next__'):
            db = next(db_gen)
        else:
            db = db_gen
        
        try:
            # Check existing admin
            existing_admin = db.query(Admin).filter(Admin.username == "admin").first()
            
            if existing_admin:
                print(f"👤 Found existing admin: {existing_admin.username}")
                # Always reset password for testing
                existing_admin.hashed_password = hash_password("admin123")
                existing_admin.is_active = True
                existing_admin.email = "admin@agent-task-system.com"
                db.commit()
                print("🔄 Updated existing admin password")
            else:
                print("🔧 Creating new admin user...")
                hashed_password = hash_password("admin123")
                
                new_admin = Admin(
                    username="admin",
                    hashed_password=hashed_password,
                    email="admin@agent-task-system.com",
                    is_active=True,
                    created_at=datetime.now()
                )
                
                db.add(new_admin)
                db.commit()
                db.refresh(new_admin)
                print("✅ Created new admin user")
            
            print("=" * 50)
            print("🔐 ADMIN LOGIN CREDENTIALS:")
            print("Username: admin")
            print("Password: admin123")
            print("=" * 50)
            print("🌍 Access at:")
            print("- Admin Panel: /admin.html")
            print("- Status Check: /api/admin/status")
            print("- Simple Login: /api/admin/simple-login")
            print("=" * 50)
            
        except Exception as db_error:
            print(f"❌ Database error: {db_error}")
            if hasattr(db, 'rollback'):
                db.rollback()
        
        finally:
            if hasattr(db, 'close'):
                db.close()
    
    except Exception as e:
        print(f"❌ Admin setup completely failed: {e}")
        import traceback
        traceback.print_exc()

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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
    print("🚀 Starting periodic cleanup task...")
    print(f"🌍 Domain: {os.environ.get('DOMAIN', 'railway')}")
    print(f"🔗 Allowed Origins: {len(ALLOWED_ORIGINS)} configured")
    
    # Create default admin user
    create_default_admin()
    
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

# ===================== ADMIN DEBUG ENDPOINTS =====================
@app.post("/api/admin/create-admin")
@limiter.limit("1/minute")
async def create_admin_user_endpoint(request: Request, db=Depends(db_dependency)):
    """Create admin user - for testing only"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        from app.models import Admin
        from app.security import hash_password
        
        # Check if admin already exists
        existing_admin = db.query(Admin).filter(Admin.username == "admin").first()
        if existing_admin:
            return {
                "message": "Admin already exists",
                "username": "admin",
                "status": "active" if existing_admin.is_active else "inactive"
            }
        
        # Create new admin
        hashed_password = hash_password("admin123")
        new_admin = Admin(
            username="admin",
            hashed_password=hashed_password,
            email="admin@agent-task-system.com",
            is_active=True,
            created_at=datetime.now()
        )
        
        db.add(new_admin)
        db.commit()
        
        return {
            "success": True,
            "message": "Admin user created successfully!",
            "credentials": {
                "username": "admin",
                "password": "admin123"
            },
            "login_url": "/admin.html"
        }
        
    except Exception as e:
        if hasattr(db, 'rollback'):
            db.rollback()
        print(f"❌ Error creating admin: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create admin: {str(e)}")

@app.post("/api/admin/simple-login")
@limiter.limit("10/minute")
async def admin_simple_login(request: Request, db=Depends(db_dependency)):
    """Simplified admin login endpoint"""
    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        
        print(f"🔐 Admin login attempt: {username}")
        
        if not username or not password:
            return {"success": False, "message": "Username and password required"}
        
        # Check hardcoded credentials first
        if username == "admin" and password == "admin123":
            print("✅ Hardcoded admin login successful")
            return {
                "success": True,
                "message": "Login successful",
                "access_token": "admin_token_" + str(int(datetime.now().timestamp())),
                "user": {
                    "username": username,
                    "role": "admin"
                }
            }
        
        # Try database validation if available
        if database_ready:
            try:
                from app.models import Admin
                from app.security import verify_password
                
                admin = db.query(Admin).filter(Admin.username == username).first()
                if admin and admin.is_active:
                    if verify_password(password, admin.hashed_password):
                        print("✅ Database admin login successful")
                        return {
                            "success": True,
                            "message": "Login successful",
                            "access_token": "admin_token_" + str(int(datetime.now().timestamp())),
                            "user": {
                                "username": username,
                                "email": admin.email,
                                "role": "admin"
                            }
                        }
            except Exception as db_error:
                print(f"⚠️ Database login failed, fallback to hardcoded: {db_error}")
        
        print("❌ Admin login failed")
        return {"success": False, "message": "Invalid credentials"}
        
    except Exception as e:
        print(f"❌ Login error: {e}")
        return {"success": False, "message": "Login error occurred"}

@app.get("/api/admin/check-admin")
@limiter.limit("10/minute")
async def check_admin_status(request: Request, db=Depends(db_dependency)):
    """Check admin user status"""
    try:
        if not database_ready:
            return {"database": "not_ready"}
        
        from app.models import Admin
        
        admin = db.query(Admin).filter(Admin.username == "admin").first()
        if not admin:
            return {
                "admin_exists": False,
                "message": "No admin user found. Use /api/admin/create-admin to create one."
            }
        
        return {
            "admin_exists": True,
            "username": admin.username,
            "email": admin.email,
            "is_active": admin.is_active,
            "created_at": admin.created_at.isoformat() if admin.created_at else None,
            "message": "Admin user found. Use credentials: admin / admin123"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/admin/test-login")
@limiter.limit("5/minute")
async def test_admin_login(request: Request, db=Depends(db_dependency)):
    """Test admin login without JWT - for debugging"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        from app.models import Admin
        from app.security import verify_password
        
        admin = db.query(Admin).filter(Admin.username == username).first()
        if not admin:
            return {"success": False, "message": "Admin user not found"}
        
        if not admin.is_active:
            return {"success": False, "message": "Admin user is not active"}
        
        password_valid = verify_password(password, admin.hashed_password)
        if not password_valid:
            return {"success": False, "message": "Invalid password"}
        
        return {
            "success": True,
            "message": "Login test successful!",
            "admin_info": {
                "username": admin.username,
                "email": admin.email,
                "is_active": admin.is_active
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/admin/reset-admin-password")
@limiter.limit("1/minute")
async def reset_admin_password_endpoint(request: Request, db=Depends(db_dependency)):
    """Reset admin password - for testing only"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        from app.models import Admin
        from app.security import hash_password
        
        admin = db.query(Admin).filter(Admin.username == "admin").first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin user not found")
        
        # Reset password
        new_password = "admin123"
        admin.hashed_password = hash_password(new_password)
        admin.is_active = True  # Make sure admin is active
        db.commit()
        
        return {
            "success": True,
            "message": "Admin password reset successfully!",
            "credentials": {
                "username": "admin",
                "password": new_password
            }
        }
        
    except Exception as e:
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to reset password: {str(e)}")

# ===================== ENHANCED HEALTH CHECK =====================
@app.get("/health")
@limiter.limit("100/minute")
async def health_check(request: Request, db=Depends(db_dependency)):
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
        .api-link { background: #f8f9fa; padding: 20px; border-radius: 5px; border какое left: 4px solid #007bff; }
        .api-link h3 { margin: 0 0 10px 0; color: #007bff; }
        .api-link a { color: #007bff; text-decoration: none; font-family: monospace; }
        .api-link a:hover { text-decoration: underline; }
        .debug-section { background: #fff3cd; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffc107; }
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

        <div class="debug-section">
            <h3>🔧 Admin Debug Tools</h3>
            <p><strong>Check Admin Status:</strong> <a href="/api/admin/check-admin" target="_blank">/api/admin/check-admin</a></p>
            <p><strong>Create Admin:</strong> <code>POST /api/admin/create-admin</code></p>
            <p><strong>Test Login:</strong> <code>POST /api/admin/test-login</code> with {"username": "admin", "password": "admin123"}
            <p><strong>Reset Password:</strong> <code>POST /api/admin/reset-admin-password</code></p>
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

# ===================== AGENT REGISTRATION ENDPOINT =====================
@app.post("/api/agents/register")
@limiter.limit("10/minute")
async def register_new_agent(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(...),
    country: str = Form(...),
    gender: str = Form(...),
    db=Depends(db_dependency)
):
    """Register a new agent with auto-generated credentials"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        import secrets
        import string
        from datetime import datetime
        
        # Validate required fields
        if not all([name, email, mobile, dob, country, gender]):
            raise HTTPException(status_code=400, detail="All fields are required")
        
        # Validate email format
        import re
        email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_pattern, email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Validate mobile format (basic validation)
        mobile_pattern = r'^\+?\d{10,15}$'
        clean_mobile = mobile.replace(' ', '').replace('-', '')
        if not re.match(mobile_pattern, clean_mobile):
            raise HTTPException(status_code=400, detail="Invalid mobile number format")
        
        # Check if agent with same email already exists
        existing_agent = db.query(Agent).filter(Agent.email == email).first()
        if existing_agent:
            raise HTTPException(status_code=409, detail="Agent with this email already exists")
        
        # Generate unique agent ID
        def generate_agent_id():
            """Generate unique agent ID in format AGT followed by 6 digits"""
            while True:
                # Generate 6-digit random number
撒                agent_number = secrets.randbelow(900000) + 100000  # Ensures 6 digits
                agent_id = f"AGT{agent_number}"
                
                # Check if ID already exists
                existing = db.query(Agent).filter(Agent.agent_id == agent_id).first()
                if not existing:
                    return agent_id
        
        # Generate secure password
        def generate_password():
            """Generate secure password with letters, numbers, and special characters"""
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            password = ''.join(secrets.choice(alphabet) for _ in range(12))
            
            # Ensure password has at least one letter, number, and special char
            if (any(c.isalpha() for c in password) and 
                any(c.isdigit() for c in password) and 
                any(c in "!@#$%^&*" for c in password)):
                return password
            else:
                return generate_password()  # Recursively generate until valid
        
        # Generate credentials
        agent_id = generate_agent_id()
        password = generate_password()
        
        # Parse date of birth
        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
            dob_str = dob  # Keep as string for compatibility
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Create new agent
        new_agent = Agent(
            agent_id=agent_id,
            name=name.strip(),
            email=email.strip().lower(),
            mobile=clean_mobile,
            dob=dob_str,  # Store as string
            country=country.strip(),
            gender=gender,
            password=password,  # Store plain password for now
            status="active",
            created_at=datetime.now()
        )
        
        # Save to database
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        print(f"✅ New agent registered: {agent_id} - {name}")
        
        # Return success response with credentials
        return {
            "success": True,
            "message": "Agent registered successfully!",
            "agent_id": agent_id,
            "password": password,
            "agent_details": {
                "name": name,
                "email": email,
                "mobile": clean_mobile,
                "status": "active",
                "created_at": new_agent.created_at.isoformat()
            }
        }
        
    except HTTPException:
        if hasattr(db, 'rollback'):
            db.rollback()
        raise
    except Exception as e:
        if hasattr(db, 'rollback'):
            db.rollback()
        print(f"❌ Error registering agent: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# ===================== TASK ENDPOINTS FOR AGENTS =====================
@app.get("/api/agents/{agent_id}/current-task")
@limiter.limit("50/minute")
async def get_current_task(agent_id: str, request: Request, db=Depends(db_dependency)):
    """Get current task for an agent"""
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
async def submit_task_form(agent_id: str, request: Request, db=Depends(db_dependency)):
    """Submit completed task form"""
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
async def get_next_task(agent_id: str, request: Request, db=Depends(db_dependency)):
    """Get next available task - Alternative endpoint"""
    try:
        if not database_ready:
            raise HTTPException(status_code=503, detail="Database not ready")
        
        # This endpoint just redirects to current-task for consistency
        return await get_current_task(agent_id, request, db)
        
    except Exception as e:
        print(f"❌ Error getting next task for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting next task: {str(e)}")

@app.post("/api/agents/{agent_id}/skip-task")
@limiter.limit("50/minute")
async def skip_current_task(agent_id: str, request: Request, db=Depends(db_dependency)):
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
async def get_agent_progress(agent_id: str, request: Request, db=Depends(db_dependency)):
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
                    <input type="text" id="agentId" name="agentId" placeholder="Enter your Agent ID (e.g., AGT123456)" required>
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
            <p><strong>View Progress:</strong> <code>GET /api/agents/{agent_id}/progress</code></p>
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
async def get_admin_statistics(request: Request, db=Depends(db_dependency)):
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
async def list_agents(request: Request, db=Depends(db_dependency)):
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
@app.get("/api/admin/preview-data")
@limiter.limit("50/minute")
async def preview_data(
    request: Request,
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db=Depends(db_dependency)
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
