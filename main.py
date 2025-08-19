import os
import sys
import uuid
import shutil
import zipfile
import asyncio
import aiofiles
import secrets
import string
import re
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv  # Add this import

load_dotenv()  # Load .env variables

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
    from app.models import Agent, TaskProgress, SubmittedForm, AgentSession, Admin
    
    print("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
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
        "https://web-production-b3ef2.up.railway.app",
        "http://localhost:5173"  # Added for Vite dev server
    ]

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

def create_default_admin():
    """Create default admin user with proper error handling"""
    try:
        print("🔧 Setting up admin user...")
        
        if not database_ready:
            print("⚠️ Database not ready, skipping admin creation")
            return
        
        from app.models import Admin
        from app.security import hash_password
        
        db_gen = db_dependency()
        if hasattr(db_gen, '__next__'):
            db = next(db_gen)
        else:
            db = db_gen
        
        try:
            existing_admin = db.query(Admin).filter(Admin.username == "admin").first()
            
            if existing_admin:
                print(f"👤 Found existing admin: {existing_admin.username}")
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
    print("🚀 Starting Agent Task System...")
    print(f"🌍 Domain: {os.environ.get('DOMAIN', 'railway')}")
    print(f"🔗 Allowed Origins: {len(ALLOWED_ORIGINS)} configured")
    
    create_default_admin()
    
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    print("🛑 Shutting down application...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("✅ Application shutdown complete")

app = FastAPI(
    title="Client Records Data Entry System",
    version="2.0.0",
    description="Enhanced system for agent-task-system.com with chunked upload support and custom domain",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

@app.middleware("http")
async def enhanced_request_middleware(request, call_next):
    """Enhanced middleware for domain detection, logging, and security"""
    host = request.headers.get("host", "unknown")
    origin = request.headers.get("origin", "unknown")
    
    if not request.url.path.startswith("/health") and not host.startswith(("127.0.0.1", "localhost")):
        print(f"🌍 Request - Host: {host}, Origin: {origin}, Path: {request.url.path}")
    
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    return response

try:
    print("📦 Importing agent routes...")
    from agent_routes import router as agent_router
    app.include_router(agent_router)
    print("✅ Agent routes included successfully!")
    routes_ready = True
except Exception as e:
    print(f"❌ Agent routes failed: {e}")
    routes_ready = False

try:
    os.makedirs("static/task_images", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Static files configured")
except Exception as e:
    print(f"❌ Static files setup failed: {e}")

async def periodic_cleanup():
    """Clean up old upload sessions every hour"""
    while True:
        try:
            now = datetime.now()
            expired_sessions = []
            
            for upload_id, session in upload_sessions.items():
                if (now - session["created_at"]).total_seconds() > 7200:
                    expired_sessions.append(upload_id)
            
            for upload_id in expired_sessions:
                print(f"🧹 Cleaning up expired upload session: {upload_id}")
                if upload_id in upload_sessions:
                    del upload_sessions[upload_id]
                
            if expired_sessions:
                print(f"🧹 Cleaned up {len(expired_sessions)} expired upload sessions")
                
        except Exception as e:
            print(f"❌ Error in periodic cleanup: {e}")
        
        await asyncio.sleep(3600)

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
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Agent Task System - Admin Panel</h1>
        <p>Admin panel is running. Please use the full admin.html file for complete functionality.</p>
    </div>
</body>
</html>"""
        
        with open("admin.html", "w") as f:
            f.write(basic_admin_html)
            
        return FileResponse("admin.html")
        
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 Agent Portal - Task Management System</h1>
        <p>Agent portal is running. Please use the full agent.html file for complete functionality.</p>
    </div>
</body>
</html>"""
        
        with open("agent.html", "w") as f:
            f.write(basic_agent_html)
            
        return FileResponse("agent.html")
        
    except Exception as e:
        return JSONResponse({"error": f"Could not serve agent panel: {e}"}, status_code=500)

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
        "active_uploads": len(upload_sessions),
        "cors_origins": len(ALLOWED_ORIGINS)
    }

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
            "agent_registration",
            "task_management",
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
    print(f"💾 Database ready: {database_ready}")
    print(f"🛣️ Routes ready: {routes_ready}")
    print(f"🏃 Starting server on port {port}")
    print("=" * 60)
    print("🔐 ADMIN CREDENTIALS:")
    print("Username: admin")
    print("Password: admin123")
    print("📱 Access Points:")
    print(f"- Admin Panel: http://localhost:{port}/admin.html")
    print(f"- Agent Panel: http://localhost:{port}/agent.html")
    print(f"- Health Check: http://localhost:{port}/health")
    print(f"- Registration Test: http://localhost:{port}/api/agents/test-registration")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
