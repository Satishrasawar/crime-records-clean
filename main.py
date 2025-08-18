# main.py - Part 1: Imports and Setup
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
    from app.models import Agent, TaskProgress, SubmittedForm, AgentSession, Admin
    
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
# main.py - Part 2: Admin Setup and App Initialization

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
                if upload_id in upload_sessions:
                    del upload_sessions[upload_id]
                
            if expired_sessions:
                print(f"🧹 Cleaned up {len(expired_sessions)} expired upload sessions")
                
        except Exception as e:
            print(f"❌ Error in periodic cleanup: {e}")
        
        # Wait 1 hour before next cleanup
        await asyncio.sleep(3600)
        # main.py - Part 3: FIXED Agent Registration Endpoint

# ===================== HELPER FUNCTIONS =====================
def generate_unique_agent_id(db):
    """Generate a unique agent ID"""
    max_attempts = 20
    for attempt in range(max_attempts):
        # Generate 6-digit number (100000-999999)
        agent_number = secrets.randbelow(900000) + 100000
        agent_id = f"AGT{agent_number}"
        
        try:
            # Check if ID exists
            existing = db.query(Agent).filter(Agent.agent_id == agent_id).first()
            if not existing:
                print(f"✅ Generated unique agent ID: {agent_id}")
                return agent_id
        except Exception as check_error:
            print(f"⚠️ ID uniqueness check error: {check_error}")
            continue
    
    # Fallback: use timestamp
    import time
    fallback_id = f"AGT{str(int(time.time()))[-6:]}"
    print(f"⚠️ Using fallback agent ID: {fallback_id}")
    return fallback_id

def generate_secure_password():
    """Generate a secure password"""
    # Include letters, numbers, and safe special characters
    characters = string.ascii_letters + string.digits + "!@#$%"
    
    # Ensure password has variety
    password_parts = [
        secrets.choice(string.ascii_uppercase),  # At least one uppercase
        secrets.choice(string.ascii_lowercase),  # At least one lowercase
        secrets.choice(string.digits),           # At least one digit
        secrets.choice("!@#$%")                  # At least one special char
    ]
    
    # Fill remaining 8 characters randomly
    for _ in range(8):
        password_parts.append(secrets.choice(characters))
    
    # Shuffle to avoid predictable patterns
    secrets.SystemRandom().shuffle(password_parts)
    return ''.join(password_parts)

def validate_agent_data(name, email, mobile, dob, country, gender):
    """Validate agent registration data"""
    errors = []
    
    # Validate required fields
    if not name or not name.strip():
        errors.append("Name is required")
    
    if not email or not email.strip():
        errors.append("Email is required")
    
    if not mobile or not mobile.strip():
        errors.append("Mobile number is required")
    
    if not dob:
        errors.append("Date of birth is required")
    
    if not country or not country.strip():
        errors.append("Country is required")
    
    if not gender or not gender.strip():
        errors.append("Gender is required")
    
    # Email format validation
    if email:
        email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_pattern, email.strip()):
            errors.append("Please enter a valid email address")
    
    # Mobile number validation
    if mobile:
        mobile_clean = re.sub(r'[\s\-\(\)\.]+', '', mobile.strip())
        if not re.match(r'^\+?\d{10,15}$', mobile_clean):
            errors.append("Mobile number must contain 10-15 digits")
    
    # Date of birth validation
    if dob:
        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d')
            age = (datetime.now() - dob_date).days // 365
            if age < 16:
                errors.append("Agent must be at least 16 years old")
            if age > 100:
                errors.append("Please enter a valid date of birth")
        except ValueError:
            errors.append("Invalid date format. Please use YYYY-MM-DD")
    
    # Gender validation
    if gender:
        valid_genders = ['Male', 'Female', 'Other']
        if gender not in valid_genders:
            errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
    
    return errors

# ===================== MAIN AGENT REGISTRATION ENDPOINT =====================

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
    """
    FIXED Agent Registration Endpoint
    This version handles all validation and error cases properly
    """
    
    print(f"🔄 Agent registration attempt - Name: '{name}', Email: '{email}'")
    
    try:
        # ===== DATABASE READINESS CHECK =====
        if not database_ready:
            print("❌ Database not ready")
            raise HTTPException(
                status_code=503, 
                detail="Database service is temporarily unavailable. Please try again in a few moments."
            )
        
        # ===== DATA VALIDATION =====
        print("📝 Validating agent data...")
        
        # Clean and prepare data
        name_clean = name.strip() if name else ''
        email_clean = email.strip().lower() if email else ''
        mobile_clean = re.sub(r'[\s\-\(\)\.]+', '', mobile.strip()) if mobile else ''
        country_clean = country.strip() if country else ''
        gender_clean = gender.strip() if gender else ''
        
        # Validate all fields
        validation_errors = validate_agent_data(name_clean, email_clean, mobile_clean, dob, country_clean, gender_clean)
        
        if validation_errors:
            print(f"❌ Validation errors: {validation_errors}")
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed: {'; '.join(validation_errors)}"
            )
        
        print("✅ Data validation passed")
        
        # ===== DATABASE CHECKS =====
        try:
            print("🔍 Checking for existing agent with same email...")
            existing_agent = db.query(Agent).filter(Agent.email == email_clean).first()
            
            if existing_agent:
                print(f"❌ Email already exists: {email_clean}")
                raise HTTPException(
                    status_code=409,
                    detail=f"An agent with email '{email}' is already registered. Please use a different email address."
                )
                
            print("✅ Email availability check passed")
            
        except HTTPException:
            raise
        except Exception as db_check_error:
            print(f"⚠️ Database check error (continuing): {db_check_error}")
            # Continue with registration - this might be a temporary DB issue
        
        # ===== CREDENTIAL GENERATION =====
        print("🔑 Generating agent credentials...")
        
        agent_id = generate_unique_agent_id(db)
        password = generate_secure_password()
        
        print(f"✅ Generated credentials - ID: {agent_id}")
        
        # ===== DATABASE INSERTION =====
        try:
            print("💾 Creating agent record...")
            
            # Create agent object
            new_agent = Agent(
                agent_id=agent_id,
                name=name_clean,
                email=email_clean,
                mobile=mobile_clean,
                dob=dob,
                country=country_clean,
                gender=gender_clean,
                password=password,  # Store plain password for now
                status="active",
                created_at=datetime.utcnow()
            )
            
            # Add to database session
            db.add(new_agent)
            print("📝 Added to database session")
            
            # Commit transaction
            db.commit()
            print("💾 Database commit successful")
            
            # Refresh object to get updated data
            db.refresh(new_agent)
            print(f"✅ Agent created with database ID: {new_agent.id}")
            
        except Exception as db_error:
            print(f"❌ Database insertion error: {db_error}")
            
            # Rollback transaction
            if hasattr(db, 'rollback'):
                db.rollback()
                print("🔄 Database rollback completed")
            
            # Handle specific database errors
            error_message = str(db_error).lower()
            
            if 'unique constraint' in error_message or 'duplicate' in error_message:
                if 'email' in error_message:
                    raise HTTPException(
                        status_code=409,
                        detail="This email address is already registered. Please use a different email."
                    )
                elif 'agent_id' in error_message:
                    raise HTTPException(
                        status_code=500,
                        detail="ID generation conflict. Please try again."
                    )
            
            # Generic database error
            raise HTTPException(
                status_code=500,
                detail="Database error occurred during registration. Please try again in a moment."
            )
        
        # ===== POST-REGISTRATION SETUP =====
        try:
            print("🔧 Setting up initial task progress...")
            # Create initial task progress record
            initial_progress = TaskProgress(
                agent_id=agent_id,
                current_index=0,
                status="pending",
                assigned_at=datetime.utcnow()
            )
            db.add(initial_progress)
            db.commit()
            print(f"✅ Created initial task progress for {agent_id}")
            
        except Exception as progress_error:
            print(f"⚠️ Task progress creation failed (non-critical): {progress_error}")
            # This is non-critical, so don't fail the entire registration
        
        # ===== SUCCESS RESPONSE =====
        print(f"🎉 Registration completed successfully for {agent_id}")
        
        response_data = {
            "success": True,
            "message": "Agent registered successfully!",
            "agent_id": agent_id,
            "password": password,
            "agent_details": {
                "id": new_agent.id,
                "agent_id": agent_id,
                "name": name_clean,
                "email": email_clean,
                "mobile": mobile_clean,
                "dob": dob,
                "country": country_clean,
                "gender": gender_clean,
                "status": "active",
                "created_at": new_agent.created_at.isoformat()
            },
            "next_steps": {
                "login_url": "/agent.html",
                "api_endpoints": {
                    "login": "/api/agents/login",
                    "current_task": f"/api/agents/{agent_id}/current-task",
                    "submit_task": f"/api/agents/{agent_id}/submit"
                }
            },
            "important_note": "Please save your Agent ID and Password securely. You'll need them to log in."
        }
        
        return response_data
        
    except HTTPException as http_error:
        # Re-raise HTTP exceptions (these are expected errors with proper status codes)
        print(f"⚠️ HTTP Exception: {http_error.detail}")
        raise http_error
        
    except Exception as unexpected_error:
        print(f"❌ Unexpected registration error: {unexpected_error}")
        
        # Log full traceback for debugging
        import traceback
        traceback.print_exc()
        
        # Ensure database rollback
        try:
            if hasattr(db, 'rollback'):
                db.rollback()
        except Exception as rollback_error:
            print(f"❌ Rollback error: {rollback_error}")
        
        # Return generic error to user (don't expose internal details)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during registration. Please try again or contact support."
        )

# ===================== ALTERNATIVE REGISTRATION ENDPOINTS =====================

@app.post("/api/admin/register-agent")
@limiter.limit("5/minute")
async def admin_register_agent(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(...),
    country: str = Form(...),
    gender: str = Form(...),
    db=Depends(db_dependency)
):
    """Admin version of agent registration - same functionality as main endpoint"""
    return await register_new_agent(request, name, email, mobile, dob, country, gender, db)
    # main.py - Part 4: Test Endpoints and Status Checks

# ===================== TEST AND STATUS ENDPOINTS =====================

@app.post("/api/test/register-agent")
@limiter.limit("5/minute")
async def test_register_agent(request: Request, db=Depends(db_dependency)):
    """Test endpoint to register an agent with sample data"""
    try:
        # Generate unique test data
        import time
        timestamp = str(int(time.time()))
        
        test_data = {
            "name": "Test Agent Sample",
            "email": f"test.agent.{timestamp}@example.com",
            "mobile": "+1234567890",
            "dob": "1990-01-01",
            "country": "United States",
            "gender": "Male"
        }
        
        print(f"🧪 Testing registration with: {test_data['email']}")
        
        # Call the main registration endpoint
        return await register_new_agent(
            request=request,
            name=test_data["name"],
            email=test_data["email"],
            mobile=test_data["mobile"],
            dob=test_data["dob"],
            country=test_data["country"],
            gender=test_data["gender"],
            db=db
        )
        
    except Exception as test_error:
        print(f"❌ Test registration error: {test_error}")
        return {
            "success": False,
            "error": f"Test registration failed: {str(test_error)}",
            "error_type": type(test_error).__name__,
            "message": "Test registration endpoint failed"
        }

@app.get("/api/agents/registration-status")
@limiter.limit("50/minute")
async def check_registration_status(request: Request, db=Depends(db_dependency)):
    """Check if agent registration system is working"""
    try:
        status_info = {
            "timestamp": datetime.now().isoformat(),
            "database_ready": database_ready,
            "status": "unknown"
        }
        
        if not database_ready:
            status_info.update({
                "status": "unavailable",
                "message": "Database is not ready for registration",
                "database_ready": False
            })
            return status_info
        
        # Test database connectivity
        try:
            agent_count = db.query(Agent).count()
            status_info.update({
                "status": "available",
                "message": "Agent registration system is operational",
                "database_ready": True,
                "total_agents": agent_count,
                "endpoints": {
                    "register": "/api/agents/register",
                    "login": "/api/agents/login",
                    "test_register": "/api/test/register-agent"
                }
            })
            
        except Exception as db_test_error:
            status_info.update({
                "status": "database_error",
                "message": f"Database connectivity issue: {str(db_test_error)[:100]}",
                "database_ready": False
            })
        
        return status_info
        
    except Exception as status_error:
        return {
            "status": "error",
            "message": f"Status check failed: {str(status_error)}",
            "database_ready": database_ready,
            "timestamp": datetime.now().isoformat()
        }

# ===================== AGENT LOGIN ENDPOINT =====================

@app.post("/api/agents/login")
@limiter.limit("20/minute")
async def login_agent(
    agent_id: str = Form(...),
    password: str = Form(...),
    db=Depends(db_dependency)
):
    """Fixed agent login endpoint"""
    print(f"🔑 Login attempt for agent: {agent_id}")
    
    try:
        # Validate inputs
        if not agent_id or not password:
            raise HTTPException(
                status_code=400,
                detail="Both Agent ID and password are required"
            )
        
        if not database_ready:
            raise HTTPException(
                status_code=503,
                detail="Authentication service is temporarily unavailable"
            )
        
        # Find agent
        agent = db.query(Agent).filter(Agent.agent_id == agent_id.strip()).first()
        if not agent:
            print(f"❌ Agent not found: {agent_id}")
            raise HTTPException(
                status_code=401,
                detail="Invalid Agent ID or password"
            )
        
        # Check password (plain text comparison for now)
        if agent.password != password:
            print(f"❌ Invalid password for agent: {agent_id}")
            raise HTTPException(
                status_code=401,
                detail="Invalid Agent ID or password"
            )
        
        # Check if agent is active
        if agent.status != "active":
            print(f"❌ Agent not active: {agent_id} (status: {agent.status})")
            raise HTTPException(
                status_code=403,
                detail="Your account is not active. Please contact support."
            )
        
        # Create session record (optional)
        try:
            # End any existing active sessions
            active_sessions = db.query(AgentSession).filter(
                AgentSession.agent_id == agent_id,
                AgentSession.logout_time.is_(None)
            ).all()
            
            for session in active_sessions:
                session.logout_time = datetime.utcnow()
                duration = (session.logout_time - session.login_time).total_seconds() / 60
                session.duration_minutes = round(duration, 2)
            
            # Create new session
            new_session = AgentSession(
                agent_id=agent_id,
                login_time=datetime.utcnow(),
                ip_address=request.client.host if hasattr(request, 'client') else "unknown",
                user_agent=request.headers.get("user-agent", "unknown")[:255]
            )
            
            db.add(new_session)
            db.commit()
            print(f"✅ Session created for agent: {agent_id}")
            
        except Exception as session_error:
            print(f"⚠️ Session creation error (non-critical): {session_error}")
            # Don't fail login if session creation fails
        
        print(f"✅ Login successful for agent: {agent_id}")
        
        return {
            "success": True,
            "message": "Login successful",
            "agent_id": agent.agent_id,
            "name": agent.name,
            "email": agent.email,
            "status": agent.status,
            "login_time": datetime.utcnow().isoformat(),
            "next_steps": {
                "get_current_task": f"/api/agents/{agent_id}/current-task",
                "submit_task": f"/api/agents/{agent_id}/submit",
                "view_progress": f"/api/agents/{agent_id}/progress"
            }
        }
        
    except HTTPException:
        raise
    except Exception as login_error:
        # main.py - Part 5: Admin Endpoints and Statistics

        print(f"❌ Login error: {login_error}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Login failed due to a system error. Please try again."
        )

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

# ===================== STATISTICS ENDPOINTS =====================
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
            
            try:
                latest_session = db.query(AgentSession).filter(
                    AgentSession.agent_id == agent.agent_id
                ).order_by(AgentSession.login_time.desc()).first()
            except Exception:
                latest_session = None
            
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

# ===================== TASK MANAGEMENT ENDPOINTS =====================

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
        
        # Get current task progress
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        if not progress:
            progress = TaskProgress(agent_id=agent_id, current_index=0)
            db.add(progress)
            db.commit()
            db.refresh(progress)
        
        # For now, return a simple response since task images aren't set up yet
        return {
            "completed": False,
            "message": "Task system ready",
            "agent_id": agent_id,
            "current_index": progress.current_index,
            "progress": f"{progress.current_index}/0",
            "note": "No tasks assigned yet. Use admin panel to upload task images."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting current task for {agent_id}: {e}")
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
        
        # For now, just acknowledge the submission
        return {
            "success": True,
            "message": "Task submission system ready",
            "agent_id": agent_id,
            "note": "Task submission will be available once tasks are assigned"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting task for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting task: {str(e)}")
        # main.py - Part 6: Health Checks and Static File Serving

# ===================== HEALTH CHECK ENDPOINTS =====================

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
        "version": "2.0.0"
    }
    
    # Test database connectivity with proper session handling
    if database_ready:
        try:
            db_gen = db_dependency()
            if hasattr(db_gen, '__next__'):
                db_session = next(db_gen)
            else:
                db_session = db_gen
            
            try:
                # Simple test for database connectivity
                if hasattr(db_session, 'execute'):
                    from sqlalchemy import text
                    result = db_session.execute(text("SELECT 1")).scalar()
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
                if hasattr(db_session, 'close'):
                    db_session.close()
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

# ===================== STATIC FILE SERVING =====================

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
        
        # Create basic admin.html if it doesn't exist
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
            
        # Create basic agent.html if it doesn't exist
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

# ===================== DEBUG ENDPOINTS =====================

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

# ===================== MAIN ENTRY POINT =====================

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
    print(f"- Registration Test: http://localhost:{port}/api/test/register-agent")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
    
