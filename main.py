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
from datetime import datetime, date
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
        # main.py - Part 3: Helper Functions

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
    
    # Date of birth validation (optional)
    if dob:
        try:
            dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
            age = (date.today() - dob_date).days // 365
            if age < 16:
                errors.append("Agent must be at least 16 years old")
            if age > 100:
                errors.append("Please enter a valid date of birth")
        except ValueError:
            errors.append("Invalid date format. Please use YYYY-MM-DD")
    
    # Gender validation (optional)
    if gender:
        valid_genders = ['Male', 'Female', 'Other']
        if gender not in valid_genders:
            errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
    
    return errors
    # main.py - Part 4: FIXED Agent Registration Endpoint

@app.post("/api/agents/register")
@limiter.limit("10/minute")
async def register_new_agent(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(None),      # FIXED: Made optional with default None
    country: str = Form(None),  # FIXED: Made optional with default None
    gender: str = Form(None),   # FIXED: Made optional with default None
    db=Depends(db_dependency)
):
    """
    FIXED Agent Registration Endpoint - Compatible with both admin panel and agent routes
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
        
        # Clean and prepare required data
        name_clean = name.strip() if name else ''
        email_clean = email.strip().lower() if email else ''
        mobile_clean = re.sub(r'[\s\-\(\)\.]+', '', mobile.strip()) if mobile else ''
        
        # Basic validation for required fields
        validation_errors = []
        
        if not name_clean or len(name_clean) < 2:
            validation_errors.append("Name must be at least 2 characters long")
        
        if not email_clean:
            validation_errors.append("Email is required")
        else:
            email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
            if not re.match(email_pattern, email_clean):
                validation_errors.append("Please enter a valid email address")
        
        if not mobile_clean:
            validation_errors.append("Mobile number is required")
        else:
            if not re.match(r'^\+?\d{10,15}$', mobile_clean):
                validation_errors.append("Mobile number must contain 10-15 digits")
        
        # Optional field validation (only if provided)
        if dob:
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                age = (date.today() - dob_date).days // 365
                if age < 16:
                    validation_errors.append("Agent must be at least 16 years old")
                if age > 100:
                    validation_errors.append("Please enter a valid date of birth")
            except ValueError:
                validation_errors.append("Invalid date format. Please use YYYY-MM-DD")
        
        if gender and gender not in ['Male', 'Female', 'Other']:
            validation_errors.append("Gender must be Male, Female, or Other")
        
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
        
        # ===== CREDENTIAL GENERATION =====
        print("🔑 Generating agent credentials...")
        
        agent_id = generate_unique_agent_id(db)
        password = generate_secure_password()
        
        print(f"✅ Generated credentials - ID: {agent_id}")
        
        # ===== DATABASE INSERTION =====
        try:
            print("💾 Creating agent record...")
            
            # Create agent object with required fields
            agent_data = {
                'agent_id': agent_id,
                'name': name_clean,
                'email': email_clean,
                'mobile': mobile_clean,
                'password': password,
                'status': 'active',
                'created_at': datetime.utcnow()
            }
            
            # Add optional fields only if provided
            if dob:
                agent_data['dob'] = datetime.strptime(dob, '%Y-%m-%d').date()
            if country:
                agent_data['country'] = country.strip()
            if gender:
                agent_data['gender'] = gender
            
            new_agent = Agent(**agent_data)
            
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
                elif 'mobile' in error_message:
                    raise HTTPException(
                        status_code=409,
                        detail="This mobile number is already registered. Please use a different mobile number."
                    )
            
            # Generic database error
            raise HTTPException(
                status_code=500,
                detail="Database error occurred during registration. Please try again in a moment."
            )
        
        # ===== POST-REGISTRATION SETUP =====
        try:
            print("🔧 Setting up initial task progress...")
            # Only create TaskProgress if the model exists
            if database_ready and 'TaskProgress' in globals():
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
        
        # Add optional fields to response if they were provided
        if dob:
            response_data["agent_details"]["dob"] = dob
        if country:
            response_data["agent_details"]["country"] = country
        if gender:
            response_data["agent_details"]["gender"] = gender
        
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
        # main.py - Part 5: Login and Test Endpoints

# ===================== AGENT LOGIN ENDPOINT =====================

@app.post("/api/agents/login")
@limiter.limit("20/minute")
async def login_agent(
    request: Request,
    agent_id: str = Form(...),
    password: str = Form(...),
    db=Depends(db_dependency)
):
    """Fixed agent login endpoint with proper rate limiting"""
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
        print(f"❌ Login error: {login_error}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Login failed due to a system error. Please try again."
        )

# ===================== TESTING ENDPOINTS =====================

@app.get("/api/agents/test-generation")
async def test_credential_generation(db=Depends(db_dependency)):
    """Test credential generation without creating agents"""
    try:
        if not database_ready:
            return {"error": "Database not ready"}
        
        # Test ID generation
        test_agent_id = generate_unique_agent_id(db)
        test_password = generate_secure_password()
        
        return {
            "success": True,
            "test_credentials": {
                "agent_id": test_agent_id,
                "password": test_password
            },
            "message": "Credential generation test successful",
            "database_ready": database_ready
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Credential generation test failed"
        }

@app.post("/api/test/register-agent-full")
async def test_register_full_agent(request: Request, db=Depends(db_dependency)):
    """Full test registration with all fields"""
    import time
    timestamp = str(int(time.time()))
    
    test_data = {
        "name": f"Test Agent {timestamp}",
        "email": f"test.agent.{timestamp}@example.com",
        "mobile": f"+1234567{timestamp[-3:]}",
        "dob": "1990-01-01",
        "country": "United States",
        "gender": "Male"
    }
    
    try:
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
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_data": test_data
        }

# ===================== SIMPLIFIED REGISTRATION FOR COMPATIBILITY =====================

@app.post("/api/agents/register-simple")
async def register_agent_simple_form(
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    db=Depends(db_dependency)
):
    """Simplified registration form for basic agent creation"""
    try:
        # Create a mock request object
        class MockRequest:
            pass
        
        request = MockRequest()
        
        # Call main registration with minimal required fields
        return await register_new_agent(
            request=request,
            name=name,
            email=email,
            mobile=mobile,
            dob=None,
            country=None,
            gender=None,
            db=db
        )
        
    except Exception as e:
        print(f"❌ Simple registration error: {e}")
        return {
            "success": False,
            "message": f"Registration failed: {str(e)}",
            "error": str(e)
        }

@app.post("/api/agents/login-simple")
async def login_agent_simple(
    agent_id: str = Form(...),
    password: str = Form(...),
    db=Depends(db_dependency)
):
    """Simple agent login without rate limiting (fallback)"""
    print(f"🔑 Simple login attempt for agent: {agent_id}")
    
    try:
        if not database_ready:
            return {"success": False, "message": "Database not ready"}
        
        # Find agent
        agent = db.query(Agent).filter(Agent.agent_id == agent_id.strip()).first()
        if not agent:
            return {"success": False, "message": "Invalid credentials"}
        
        # Check password
        if agent.password != password:
            return {"success": False, "message": "Invalid credentials"}
        
        # Check if agent is active
        if agent.status != "active":
            return {"success": False, "message": "Account not active"}
        
        return {
            "success": True,
            "message": "Login successful",
            "agent_id": agent.agent_id,
            "name": agent.name,
            "email": agent.email,
            "status": agent.status
        }
        
    except Exception as e:
        print(f"❌ Simple login error: {e}")
        return {"success": False, "message": "Login error occurred"}

# ===================== ADMIN ENDPOINTS =====================

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
        # main.py - Part 6: Health Checks, Static Files, and Entry Point (Final)

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

# ===================== REGISTRATION STATUS ENDPOINTS =====================

@app.get("/api/agents/registration-status")
@limiter.limit("50/minute")
async def check_registration_status(request: Request, db=Depends(db_dependency)):
    """Check if agent registration system is ready"""
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

# ===================== TASK ENDPOINTS =====================

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
