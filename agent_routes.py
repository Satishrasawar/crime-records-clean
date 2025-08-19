# agent_routes.py - Part 1: Imports, Setup & Utility Functions
from fastapi import APIRouter, Form, Depends, HTTPException, UploadFile, File, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Agent, TaskProgress, SubmittedForm, AgentSession, Admin
from app.schemas import AgentStatusUpdateSchema
from app.security import hash_password, verify_password
import os
import secrets
import string
from datetime import datetime, timedelta, date
import json
import zipfile
import io
import pandas as pd
from typing import Optional
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import re

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def generate_agent_credentials():
    """Generate unique agent ID and secure password"""
    agent_id = "AGT" + "".join(secrets.choice(string.digits) for _ in range(6))
    password = "".join(secrets.choice(string.ascii_letters + string.digits + "!@#$%") for _ in range(10))
    return agent_id, password

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number format"""
    # Remove spaces, dashes, and parentheses
    clean_mobile = re.sub(r'[\s\-\(\)]', '', mobile)
    # Check if it's 10-15 digits
    return re.match(r'^\+?[1-9]\d{9,14}$', clean_mobile) is not None

def get_agent_image_files(agent_id: str):
    """Get all image files assigned to a specific agent"""
    agent_folder = f"static/task_images/agent_{agent_id}"
    if not os.path.exists(agent_folder):
        # Fallback to general folder
        agent_folder = "static/task_images/crime_records_wide"
    
    if not os.path.exists(agent_folder):
        return []
    
    return sorted([f for f in os.listdir(agent_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def get_agent_mobile_safe(agent):
    """Safely get agent mobile number"""
    return getattr(agent, 'mobile', 'N/A')
    # agent_routes.py - Part 2: Agent Registration Endpoints

# ===================== AGENT REGISTRATION ENDPOINTS =====================

@router.post("/api/agents/register")
@limiter.limit("3/minute")
async def register_agent(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(None),      # Made optional
    country: str = Form(None),  # Made optional
    gender: str = Form(None),   # Made optional
    db: Session = Depends(get_db)
):
    """Register a new agent"""
    try:
        print(f"🆕 Agent registration attempt: {name}, {email}")
        
        # Validate input data
        if not name or len(name.strip()) < 2:
            raise HTTPException(status_code=400, detail="Name must be at least 2 characters long")
        
        if not validate_email(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        if not validate_mobile(mobile):
            raise HTTPException(status_code=400, detail="Invalid mobile number format. Use 10-15 digits")
        
        # Clean inputs
        name = name.strip()
        email = email.strip().lower()
        mobile = re.sub(r'[\s\-\(\)]', '', mobile)
        
        # Validate optional fields
        if dob:
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                age = (date.today() - dob_date).days // 365
                if age < 16 or age > 80:
                    raise HTTPException(status_code=400, detail="Agent must be between 16 and 80 years old")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        if gender and gender not in ['Male', 'Female', 'Other']:
            raise HTTPException(status_code=400, detail="Gender must be Male, Female, or Other")
        
        # Check if agent already exists
        existing_agent = db.query(Agent).filter(
            (Agent.email == email) | (Agent.mobile == mobile)
        ).first()
        
        if existing_agent:
            if existing_agent.email == email:
                raise HTTPException(status_code=409, detail="Email already registered")
            else:
                raise HTTPException(status_code=409, detail="Mobile number already registered")
        
        # Generate unique agent credentials
        max_attempts = 10
        agent_id = None
        password = None
        
        for attempt in range(max_attempts):
            temp_agent_id, temp_password = generate_agent_credentials()
            # Check if agent_id is unique
            existing_id = db.query(Agent).filter(Agent.agent_id == temp_agent_id).first()
            if not existing_id:
                agent_id = temp_agent_id
                password = temp_password
                break
        
        if not agent_id:
            raise HTTPException(status_code=500, detail="Failed to generate unique agent ID")
        
        # Create new agent - Handle optional fields properly
        agent_data = {
            'agent_id': agent_id,
            'name': name,
            'email': email,
            'mobile': mobile,
            'password': password,
            'status': "active",
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
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        print(f"✅ Agent registered successfully: {agent_id}")
        
        # Build response
        response_data = {
            "success": True,
            "message": "Registration successful! Please save your credentials.",
            "agent_id": agent_id,
            "password": password,
            "name": name,
            "email": email,
            "mobile": mobile,
            "status": "active",
            "instructions": [
                "Save your Agent ID and Password securely",
                "Use these credentials to log in",
                "Contact admin if you need assistance"
            ]
        }
        
        # Add optional fields to response if provided
        if dob:
            response_data["dob"] = dob
        if country:
            response_data["country"] = country
        if gender:
            response_data["gender"] = gender
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/api/agents/check-availability")
@limiter.limit("10/minute")
async def check_availability(
    request: Request,
    email: str = Form(None),
    mobile: str = Form(None),
    db: Session = Depends(get_db)
):
    """Check if email or mobile is available for registration"""
    try:
        result = {"available": True, "message": "Available"}
        
        if email:
            email = email.strip().lower()
            if not validate_email(email):
                return {"available": False, "message": "Invalid email format"}
            
            existing_email = db.query(Agent).filter(Agent.email == email).first()
            if existing_email:
                return {"available": False, "message": "Email already registered"}
        
        if mobile:
            mobile = re.sub(r'[\s\-\(\)]', '', mobile)
            if not validate_mobile(mobile):
                return {"available": False, "message": "Invalid mobile format"}
            
            existing_mobile = db.query(Agent).filter(Agent.mobile == mobile).first()
            if existing_mobile:
                return {"available": False, "message": "Mobile number already registered"}
        
        return result
        
    except Exception as e:
        print(f"❌ Availability check error: {e}")
        return {"available": False, "message": "Check failed"}

@router.get("/api/agents/test-registration")
@limiter.limit("5/minute")
async def test_registration_system(request: Request, db: Session = Depends(get_db)):
    """Test the registration system"""
    try:
        # Test database connectivity
        agent_count = db.query(Agent).count()
        
        # Test credential generation
        test_agent_id, test_password = generate_agent_credentials()
        
        return {
            "system_status": "healthy",
            "database_connected": True,
            "current_agent_count": agent_count,
            "sample_credentials": {
                "agent_id": test_agent_id,
                "password": test_password
            },
            "registration_endpoint": "/api/agents/register",
            "required_fields": ["name", "email", "mobile"],
            "optional_fields": ["dob", "country", "gender"],
            "validation_rules": {
                "name": "Minimum 2 characters",
                "email": "Valid email format",
                "mobile": "10-15 digits, international format supported"
            }
        }
        
    except Exception as e:
        return {
            "system_status": "error",
            "error": str(e),
            "database_connected": False
        }
        # agent_routes.py - Part 3: Admin Login & Authentication

# ===================== ADMIN LOGIN ENDPOINTS =====================

@router.post("/api/admin/login")
@limiter.limit("5/minute")
async def admin_login(request: Request, db: Session = Depends(get_db)):
    """FIXED Admin login endpoint"""
    try:
        print("🔐 FIXED Admin login attempt received")
        
        # Parse request data
        try:
            data = await request.json()
            print(f"📨 Login data received: {list(data.keys()) if data else 'No data'}")
        except Exception as parse_error:
            print(f"❌ Failed to parse login data: {parse_error}")
            raise HTTPException(status_code=400, detail="Invalid JSON data")
        
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        
        print(f"🔍 Login attempt - Username: '{username}', Password provided: {bool(password)}")

        if not username or not password:
            print("❌ Missing username or password")
            raise HTTPException(status_code=400, detail="Username and password are required")

        print(f"🔍 Querying database for admin: {username}")
        admin = db.query(Admin).filter(Admin.username == username).first()
        
        if not admin:
            print(f"❌ Admin user '{username}' not found in database")
            # Check if any admins exist
            admin_count = db.query(Admin).count()
            print(f"📊 Total admins in database: {admin_count}")
            if admin_count == 0:
                print("⚠️ No admin users exist in database!")
                # Auto-create admin for testing
                try:
                    print("🔧 Auto-creating admin user...")
                    hashed_password = hash_password("admin123")
                    new_admin = Admin(
                        username="admin",
                        hashed_password=hashed_password,
                        email="admin@agent-task-system.com",
                        is_active=True,
                        created_at=datetime.utcnow()
                    )
                    db.add(new_admin)
                    db.commit()
                    db.refresh(new_admin)
                    admin = new_admin
                    print("✅ Auto-created admin user successfully!")
                except Exception as create_error:
                    print(f"❌ Failed to auto-create admin: {create_error}")
                    raise HTTPException(status_code=401, detail="Invalid credentials")
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        print(f"✅ Admin user found: {admin.username}, Active: {admin.is_active}")
        
        # Check if admin is active
        if not admin.is_active:
            print(f"❌ Admin {username} is not active")
            raise HTTPException(status_code=403, detail="Account is not active")

        # Verify password
        try:
            # Handle both hashed and plain passwords for debugging
            if admin.hashed_password.startswith('$2b$') or admin.hashed_password.startswith('$2a$'):
                # This is a bcrypt hash
                password_valid = verify_password(password, admin.hashed_password)
                print(f"🔐 Bcrypt password verification result: {password_valid}")
            else:
                # Might be plain text (for debugging) - re-hash it
                print("⚠️ Plain text password detected - converting to hash")
                if admin.hashed_password == password:
                    # Update to hashed password
                    admin.hashed_password = hash_password(password)
                    db.commit()
                    password_valid = True
                    print("✅ Password updated to hash and verified")
                else:
                    password_valid = False
                    print("❌ Plain text password doesn't match")
            
        except Exception as verify_error:
            print(f"❌ Password verification error: {verify_error}")
            # Fallback: try direct comparison for debugging
            if admin.hashed_password == password:
                password_valid = True
                print("✅ Fallback: Direct password match")
            else:
                password_valid = False
                print("❌ Fallback: Direct password mismatch")
        
        if not password_valid:
            print(f"❌ Invalid password for admin {username}")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Create access token
        try:
            access_token = create_access_token(data={"sub": username})
            print(f"✅ JWT token created successfully for {username}")
        except Exception as token_error:
            print(f"❌ Token creation error: {token_error}")
            # Return success without token for debugging
            access_token = "debug_token_" + secrets.token_urlsafe(32)
        
        print(f"🎉 Admin login successful for {username}")

        return {
            "success": True,
            "access_token": access_token,
            "token_type": "bearer",
            "message": "Login successful",
            "admin_info": {
                "username": admin.username,
                "email": admin.email,
                "is_active": admin.is_active
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected admin login error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# SIMPLE TEST LOGIN ENDPOINT (NO JWT)
@router.post("/api/admin/simple-login")
@limiter.limit("10/minute")
async def simple_admin_login(request: Request, db: Session = Depends(get_db)):
    """Simple admin login for testing without JWT complexity"""
    try:
        data = await request.json()
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        
        print(f"🧪 Simple login test - Username: '{username}'")
        
        if not username or not password:
            return {"success": False, "message": "Username and password required"}
        
        # Hardcoded test for debugging
        if username == "admin" and password == "admin123":
            print("✅ Hardcoded credentials matched!")
            return {
                "success": True,
                "message": "Login successful (hardcoded test)",
                "access_token": "test_token_123",
                "token_type": "bearer"
            }
        
        # Try database
        admin = db.query(Admin).filter(Admin.username == username).first()
        if admin:
            print(f"👤 Found admin in database: {admin.username}")
            # Try multiple password checks
            if (admin.hashed_password == password or 
                (admin.hashed_password.startswith('$') and verify_password(password, admin.hashed_password))):
                return {
                    "success": True,
                    "message": "Login successful (database)",
                    "access_token": "db_token_123",
                    "token_type": "bearer"
                }
        
        return {"success": False, "message": "Invalid credentials"}
        
    except Exception as e:
        print(f"❌ Simple login error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

# ADMIN STATUS CHECK
@router.get("/api/admin/status")
@limiter.limit("20/minute")
async def check_admin_status(request: Request, db: Session = Depends(get_db)):
    """Check admin system status"""
    try:
        admin_count = db.query(Admin).count()
        
        if admin_count == 0:
            # Auto-create admin
            try:
                hashed_password = hash_password("admin123")
                new_admin = Admin(
                    username="admin",
                    hashed_password=hashed_password,
                    email="admin@agent-task-system.com",
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                db.add(new_admin)
                db.commit()
                admin_count = 1
                print("✅ Auto-created admin user")
            except Exception as create_error:
                print(f"❌ Failed to create admin: {create_error}")
        
        admins = db.query(Admin).all()
        admin_list = []
        
        for admin in admins:
            admin_list.append({
                "username": admin.username,
                "email": admin.email,
                "is_active": admin.is_active,
                "has_hash": admin.hashed_password.startswith('$') if admin.hashed_password else False,
                "created_at": admin.created_at.isoformat() if admin.created_at else None
            })
        
        return {
            "admin_count": admin_count,
            "admins": admin_list,
            "test_credentials": {
                "username": "admin",
                "password": "admin123"
            },
            "status": "ready"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "error"}

# CORS PREFLIGHT HANDLER
@router.options("/api/admin/login")
async def admin_login_options(request: Request):
    """Handle CORS preflight for admin login"""
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )
    # agent_routes.py - Part 4: COMPLETE Admin Registration & Debug Endpoints

# ===================== ADMIN REGISTRATION ENDPOINT =====================

@router.post("/api/admin/register-agent")
@limiter.limit("5/minute")
async def admin_register_agent(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: str = Form(None),      # Made optional
    country: str = Form(None),  # Made optional
    gender: str = Form(None),   # Made optional
    status: str = Form("active"),
    db: Session = Depends(get_db)
):
    """Admin-specific agent registration with additional options"""
    try:
        print(f"🔐 Admin registering agent: {name} ({email})")
        
        # Validate inputs
        if not validate_email(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        if not validate_mobile(mobile):
            raise HTTPException(status_code=400, detail="Invalid mobile format")
        
        if status not in ["active", "inactive", "pending"]:
            status = "active"
        
        # Clean inputs
        name = name.strip()
        email = email.strip().lower()
        mobile = re.sub(r'[\s\-\(\)]', '', mobile)
        
        # Check duplicates
        existing = db.query(Agent).filter(
            (Agent.email == email) | (Agent.mobile == mobile)
        ).first()
        
        if existing:
            raise HTTPException(status_code=409, detail="Email or mobile already exists")
        
        # Generate credentials
        agent_id, password = generate_agent_credentials()
        
        # Ensure unique agent_id
        while db.query(Agent).filter(Agent.agent_id == agent_id).first():
            agent_id, password = generate_agent_credentials()
        
        # Create agent data
        agent_data = {
            'agent_id': agent_id,
            'name': name,
            'email': email,
            'mobile': mobile,
            'password': password,
            'status': status,
            'created_at': datetime.utcnow()
        }
        
        # Add optional fields only if provided
        if dob:
            try:
                agent_data['dob'] = datetime.strptime(dob, '%Y-%m-%d').date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        if country:
            agent_data['country'] = country.strip()
        if gender:
            agent_data['gender'] = gender
        
        new_agent = Agent(**agent_data)
        
        db.add(new_agent)
        db.commit()
        db.refresh(new_agent)
        
        response_data = {
            "success": True,
            "message": "Agent created successfully",
            "agent_id": agent_id,
            "password": password,
            "name": name,
            "email": email,
            "mobile": mobile,
            "status": status
        }
        
        # Add optional fields to response
        if dob:
            response_data["dob"] = dob
        if country:
            response_data["country"] = country
        if gender:
            response_data["gender"] = gender
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Admin create agent error: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

# ===================== ADMIN DEBUG ENDPOINTS =====================

@router.post("/api/admin/debug-create")
@limiter.limit("1/minute")
async def debug_create_admin(request: Request, db: Session = Depends(get_db)):
    """Debug endpoint to create admin user"""
    try:
        print("🔧 Debug: Creating admin user")
        
        # Check if admin already exists
        existing_admin = db.query(Admin).filter(Admin.username == "admin").first()
        if existing_admin:
            print(f"⚠️ Admin already exists: {existing_admin.username}, Active: {existing_admin.is_active}")
            return {
                "message": "Admin already exists",
                "username": existing_admin.username,
                "email": existing_admin.email,
                "is_active": existing_admin.is_active,
                "created_at": existing_admin.created_at.isoformat() if existing_admin.created_at else None
            }
        
        # Create new admin
        print("🔧 Creating new admin user")
        hashed_password = hash_password("admin123")
        print(f"🔐 Generated hash: {hashed_password[:20]}...")
        
        new_admin = Admin(
            username="admin",
            hashed_password=hashed_password,
            email="admin@agent-task-system.com",
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)
        
        print(f"✅ Admin created successfully: ID {new_admin.id}")
        
        return {
            "success": True,
            "message": "Admin user created successfully!",
            "credentials": {
                "username": "admin",
                "password": "admin123"
            },
            "admin_info": {
                "id": new_admin.id,
                "username": new_admin.username,
                "email": new_admin.email,
                "is_active": new_admin.is_active
            }
        }
        
    except Exception as e:
        print(f"❌ Error creating admin: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
            print("🔄 Database rollback completed")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create admin: {str(e)}")

@router.get("/api/admin/debug-check")
@limiter.limit("10/minute")
async def debug_check_admin(request: Request, db: Session = Depends(get_db)):
    """Debug endpoint to check admin status"""
    try:
        print("🔍 Debug: Checking admin status")
        
        # Get all admins
        admins = db.query(Admin).all()
        admin_count = len(admins)
        
        print(f"📊 Found {admin_count} admin users")
        
        if admin_count == 0:
            return {
                "admin_exists": False,
                "admin_count": 0,
                "message": "No admin users found. Use /api/admin/debug-create to create one."
            }
        
        admin_list = []
        for admin in admins:
            admin_info = {
                "id": admin.id,
                "username": admin.username,
                "email": admin.email,
                "is_active": admin.is_active,
                "created_at": admin.created_at.isoformat() if admin.created_at else None,
                "hash_preview": admin.hashed_password[:20] + "..." if admin.hashed_password else "No hash"
            }
            admin_list.append(admin_info)
            print(f"👤 Admin: {admin.username}, Active: {admin.is_active}")
        
        return {
            "admin_exists": True,
            "admin_count": admin_count,
            "admins": admin_list,
            "message": f"Found {admin_count} admin user(s). Use credentials: admin / admin123"
        }
        
    except Exception as e:
        print(f"❌ Error checking admin: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# ===================== ENHANCED SESSION MANAGEMENT =====================

@router.get("/api/admin/session-report")
async def get_session_report(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get session report"""
    try:
        query = db.query(AgentSession)
        
        if agent_id:
            query = query.filter(AgentSession.agent_id == agent_id)
        
        if date_from:
            try:
                from_date = datetime.strptime(date_from, '%Y-%m-%d')
                query = query.filter(AgentSession.login_time >= from_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                to_date = datetime.strptime(date_to, '%Y-%m-%d')
                to_date = to_date.replace(hour=23, minute=59, second=59)
                query = query.filter(AgentSession.login_time <= to_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        sessions = query.order_by(AgentSession.login_time.desc()).all()
        
        result = []
        for session in sessions:
            agent = db.query(Agent).filter(Agent.agent_id == session.agent_id).first()
            result.append({
                "session_id": session.id,
                "agent_id": session.agent_id,
                "agent_name": agent.name if agent else "Unknown",
                "login_time": session.login_time.isoformat() if session.login_time else None,
                "logout_time": session.logout_time.isoformat() if session.logout_time else None,
                "duration_minutes": session.duration_minutes,
                "ip_address": getattr(session, 'ip_address', 'N/A'),
                "user_agent": getattr(session, 'user_agent', 'N/A'),
                "is_active": session.logout_time is None
            })
        
        return result
    except Exception as e:
        print(f"❌ Error in session report: {e}")
        raise HTTPException(status_code=500, detail=f"Session report failed: {str(e)}")

# ===================== TESTING ENDPOINTS =====================

@router.get("/api/agents/test-generation")
async def test_credential_generation(db: Session = Depends(get_db)):
    """Test credential generation without creating agents"""
    try:
        # Test database connectivity
        agent_count = db.query(Agent).count()
        
        # Test ID generation
        test_agent_id, test_password = generate_agent_credentials()
        
        return {
            "success": True,
            "test_credentials": {
                "agent_id": test_agent_id,
                "password": test_password
            },
            "message": "Credential generation test successful",
            "database_connected": True,
            "current_agent_count": agent_count
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Credential generation test failed",
            "database_connected": False
        }
        # agent_routes.py - Part 5: Agent Management & Enhanced Functions

# ===================== AGENT MANAGEMENT ENDPOINTS =====================

@router.get("/api/agents")
def get_all_agents(db: Session = Depends(get_db)):
    """Get all agents with enhanced statistics"""
    try:
        agents = db.query(Agent).order_by(Agent.created_at.desc()).all()
        result = []
        
        for agent in agents:
            # Get task completion count
            completed_count = db.query(SubmittedForm).filter(SubmittedForm.agent_id == agent.agent_id).count()
            
            # Get total assigned tasks
            assigned_images = get_agent_image_files(agent.agent_id)
            total_tasks = len(assigned_images)
            
            # Get session information with proper error handling
            try:
                sessions = db.query(AgentSession).filter(
                    AgentSession.agent_id == agent.agent_id
                ).order_by(AgentSession.login_time.desc()).limit(5).all()
                
                current_session = next((s for s in sessions if s.logout_time is None), None)
                last_login = sessions[0].login_time if sessions else None
                
                completed_sessions = [s for s in sessions if s.logout_time is not None]
                last_logout = completed_sessions[0].logout_time if completed_sessions else None
                
            except Exception as session_error:
                print(f"Session query error for {agent.agent_id}: {session_error}")
                current_session = None
                last_login = None
                last_logout = None
                sessions = []
            
            # Calculate progress
            progress_percentage = (completed_count / total_tasks * 100) if total_tasks > 0 else 0
            
            result.append({
                "id": agent.id,
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "mobile": get_agent_mobile_safe(agent),
                "password": agent.password,
                "status": agent.status,
                "tasks_completed": completed_count,
                "total_tasks": total_tasks,
                "progress_percentage": round(progress_percentage, 2),
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "last_login": last_login.strftime('%Y-%m-%d %H:%M:%S') if last_login else None,
                "last_logout": last_logout.strftime('%Y-%m-%d %H:%M:%S') if last_logout else None,
                "is_currently_logged_in": current_session is not None,
                "recent_sessions": [
                    {
                        "login_time": s.login_time.strftime('%Y-%m-%d %H:%M:%S') if s.login_time else None,
                        "logout_time": s.logout_time.strftime('%Y-%m-%d %H:%M:%S') if s.logout_time else None,
                        "duration_minutes": s.duration_minutes
                    } for s in sessions
                ]
            })
        
        return {"success": True, "agents": result, "total_count": len(result)}
        
    except Exception as e:
        print(f"❌ Error getting agents: {e}")
        # Return the agents list format for compatibility
        agents = db.query(Agent).all()
        result = []
        
        for agent in agents:
            completed_count = db.query(SubmittedForm).filter(SubmittedForm.agent_id == agent.agent_id).count()
            result.append({
                "id": agent.id,
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "password": agent.password,
                "status": agent.status,
                "tasks_completed": completed_count,
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "last_login": None,
                "last_logout": None,
                "current_session_duration": None,
                "is_currently_logged_in": False,
                "recent_sessions": []
            })
        
        return result

@router.get("/api/admin/agent-password/{agent_id}")
def get_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Get agent password for admin (this should be protected in production)"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent_id,
        "password": agent.password,  # Return plain password
        "message": f"Password for agent {agent_id}: {agent.password}",
        "can_reset": True
    }

@router.post("/api/admin/reset-password/{agent_id}")
def reset_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Reset agent password and return new one"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Generate new password
    _, new_password = generate_agent_credentials()
    
    # Update password (store as plain text for now)
    agent.password = new_password
    db.commit()
    
    return {
        "agent_id": agent_id,
        "new_password": new_password,
        "message": "Password reset successfully"
    }

@router.patch("/api/agents/{agent_id}/status")
@limiter.limit("20/minute")
async def update_agent_status(
    agent_id: str, 
    status_data: AgentStatusUpdateSchema, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Update agent status with enhanced validation"""
    try:
        # Find by agent_id string, not integer id
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        valid_statuses = ["active", "inactive", "pending", "suspended"]
        if status_data.status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        old_status = agent.status
        agent.status = status_data.status
        
        # If deactivating, logout any active sessions
        if status_data.status in ["inactive", "suspended"]:
            try:
                active_sessions = db.query(AgentSession).filter(
                    AgentSession.agent_id == agent_id,
                    AgentSession.logout_time.is_(None)
                ).all()
                
                for session in active_sessions:
                    session.logout_time = datetime.utcnow()
                    if session.login_time:
                        duration = (session.logout_time - session.login_time).total_seconds() / 60
                        session.duration_minutes = round(duration, 2)
            except Exception as session_error:
                print(f"⚠️ Session cleanup error: {session_error}")
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Agent status updated from {old_status} to {status_data.status}",
            "agent_id": agent_id,
            "old_status": old_status,
            "new_status": status_data.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Status update error: {e}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")

@router.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete an agent"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    db.delete(agent)
    db.commit()
    return {"message": "Agent deleted successfully"}
    # agent_routes.py - Part 6: Agent Login & Session Management

# ===================== AGENT LOGIN AND SESSION MANAGEMENT =====================

@router.post("/api/agents/login")
@limiter.limit("5/minute")
async def login_agent(
    request: Request,
    agent_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Enhanced agent login with better error handling"""
    try:
        print(f"🔑 Login attempt for agent: {agent_id}")
        
        # Validate inputs
        if not agent_id or not agent_id.strip():
            raise HTTPException(status_code=400, detail="Agent ID is required")
        
        if not password or not password.strip():
            raise HTTPException(status_code=400, detail="Password is required")
        
        agent_id = agent_id.strip()
        password = password.strip()
        
        # Find agent
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            print(f"❌ Agent not found: {agent_id}")
            raise HTTPException(status_code=401, detail="Invalid agent ID or password")
        
        # Check password (handle both plain and hashed)
        password_valid = False
        try:
            if agent.password.startswith('$2b$') or agent.password.startswith('$2a$'):
                # Hashed password
                password_valid = verify_password(password, agent.password)
            else:
                # Plain password
                password_valid = (agent.password == password)
        except Exception as pwd_error:
            print(f"❌ Password verification error: {pwd_error}")
            password_valid = (agent.password == password)  # Fallback to plain comparison
        
        if not password_valid:
            print(f"❌ Invalid password for {agent_id}")
            raise HTTPException(status_code=401, detail="Invalid agent ID or password")
        
        # Check agent status
        if agent.status == "inactive":
            raise HTTPException(status_code=403, detail="Account is inactive. Contact administrator.")
        elif agent.status == "pending":
            raise HTTPException(status_code=403, detail="Account is pending approval. Contact administrator.")
        elif agent.status == "suspended":
            raise HTTPException(status_code=403, detail="Account is suspended. Contact administrator.")
        elif agent.status != "active":
            raise HTTPException(status_code=403, detail=f"Account status: {agent.status}. Contact administrator.")
        
        # Handle sessions
        try:
            # End any existing active sessions for this agent
            active_sessions = db.query(AgentSession).filter(
                AgentSession.agent_id == agent_id,
                AgentSession.logout_time.is_(None)
            ).all()
            
            for session in active_sessions:
                session.logout_time = datetime.utcnow()
                if session.login_time:
                    duration = (session.logout_time - session.login_time).total_seconds() / 60
                    session.duration_minutes = round(duration, 2)
            
            # Create new session
            new_session = AgentSession(
                agent_id=agent_id,
                login_time=datetime.utcnow(),
                ip_address=request.client.host if hasattr(request, 'client') and hasattr(request.client, 'host') else "127.0.0.1",
                user_agent=request.headers.get("user-agent", "Unknown")
            )
            
            db.add(new_session)
            db.commit()
            
        except Exception as session_error:
            print(f"⚠️ Session creation error: {session_error}")
            # Continue with login even if session tracking fails
        
        print(f"✅ Login successful for {agent_id}")
        
        return {
            "success": True,
            "message": "Login successful",
            "agent_id": agent.agent_id,
            "name": agent.name,
            "email": agent.email,
            "mobile": getattr(agent, 'mobile', 'N/A'),
            "status": agent.status,
            "login_time": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Login failed due to server error")

@router.post("/api/agents/{agent_id}/logout")
async def logout_agent(agent_id: str, request: Request, db: Session = Depends(get_db)):
    """Enhanced logout with better error handling"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        active_session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).first()
        
        if active_session:
            active_session.logout_time = datetime.utcnow()
            if active_session.login_time:
                duration = (active_session.logout_time - active_session.login_time).total_seconds() / 60
                active_session.duration_minutes = round(duration, 2)
                session_duration = f"{active_session.duration_minutes} minutes"
            else:
                session_duration = "Unknown duration"
            
            db.commit()
            
            return {
                "success": True,
                "message": "Logout successful",
                "agent_id": agent_id,
                "session_duration": session_duration,
                "logout_time": active_session.logout_time.isoformat()
            }
        else:
            return {
                "success": True,
                "message": "Agent was not logged in",
                "agent_id": agent_id
            }
            
    except Exception as e:
        print(f"❌ Logout error: {e}")
        return {
            "success": True,
            "message": "Logout completed (with errors)",
            "error": str(e)
        }

@router.post("/api/admin/force-logout/{agent_id}")
async def force_logout_agent(agent_id: str, db: Session = Depends(get_db)):
    """Force logout an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        active_session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.logout_time.is_(None)
        ).first()
        
        if active_session:
            active_session.logout_time = datetime.utcnow()
            duration = (active_session.logout_time - active_session.login_time).total_seconds() / 60
            active_session.duration_minutes = round(duration, 2)
            db.commit()
            
            return {
                "message": f"Agent {agent_id} has been forcefully logged out",
                "session_duration": f"{active_session.duration_minutes} minutes"
            }
    except Exception as e:
        print(f"Force logout error: {e}")
    
    return {"message": "Agent was not logged in"}
    # agent_routes.py - Part 7: Task Management & Operations

# ===================== TASK MANAGEMENT =====================

@router.get("/api/agents/{agent_id}/current-task")
def get_current_task(agent_id: str, db: Session = Depends(get_db)):
    """Get current task for an agent"""
    print(f"📋 Getting current task for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
    if not progress:
        progress = TaskProgress(agent_id=agent_id, current_index=0)
        db.add(progress)
        db.commit()
        db.refresh(progress)
        print(f"📋 Created new progress tracker for {agent_id}")

    image_files = get_agent_image_files(agent_id)
    
    if not image_files:
        print(f"📋 No images found for agent {agent_id}")
        return {"message": "No tasks assigned", "completed": True}
    
    if progress.current_index >= len(image_files):
        completed_count = db.query(SubmittedForm).filter(SubmittedForm.agent_id == agent_id).count()
        print(f"✅ All tasks completed for {agent_id}: {completed_count} submissions")
        return {
            "message": "All tasks completed", 
            "completed": True,
            "total_completed": completed_count
        }

    current_image = image_files[progress.current_index]
    
    agent_folder = f"static/task_images/agent_{agent_id}"
    if os.path.exists(agent_folder) and current_image in os.listdir(agent_folder):
        image_url = f"/static/task_images/agent_{agent_id}/{current_image}"
    else:
        image_url = f"/static/task_images/crime_records_wide/{current_image}"

    print(f"📊 Current task for {agent_id}: {progress.current_index + 1}/{len(image_files)} - {current_image}")

    return {
        "image_url": image_url,
        "image_name": current_image,
        "progress": f"{progress.current_index + 1}/{len(image_files)}",
        "current_index": progress.current_index,
        "total_images": len(image_files),
        "task_number": progress.current_index + 1,
        "completed": False
    }

# ===================== TASK SUBMISSION =====================

@router.post("/api/agents/{agent_id}/submit")
async def submit_task_data(
    agent_id: str,
    DR_NO: str = Form(...),
    Date_Rptd: str = Form(...),
    DATE_OCC: str = Form(...),
    TIME_OCC: str = Form(...),
    Unique_Identifier: str = Form(...),
    AREA_NAME: str = Form(...),
    Rpt_Dist_No: str = Form(...),
    VIN: str = Form(...),
    Crm: str = Form(...),
    Crm_Cd_Desc: str = Form(...),
    Mocodes: str = Form(...),
    Vict_Age: str = Form(...),
    Geolocation: str = Form(...),
    DEPARTMENT: str = Form(...),
    Premis_Cd: str = Form(...),
    Premis_Desc: str = Form(...),
    ARREST_KEY: str = Form(...),
    PD_DESC: str = Form(...),
    CCD_LONCOD: str = Form(...),
    Status_Desc: str = Form(...),
    LAW_CODE: str = Form(...),
    SubAgency: str = Form(...),
    Charge: str = Form(...),
    Race: str = Form(...),
    LOCATION: str = Form(...),
    SeqID: str = Form(...),
    LAT: str = Form(...),
    LON: str = Form(...),
    Point: str = Form(...),
    Shape__Area: str = Form(...),
    db: Session = Depends(get_db)
):
    """Submit task form data"""
    print(f"📤 Submit request received for agent: {agent_id}")
    
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        print(f"❌ Agent not found: {agent_id}")
        raise HTTPException(status_code=404, detail="Agent not found")
    
    progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
    current_image_name = None
    
    if progress:
        image_files = get_agent_image_files(agent_id)
        if progress.current_index < len(image_files):
            current_image_name = image_files[progress.current_index]
        print(f"📋 Current task index: {progress.current_index}, Image: {current_image_name}")
    
    form_data = {
        "DR_NO": DR_NO, "Date_Rptd": Date_Rptd, "DATE_OCC": DATE_OCC, "TIME_OCC": TIME_OCC,
        "Unique_Identifier": Unique_Identifier, "AREA_NAME": AREA_NAME, "Rpt_Dist_No": Rpt_Dist_No,
        "VIN": VIN, "Crm": Crm, "Crm_Cd_Desc": Crm_Cd_Desc, "Mocodes": Mocodes,
        "Vict_Age": Vict_Age, "Geolocation": Geolocation, "DEPARTMENT": DEPARTMENT,
        "Premis_Cd": Premis_Cd, "Premis_Desc": Premis_Desc, "ARREST_KEY": ARREST_KEY,
        "PD_DESC": PD_DESC, "CCD_LONCOD": CCD_LONCOD, "Status_Desc": Status_Desc,
        "LAW_CODE": LAW_CODE, "SubAgency": SubAgency, "Charge": Charge, "Race": Race,
        "LOCATION": LOCATION, "SeqID": SeqID, "LAT": LAT, "LON": LON,
        "Point": Point, "Shape__Area": Shape__Area,
        "image_name": current_image_name
    }
    
    print(f"📝 Form data prepared with {len([k for k, v in form_data.items() if v and str(v).strip()])} filled fields")
    
    try:
        submission = SubmittedForm(
            agent_id=agent_id,
            form_data=json.dumps(form_data),
            submitted_at=datetime.utcnow()
        )
        db.add(submission)
        
        if progress:
            progress.current_index += 1
            progress.updated_at = datetime.utcnow()
        else:
            progress = TaskProgress(agent_id=agent_id, current_index=1)
            db.add(progress)
        
        db.commit()
        
        print(f"✅ Task submitted successfully for agent {agent_id}, new index: {progress.current_index}")
        
        return {
            "message": "Task submitted successfully", 
            "success": True,
            "submission_id": submission.id,
            "next_task_index": progress.current_index
        }
    except Exception as db_error:
        print(f"❌ Database error during submission: {db_error}")
        if hasattr(db, 'rollback'):
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save submission: {str(db_error)}")

# ===================== STATISTICS AND REPORTING =====================

@router.get("/api/admin/statistics")
def get_admin_statistics(db: Session = Depends(get_db)):
    """Get admin dashboard statistics"""
    try:
        total_agents = db.query(Agent).count()
        active_agents = db.query(Agent).filter(Agent.status == "active").count()
        total_submissions = db.query(SubmittedForm).count()
        
        # Calculate total tasks across all agents
        total_tasks = 0
        for agent in db.query(Agent).all():
            agent_images = get_agent_image_files(agent.agent_id)
            total_tasks += len(agent_images)
        
        pending_tasks = max(0, total_tasks - total_submissions)
        
        # Get recent activity
        recent_submissions = db.query(SubmittedForm).order_by(
            SubmittedForm.submitted_at.desc()
        ).limit(5).all()
        
        recent_logins = db.query(AgentSession).filter(
            AgentSession.login_time >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "total_tasks": total_tasks,
            "completed_tasks": total_submissions,
            "pending_tasks": pending_tasks,
            "recent_logins_24h": recent_logins,
            "recent_submissions": [
                {
                    "agent_id": sub.agent_id,
                    "submitted_at": sub.submitted_at.isoformat(),
                    "id": sub.id
                } for sub in recent_submissions
            ],
            "completion_rate": round((total_submissions / total_tasks * 100), 2) if total_tasks > 0 else 0
        }
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return {
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "pending_tasks": 0,
            "error": str(e)
        }
        # agent_routes.py - Part 8: COMPLETE File Upload & Task Assignment

# ===================== FILE UPLOAD AND TASK ASSIGNMENT =====================

@router.post("/api/admin/upload-tasks")
async def upload_task_images(
    agent_id: str = Form(...),
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload ZIP file with task images for an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if not zip_file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    try:
        agent_dir = f"static/task_images/agent_{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)
        
        zip_content = await zip_file.read()
        
        images_processed = 0
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue
                
                file_ext = os.path.splitext(file_info.filename.lower())[1]
                if file_ext not in supported_extensions:
                    continue
                
                with zip_ref.open(file_info) as source_file:
                    safe_filename = os.path.basename(file_info.filename)
                    if not safe_filename:
                        continue
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_name = safe_filename
                    while os.path.exists(os.path.join(agent_dir, safe_filename)):
                        name, ext = os.path.splitext(original_name)
                        safe_filename = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    image_path = os.path.join(agent_dir, safe_filename)
                    with open(image_path, 'wb') as dest_file:
                        dest_file.write(source_file.read())
                    
                    images_processed += 1
                    
                    # Prevent excessive uploads
                    if images_processed >= 5000:
                        break
        
        # Reset agent progress to start from beginning
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        if progress:
            progress.current_index = 0
            progress.updated_at = datetime.utcnow()
        else:
            progress = TaskProgress(agent_id=agent_id, current_index=0)
            db.add(progress)
        
        db.commit()
        
        return {
            "message": "Images uploaded successfully",
            "agent_id": agent_id,
            "images_processed": images_processed,
            "agent_directory": agent_dir
        }
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/api/admin/bulk-upload-tasks")
async def bulk_upload_task_images(
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload ZIP file with task images for general distribution"""
    if not zip_file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    try:
        general_dir = "static/task_images/crime_records_wide"
        os.makedirs(general_dir, exist_ok=True)
        
        zip_content = await zip_file.read()
        
        images_processed = 0
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue
                
                file_ext = os.path.splitext(file_info.filename.lower())[1]
                if file_ext not in supported_extensions:
                    continue
                
                with zip_ref.open(file_info) as source_file:
                    safe_filename = os.path.basename(file_info.filename)
                    if not safe_filename:
                        continue
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_name = safe_filename
                    while os.path.exists(os.path.join(general_dir, safe_filename)):
                        name, ext = os.path.splitext(original_name)
                        safe_filename = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    image_path = os.path.join(general_dir, safe_filename)
                    with open(image_path, 'wb') as dest_file:
                        dest_file.write(source_file.read())
                    
                    images_processed += 1
                    
                    if images_processed >= 10000:  # Higher limit for general upload
                        break
        
        return {
            "message": "Bulk images uploaded successfully",
            "images_processed": images_processed,
            "directory": general_dir,
            "note": "These images are available to all agents who don't have specific task assignments"
        }
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        print(f"❌ Bulk upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")

@router.get("/api/admin/task-files/{agent_id}")
async def get_agent_task_files(agent_id: str, db: Session = Depends(get_db)):
    """Get list of task files assigned to an agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        image_files = get_agent_image_files(agent_id)
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        current_index = progress.current_index if progress else 0
        
        return {
            "agent_id": agent_id,
            "total_files": len(image_files),
            "current_index": current_index,
            "files": image_files[:50],  # Return first 50 files to avoid large responses
            "remaining_files": max(0, len(image_files) - 50),
            "completed_count": db.query(SubmittedForm).filter(SubmittedForm.agent_id == agent_id).count()
        }
    except Exception as e:
        print(f"❌ Error getting task files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task files: {str(e)}")

@router.delete("/api/admin/clear-tasks/{agent_id}")
async def clear_agent_tasks(agent_id: str, db: Session = Depends(get_db)):
    """Clear all tasks for a specific agent"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        agent_dir = f"static/task_images/agent_{agent_id}"
        files_deleted = 0
        
        if os.path.exists(agent_dir):
            for filename in os.listdir(agent_dir):
                file_path = os.path.join(agent_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_deleted += 1
            
            # Remove directory if empty
            try:
                os.rmdir(agent_dir)
            except OSError:
                pass  # Directory not empty or other error
        
        # Reset progress
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        if progress:
            progress.current_index = 0
            progress.updated_at = datetime.utcnow()
            db.commit()
        
        return {
            "message": f"Tasks cleared for agent {agent_id}",
            "files_deleted": files_deleted,
            "agent_id": agent_id
        }
    except Exception as e:
        print(f"❌ Error clearing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear tasks: {str(e)}")

# ===================== CHUNKED UPLOAD SUPPORT =====================

@router.post("/api/admin/init-chunked-upload")
async def init_chunked_upload(
    filename: str = Form(...),
    filesize: int = Form(...),
    total_chunks: int = Form(...),
    agent_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Initialize chunked upload session"""
    try:
        # Validate agent exists
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Generate upload session ID
        upload_id = secrets.token_urlsafe(32)
        
        # Create upload session
        upload_session = {
            "upload_id": upload_id,
            "filename": filename,
            "filesize": filesize,
            "total_chunks": total_chunks,
            "agent_id": agent_id,
            "chunks_received": set(),
            "created_at": datetime.utcnow(),
            "temp_dir": os.path.join("temp_chunks", upload_id)
        }
        
        # Create temporary directory
        os.makedirs(upload_session["temp_dir"], exist_ok=True)
        
        # Store session (in production, use Redis or database)
        upload_sessions[upload_id] = upload_session
        
        return {
            "upload_id": upload_id,
            "message": "Chunked upload session initialized",
            "chunk_size_recommended": 5 * 1024 * 1024  # 5MB
        }
        
    except Exception as e:
        print(f"❌ Error initializing chunked upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize upload: {str(e)}")

@router.post("/api/admin/upload-chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a single chunk"""
    try:
        # Get upload session
        if upload_id not in upload_sessions:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        session = upload_sessions[upload_id]
        
        # Save chunk to temporary file
        chunk_path = os.path.join(session["temp_dir"], f"chunk_{chunk_index}")
        
        async with aiofiles.open(chunk_path, 'wb') as f:
            content = await chunk.read()
            await f.write(content)
        
        # Track received chunk
        session["chunks_received"].add(chunk_index)
        
        return {
            "message": f"Chunk {chunk_index} uploaded successfully",
            "chunks_received": len(session["chunks_received"]),
            "total_chunks": session["total_chunks"],
            "progress": len(session["chunks_received"]) / session["total_chunks"] * 100
        }
        
    except Exception as e:
        print(f"❌ Error uploading chunk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload chunk: {str(e)}")

@router.post("/api/admin/finalize-chunked-upload")
async def finalize_chunked_upload(
    upload_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Finalize chunked upload by combining chunks"""
    try:
        # Get upload session
        if upload_id not in upload_sessions:
            raise HTTPException(status_code=404, detail="Upload session not found")
        
        session = upload_sessions[upload_id]
        agent_id = session["agent_id"]
        
        # Verify all chunks received
        expected_chunks = set(range(session["total_chunks"]))
        if session["chunks_received"] != expected_chunks:
            missing_chunks = expected_chunks - session["chunks_received"]
            raise HTTPException(
                status_code=400, 
                detail=f"Missing chunks: {sorted(missing_chunks)}"
            )
        
        # Combine chunks into final file
        temp_file_path = os.path.join(session["temp_dir"], "combined.zip")
        
        with open(temp_file_path, 'wb') as output_file:
            for chunk_index in range(session["total_chunks"]):
                chunk_path = os.path.join(session["temp_dir"], f"chunk_{chunk_index}")
                with open(chunk_path, 'rb') as chunk_file:
                    output_file.write(chunk_file.read())
        
        # Process the combined file (same as regular upload)
        agent_dir = f"static/task_images/agent_{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)
        
        images_processed = 0
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue
                
                file_ext = os.path.splitext(file_info.filename.lower())[1]
                if file_ext not in supported_extensions:
                    continue
                
                with zip_ref.open(file_info) as source_file:
                    safe_filename = os.path.basename(file_info.filename)
                    if not safe_filename:
                        continue
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_name = safe_filename
                    while os.path.exists(os.path.join(agent_dir, safe_filename)):
                        name, ext = os.path.splitext(original_name)
                        safe_filename = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    image_path = os.path.join(agent_dir, safe_filename)
                    with open(image_path, 'wb') as dest_file:
                        dest_file.write(source_file.read())
                    
                    images_processed += 1
                    
                    if images_processed >= 5000:
                        break
        
        # Reset agent progress
        progress = db.query(TaskProgress).filter(TaskProgress.agent_id == agent_id).first()
        if progress:
            progress.current_index = 0
            progress.updated_at = datetime.utcnow()
        else:
            progress = TaskProgress(agent_id=agent_id, current_index=0)
            db.add(progress)
        
        db.commit()
        
        # Cleanup temporary files
        try:
            shutil.rmtree(session["temp_dir"])
        except Exception as cleanup_error:
            print(f"⚠️ Error cleaning up temp files: {cleanup_error}")
        
        # Remove upload session
        del upload_sessions[upload_id]
        
        return {
            "message": "Chunked upload completed successfully",
            "agent_id": agent_id,
            "images_processed": images_processed,
            "agent_directory": agent_dir
        }
        
    except Exception as e:
        print(f"❌ Error finalizing chunked upload: {e}")
        
        # Cleanup on error
        try:
            if upload_id in upload_sessions:
                session = upload_sessions[upload_id]
                if os.path.exists(session["temp_dir"]):
                    shutil.rmtree(session["temp_dir"])
                del upload_sessions[upload_id]
        except Exception as cleanup_error:
            print(f"⚠️ Error during error cleanup: {cleanup_error}")
        
        raise HTTPException(status_code=500, detail=f"Failed to finalize upload: {str(e)}")
        # agent_routes.py - Part 9: Data Export & Final Endpoints

    try:
        general_dir = "static/task_images/crime_records_wide"
        os.makedirs(general_dir, exist_ok=True)
        
        zip_content = await zip_file.read()
        
        images_processed = 0
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue
                
                file_ext = os.path.splitext(file_info.filename.lower())[1]
                if file_ext not in supported_extensions:
                    continue
                
                with zip_ref.open(file_info) as source_file:
                    safe_filename = os.path.basename(file_info.filename)
                    if not safe_filename:
                        continue
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_name = safe_filename
                    while os.path.exists(os.path.join(general_dir, safe_filename)):
                        name, ext = os.path.splitext(original_name)
                        safe_filename = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    image_path = os.path.join(general_dir, safe_filename)
                    with open(image_path, 'wb') as dest_file:
                        dest_file.write(source_file.read())
                    
                    images_processed += 1
                    
                    if images_processed >= 10000:  # Higher limit for general upload
                        break
        
        return {
            "message": "Bulk images uploaded successfully",
            "images_processed": images_processed,
            "directory": general_dir,
            "note": "These images are available to all agents who don't have specific task assignments"
        }
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        print(f"❌ Bulk upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")

# ===================== DATA EXPORT AND PREVIEW =====================

@router.get("/api/admin/export-excel")
def export_to_excel(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Export submitted data to Excel"""
    try:
        query = db.query(SubmittedForm)
        
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        
        if date_from:
            try:
                from_date = datetime.fromisoformat(date_from)
                query = query.filter(SubmittedForm.submitted_at >= from_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                to_date = datetime.fromisoformat(date_to)
                to_date = to_date.replace(hour=23, minute=59, second=59)
                query = query.filter(SubmittedForm.submitted_at <= to_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        submissions = query.order_by(SubmittedForm.submitted_at.desc()).all()
        
        if not submissions:
            raise HTTPException(status_code=404, detail="No data found with current filters")
        
        excel_data = []
        for submission in submissions:
            try:
                form_data = json.loads(submission.form_data)
                
                row_data = {
                    'Submission_ID': submission.id,
                    'Agent_ID': submission.agent_id,
                    'Submitted_At': submission.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'Image_Name': form_data.get('image_name', 'Unknown')
                }
                
                form_fields = [
                    'DR_NO', 'Date_Rptd', 'DATE_OCC', 'TIME_OCC', 'Unique_Identifier',
                    'AREA_NAME', 'Rpt_Dist_No', 'VIN', 'Crm', 'Crm_Cd_Desc', 'Mocodes',
                    'Vict_Age', 'Geolocation', 'DEPARTMENT', 'Premis_Cd', 'Premis_Desc',
                    'ARREST_KEY', 'PD_DESC', 'CCD_LONCOD', 'Status_Desc', 'LAW_CODE',
                    'SubAgency', 'Charge', 'Race', 'LOCATION', 'SeqID', 'LAT', 'LON',
                    'Point', 'Shape__Area'
                ]
                
                for field in form_fields:
                    row_data[field] = form_data.get(field, '')
                
                excel_data.append(row_data)
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error for submission {submission.id}: {e}")
                continue
        
        if not excel_data:
            raise HTTPException(status_code=404, detail="No valid data found")
        
        try:
            import pandas as pd
        except ImportError:
            raise HTTPException(status_code=500, detail="pandas not installed. Run: pip install pandas")
        
        df = pd.DataFrame(excel_data)
        
        output = io.BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Crime Records Data', index=False)
                
                workbook = writer.book
                worksheet = writer.sheets['Crime Records Data']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value and len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max(max_length + 2, 10), 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating Excel file: {str(e)}")
        
        output.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crime_records_export_{timestamp}.xlsx"
        
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        
        return StreamingResponse(
            io.BytesIO(output.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in Excel export: {e}")
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")

@router.get("/api/admin/preview-data")
async def preview_data(
    agent_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Preview submitted data with pagination"""
    try:
        query = db.query(SubmittedForm)
        
        if agent_id:
            query = query.filter(SubmittedForm.agent_id == agent_id)
        
        if date_from:
            try:
                from_date = datetime.strptime(date_from, '%Y-%m-%d')
                query = query.filter(SubmittedForm.submitted_at >= from_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                to_date = datetime.strptime(date_to, '%Y-%m-%d')
                to_date = to_date.replace(hour=23, minute=59, second=59)
                query = query.filter(SubmittedForm.submitted_at <= to_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        total_count = query.count()
        submissions = query.order_by(SubmittedForm.submitted_at.desc()).limit(min(limit, 1000)).all()
        
        result = []
        for submission in submissions:
            try:
                # Handle both JSON and string format
                if isinstance(submission.form_data, str):
                    form_data = json.loads(submission.form_data)
                else:
                    form_data = submission.form_data
                
                # Get agent name
                agent = db.query(Agent).filter(Agent.agent_id == submission.agent_id).first()
                agent_name = agent.name if agent else "Unknown"
                
                result.append({
                    "id": submission.id,
                    "agent_id": submission.agent_id,
                    "agent_name": agent_name,
                    "submitted_at": submission.submitted_at.isoformat(),
                    "image_name": form_data.get('image_name', 'Unknown'),
                    "form_data": form_data,
                    "data_preview": {
                        "DR_NO": form_data.get('DR_NO', ''),
                        "DATE_OCC": form_data.get('DATE_OCC', ''),
                        "AREA_NAME": form_data.get('AREA_NAME', ''),
                        "Crm_Cd_Desc": form_data.get('Crm_Cd_Desc', ''),
                        "LOCATION": form_data.get('LOCATION', '')
                    }
                })
            except Exception as parse_error:
                print(f"Error parsing submission {submission.id}: {parse_error}")
                continue
        
        return {
            "success": True,
            "data": result,
            "total_count": total_count,
            "returned_count": len(result),
            "filters_applied": {
                "agent_id": agent_id,
                "date_from": date_from,
                "date_to": date_to,
                "limit": limit
            }
        }
        
    except Exception as e:
        print(f"❌ Error in data preview: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")

@router.get("/api/admin/test-data")
async def test_data_availability(db: Session = Depends(get_db)):
    """Test data availability and system health"""
    try:
        # Count records in each table
        agent_count = db.query(Agent).count()
        submission_count = db.query(SubmittedForm).count()
        session_count = db.query(AgentSession).count()
        progress_count = db.query(TaskProgress).count()
        
        # Test recent activity
        recent_submissions = db.query(SubmittedForm).filter(
            SubmittedForm.submitted_at >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        active_sessions = db.query(AgentSession).filter(
            AgentSession.logout_time.is_(None)
        ).count()
        
        return {
            "success": True,
            "message": f"Data available - Agents: {agent_count}, Submissions: {submission_count}, Sessions: {session_count}",
            "counts": {
                "agents": agent_count,
                "submissions": submission_count,
                "sessions": session_count,
                "task_progress": progress_count,
                "recent_submissions_24h": recent_submissions,
                "active_sessions": active_sessions
            },
            "system_health": {
                "database_connected": True,
                "timestamp": datetime.utcnow().isoformat(),
                "tables_accessible": True
            }
        }
        
    except Exception as e:
        print(f"❌ Error testing data: {e}")
        return {
            "success": False,
            "error": str(e),
            "system_health": {
                "database_connected": False,
                "timestamp": datetime.utcnow().isoformat(),
                "tables_accessible": False
            }
        }

# ===================== HEALTH CHECK ENDPOINT =====================

@router.get("/api/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "Agent Task Management System",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# ===================== END OF ROUTES =====================
