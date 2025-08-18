# COMPLETE FIXED AGENT REGISTRATION SYSTEM
# Replace your existing registration code with this

# ===================== STEP 1: UPDATE YOUR models.py =====================

# app/models.py - Complete replacement
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Agent(Base):
    __tablename__ = "agents"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(10), unique=True, index=True, nullable=False)  # AGT123456
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    mobile = Column(String(20), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)  # Plain text for now
    status = Column(String(20), default="active", nullable=False)  # active, inactive, pending, suspended
    
    # Optional demographic fields (for compatibility with admin form)
    dob = Column(Date, nullable=True)  # Date of birth
    country = Column(String(100), nullable=True)
    gender = Column(String(10), nullable=True)  # Male, Female, Other
    
    # Timestamp fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Additional tracking fields
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<Agent(agent_id='{self.agent_id}', name='{self.name}', email='{self.email}')>"

# Database migration script if you need to update existing database
"""
-- SQL to add missing columns to existing agents table:

ALTER TABLE agents ADD COLUMN mobile VARCHAR(20);
ALTER TABLE agents ADD COLUMN dob DATE;
ALTER TABLE agents ADD COLUMN country VARCHAR(100);
ALTER TABLE agents ADD COLUMN gender VARCHAR(10);
ALTER TABLE agents ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;
ALTER TABLE agents ADD COLUMN last_login TIMESTAMP NULL;
ALTER TABLE agents ADD COLUMN login_count INTEGER DEFAULT 0;

-- Add unique constraint on mobile (if not exists)
ALTER TABLE agents ADD CONSTRAINT unique_mobile UNIQUE (mobile);

-- Update existing records with default mobile numbers if needed
UPDATE agents SET mobile = CONCAT('+1234567', LPAD(id, 3, '0')) WHERE mobile IS NULL;
"""
class Admin(Base):
    """Admin model for admin login"""
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TaskProgress(Base):
    """Task progress tracking"""
    __tablename__ = "task_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    image_filename = Column(String(255), nullable=True)
    image_path = Column(String(500), nullable=True)
    status = Column(String(20), default="pending")  # pending, in_progress, completed, skipped
    current_index = Column(Integer, default=0)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SubmittedForm(Base):
    """Submitted form data"""
    __tablename__ = "submitted_forms"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    task_id = Column(Integer, ForeignKey("task_progress.id"), nullable=True)
    image_filename = Column(String(255), nullable=True)
    form_data = Column(Text, nullable=False)  # JSON string
    submitted_at = Column(DateTime, default=datetime.utcnow)

class AgentSession(Base):
    """Agent session tracking"""
    __tablename__ = "agent_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    login_time = Column(DateTime, default=datetime.utcnow)
    logout_time = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)

# ===================== STEP 2: UPDATE YOUR main.py REGISTRATION =====================

# Add this to your main.py - REPLACE existing registration endpoint

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
    This version handles all common error cases and provides detailed feedback
    """
    
    # Detailed logging for debugging
    print(f"🔄 Registration attempt - Name: '{name}', Email: '{email}'")
    
    try:
        # ===== VALIDATION PHASE =====
        
        # Check if database is ready
        if not database_ready:
            print("❌ Database not ready")
            raise HTTPException(
                status_code=503, 
                detail="Database service is temporarily unavailable. Please try again in a few moments."
            )
        
        # Validate required fields
        fields = {
            'name': name.strip() if name else '',
            'email': email.strip() if email else '',
            'mobile': mobile.strip() if mobile else '',
            'dob': dob.strip() if dob else '',
            'country': country.strip() if country else '',
            'gender': gender.strip() if gender else ''
        }
        
        print(f"📝 Field validation - All fields present: {all(fields.values())}")
        
        # Check for empty fields
        empty_fields = [field_name for field_name, value in fields.items() if not value]
        if empty_fields:
            raise HTTPException(
                status_code=400,
                detail=f"The following fields are required and cannot be empty: {', '.join(empty_fields)}"
            )
        
        # Email format validation
        import re
        email_pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        if not re.match(email_pattern, fields['email']):
            raise HTTPException(
                status_code=400,
                detail="Please enter a valid email address (e.g., user@example.com)"
            )
        
        # Mobile number validation (accept various formats)
        mobile_clean = re.sub(r'[\s\-\(\)\.]+', '', fields['mobile'])  # Remove spaces, dashes, parentheses, dots
        if not re.match(r'^\+?\d{10,15}$', mobile_clean):
            raise HTTPException(
                status_code=400,
                detail="Mobile number must contain 10-15 digits. You can include country code with + prefix."
            )
        
        # Date of birth validation
        try:
            from datetime import datetime
            dob_date = datetime.strptime(fields['dob'], '%Y-%m-%d')
            today = datetime.now()
            age = (today - dob_date).days // 365
            
            if age < 16:
                raise HTTPException(status_code=400, detail="Agent must be at least 16 years old")
            if age > 100:
                raise HTTPException(status_code=400, detail="Please enter a valid date of birth")
                
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Please use YYYY-MM-DD format (e.g., 1990-12-25)"
            )
        
        # Gender validation
        valid_genders = ['Male', 'Female', 'Other', 'Prefer not to say']
        if fields['gender'] not in valid_genders:
            raise HTTPException(
                status_code=400,
                detail=f"Gender must be one of: {', '.join(valid_genders)}"
            )
        
        print("✅ All field validation passed")
        
        # ===== DATABASE CHECKS =====
        
        try:
            # Check for existing email (case-insensitive)
            existing_agent = db.query(Agent).filter(
                Agent.email == fields['email'].lower()
            ).first()
            
            if existing_agent:
                print(f"❌ Email already exists: {fields['email']}")
                raise HTTPException(
                    status_code=409,
                    detail=f"An agent with email '{fields['email']}' is already registered. Please use a different email address."
                )
                
            print("✅ Email availability check passed")
            
        except HTTPException:
            raise
        except Exception as db_check_error:
            print(f"⚠️ Database check error (continuing): {db_check_error}")
            # Continue with registration - this might be a temporary DB issue
        
        # ===== CREDENTIAL GENERATION =====
        
        import secrets
        import string
        
        def generate_unique_agent_id():
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
        
        # Generate credentials
        agent_id = generate_unique_agent_id()
        password = generate_secure_password()
        
        print(f"🔑 Generated credentials - ID: {agent_id}, Password: [HIDDEN]")
        
        # ===== DATABASE INSERTION =====
        
        try:
            print("💾 Creating agent record...")
            
            # Create agent object
            new_agent = Agent(
                agent_id=agent_id,
                name=fields['name'],
                email=fields['email'].lower(),
                mobile=mobile_clean,
                dob=fields['dob'],
                country=fields['country'],
                gender=fields['gender'],
                password=password,
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
                "name": fields['name'],
                "email": fields['email'].lower(),
                "mobile": mobile_clean,
                "dob": fields['dob'],
                "country": fields['country'],
                "gender": fields['gender'],
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

# ===================== STEP 3: ADD SUPPORTING ENDPOINTS =====================

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

@app.post("/api/test/register-agent")
@limiter.limit("5/minute")
async def test_register_agent(request: Request, db=Depends(db_dependency)):
    """Test registration endpoint with sample data"""
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

# ===================== STEP 4: AGENT LOGIN ENDPOINT =====================

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
        print(f"❌ Login error: {login_error}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Login failed due to a system error. Please try again."
        )

