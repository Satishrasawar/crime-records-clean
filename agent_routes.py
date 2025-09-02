from fastapi import APIRouter, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta
import uuid
import secrets
import string
import re
import logging
import bcrypt

# Import from main directly
from main import get_db, Agent, AgentSession, hash_password, verify_password

router = APIRouter()
logger = logging.getLogger(__name__)

# Utility functions
def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_mobile(mobile: str) -> bool:
    """Validate mobile number format"""
    if not mobile:
        return False
        
    # Remove all non-digit characters except leading +
    clean_mobile = re.sub(r'[^\d+]', '', mobile)
    
    # Check if it has a country code (starts with +) or is just digits
    if clean_mobile.startswith('+'):
        # International format: + followed by 10-15 digits
        return re.match(r'^\+\d{10,15}$', clean_mobile) is not None
    else:
        # Local format: 10-15 digits
        return re.match(r'^\d{10,15}$', clean_mobile) is not None

def generate_unique_agent_id(db: Session):
    """Generate a unique agent ID"""
    max_attempts = 100
    for attempt in range(max_attempts):
        agent_number = secrets.randbelow(900000) + 100000
        agent_id = f"AGT{agent_number}"
        existing = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not existing:
            return agent_id
    
    # Fallback to timestamp-based ID
    import time
    fallback_id = f"AGT{str(int(time.time()))[-6:]}"
    existing_fallback = db.query(Agent).filter(Agent.agent_id == fallback_id).first()
    if existing_fallback:
        raise HTTPException(status_code=500, detail="Failed to generate unique agent ID")
    return fallback_id

def generate_secure_password():
    """Generate a secure password ensuring variety"""
    characters = string.ascii_letters + string.digits + "!@#$%"
    password_parts = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%")
    ]
    for _ in range(6):  # Total length 10
        password_parts.append(secrets.choice(characters))
    secrets.SystemRandom().shuffle(password_parts)
    return ''.join(password_parts)

# ===================== AGENT REGISTRATION ENDPOINTS =====================

@router.post("/agents/register")
async def register_agent(
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    dob: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Register a new agent with proper credentials"""
    try:
        logger.info(f"🆕 Agent registration attempt: {name}, {email}, {mobile}")
        
        # Clean inputs
        name = name.strip()
        email = email.strip().lower()
        mobile = re.sub(r'[\s\-\(\)]', '', mobile)
        
        logger.info(f"Cleaned inputs - Name: {name}, Email: {email}, Mobile: {mobile}")
        
        # Collect all validation errors
        validation_errors = []
        
        if not name or len(name) < 2:
            validation_errors.append("Name must be at least 2 characters long")
        
        if not validate_email(email):
            validation_errors.append("Invalid email format")
        
        if not validate_mobile(mobile):
            validation_errors.append(f"Invalid mobile number format: {mobile}. Use 10-15 digits")
        
        # Validate optional fields
        dob_date = None
        if dob:
            try:
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                age = (datetime.now().date() - dob_date).days // 365
                if age < 16 or age > 80:
                    validation_errors.append("Agent must be between 16 and 80 years old")
            except ValueError:
                validation_errors.append("Invalid date format. Use YYYY-MM-DD")
        
        if gender and gender not in ['Male', 'Female', 'Other']:
            validation_errors.append("Gender must be Male, Female, or Other")
        
        if validation_errors:
            logger.error(f"Validation errors: {validation_errors}")
            raise HTTPException(status_code=400, detail="Validation errors: " + "; ".join(validation_errors))
        
        # Check if agent already exists
        existing_agent = db.query(Agent).filter(
            (Agent.email == email) | (Agent.mobile == mobile)
        ).first()
        
        if existing_agent:
            if existing_agent.email == email:
                logger.error(f"Email already registered: {email}")
                raise HTTPException(status_code=409, detail="Email already registered")
            else:
                logger.error(f"Mobile number already registered: {mobile}")
                raise HTTPException(status_code=409, detail="Mobile number already registered")
        
        # Generate unique agent credentials
        agent_id = generate_unique_agent_id(db)
        password = generate_secure_password()
        hashed_password = hash_password(password)
        
        logger.info(f"Generated credentials - Agent ID: {agent_id}, Password: {password}")
        
        # Create new agent
        agent_data = {
            'agent_id': agent_id,
            'name': name,
            'email': email,
            'mobile': mobile,
            'hashed_password': hashed_password,
            'status': "active",
            'created_at': datetime.utcnow()
        }
        
        if dob:
            agent_data['dob'] = dob
        if country:
            agent_data['country'] = country.strip()
        if gender:
            agent_data['gender'] = gender
        
        new_agent = Agent(**agent_data)
        
        db.add(new_agent)
        db.commit()
        
        logger.info(f"✅ Agent registered successfully: {agent_id}")
        
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
                "You can now log in with these credentials"
            ]
        }
        
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
        logger.error(f"❌ Registration error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/agents/check-availability")
async def check_availability(
    email: Optional[str] = Form(None),
    mobile: Optional[str] = Form(None),
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
        logger.error(f"❌ Availability check error: {e}")
        return {"available": False, "message": "Check failed"}

@router.get("/agents/test-registration")
async def test_registration_system(db: Session = Depends(get_db)):
    """Test the registration system"""
    try:
        agent_count = db.query(Agent).count()
        test_agent_id = generate_unique_agent_id(db)
        test_password = generate_secure_password()
        
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
            "optional_fields": ["dob", "country", "gender"]
        }
        
    except Exception as e:
        return {
            "system_status": "error",
            "error": str(e),
            "database_connected": False
        }

# ===================== AGENT AUTHENTICATION ENDPOINTS =====================

@router.post("/agents/login")
async def agent_login(
    agent_id: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Agent login with proper authentication"""
    try:
        # Check if agent is locked
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        
        if not agent:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check if account is locked
        if agent.locked_until and agent.locked_until > datetime.utcnow():
            remaining_time = (agent.locked_until - datetime.utcnow()).seconds
            raise HTTPException(
                status_code=423, 
                detail=f"Account locked. Try again in {remaining_time} seconds"
            )
        
        # Verify password
        if not verify_password(password, agent.hashed_password):
            # Increment failed attempts
            agent.login_attempts += 1
            
            # Lock account after 3 failed attempts for 5 minutes
            if agent.login_attempts >= 3:
                agent.locked_until = datetime.utcnow() + timedelta(minutes=5)
                agent.login_attempts = 0
                db.commit()
                raise HTTPException(
                    status_code=423, 
                    detail="Account locked for 5 minutes due to multiple failed attempts"
                )
            
            db.commit()
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Reset login attempts on successful login
        agent.login_attempts = 0
        agent.locked_until = None
        agent.last_login = datetime.utcnow()
        agent.is_currently_logged_in = True
        db.commit()
        
        # Create session token
        session_token = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=8)
        
        agent_session = AgentSession(
            agent_id=agent_id,
            session_token=session_token,
            expires_at=expires_at,
            is_active=True
        )
        db.add(agent_session)
        db.commit()
        
        return {
            "success": True,
            "agent_id": agent.agent_id,
            "name": agent.name,
            "session_token": session_token,
            "expires_at": expires_at.isoformat(),
            "message": "Login successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/agents/logout")
async def agent_logout(
    agent_id: str = Form(...),
    session_token: str = Form(...),
    db: Session = Depends(get_db)
):
    """Agent logout"""
    try:
        # Find and invalidate session
        session = db.query(AgentSession).filter(
            AgentSession.agent_id == agent_id,
            AgentSession.session_token == session_token,
            AgentSession.is_active == True
        ).first()
        
        if session:
            session.is_active = False
            session.expires_at = datetime.utcnow()
        
        # Update agent status
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if agent:
            agent.is_currently_logged_in = False
        
        db.commit()
        
        return {
            "success": True,
            "message": "Logout successful"
        }
        
    except Exception as e:
        logger.error(f"❌ Agent logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

# ===================== AGENT MANAGEMENT ENDPOINTS =====================

@router.get("/agents")
async def get_agents(db: Session = Depends(get_db)):
    """Get all agents"""
    try:
        agents = db.query(Agent).all()
        return [
            {
                "id": agent.id,
                "agent_id": agent.agent_id,
                "name": agent.name,
                "email": agent.email,
                "mobile": agent.mobile,
                "status": agent.status,
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "last_login": agent.last_login.isoformat() if agent.last_login else None,
                "is_currently_logged_in": agent.is_currently_logged_in,
                "tasks_completed": agent.tasks_completed,
                "dob": agent.dob,
                "country": agent.country,
                "gender": agent.gender,
                "login_attempts": agent.login_attempts,
                "locked_until": agent.locked_until.isoformat() if agent.locked_until else None
            }
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Get agents error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch agents")

@router.patch("/agents/{agent_id}/status")
async def update_agent_status(
    agent_id: str,
    status_data: dict,
    db: Session = Depends(get_db)
):
    """Update agent status"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    new_status = status_data.get("status")
    if new_status not in ["active", "inactive"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    agent.status = new_status
    db.commit()
    
    return {"success": True, "message": f"Agent status updated to {new_status}"}

@router.post("/agents/{agent_id}/reset-password")
async def reset_agent_password(agent_id: str, db: Session = Depends(get_db)):
    """Reset agent password"""
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Generate new password
    new_password = generate_secure_password()
    agent.hashed_password = hash_password(new_password)
    agent.login_attempts = 0
    agent.locked_until = None
    
    db.commit()
    
    return {
        "success": True,
        "agent_id": agent_id,
        "new_password": new_password,
        "message": f"Password reset for {agent.name}"
    }

# ===================== HEALTH CHECK ENDPOINT =====================

@router.get("/agents/health")
async def agent_health_check():
    """Agent service health check"""
    return {
        "status": "healthy",
        "service": "Agent Management Service",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints_available": [
            "/api/agents/register",
            "/api/agents/login",
            "/api/agents/logout",
            "/api/agents",
            "/api/agents/{agent_id}/status",
            "/api/agents/{agent_id}/reset-password"
        ]
    }

# ===================== AGENT PROFILE ENDPOINTS =====================

@router.get("/agents/{agent_id}/profile")
async def get_agent_profile(agent_id: str, db: Session = Depends(get_db)):
    """Get agent profile information"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "email": agent.email,
            "mobile": agent.mobile,
            "status": agent.status,
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "last_login": agent.last_login.isoformat() if agent.last_login else None,
            "tasks_completed": agent.tasks_completed,
            "dob": agent.dob,
            "country": agent.country,
            "gender": agent.gender
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting agent profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent profile")

@router.put("/agents/{agent_id}/profile")
async def update_agent_profile(
    agent_id: str,
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    mobile: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Update agent profile information"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Validate inputs if provided
        if email and not validate_email(email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        if mobile and not validate_mobile(mobile):
            raise HTTPException(status_code=400, detail="Invalid mobile number format")
        
        # Check for duplicate email
        if email and email != agent.email:
            existing_agent = db.query(Agent).filter(Agent.email == email).first()
            if existing_agent:
                raise HTTPException(status_code=409, detail="Email already registered")
        
        # Check for duplicate mobile
        if mobile and mobile != agent.mobile:
            existing_agent = db.query(Agent).filter(Agent.mobile == mobile).first()
            if existing_agent:
                raise HTTPException(status_code=409, detail="Mobile number already registered")
        
        # Update fields
        if name:
            agent.name = name.strip()
        if email:
            agent.email = email.strip().lower()
        if mobile:
            agent.mobile = re.sub(r'[\s\-\(\)]', '', mobile)
        
        db.commit()
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating agent profile: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update profile")

@router.post("/agents/{agent_id}/change-password")
async def change_agent_password(
    agent_id: str,
    current_password: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Change agent password"""
    try:
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Verify current password
        if not verify_password(current_password, agent.hashed_password):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        
        # Update password
        agent.hashed_password = hash_password(new_password)
        db.commit()
        
        return {
            "success": True,
            "message": "Password changed successfully",
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error changing password: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to change password")
