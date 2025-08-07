import os
import sys
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

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

# ADD DATABASE SECTION:
database_ready = False
db_dependency = get_mock_db

try:
    print("📦 Importing database modules...")
    from database import Base, engine, get_db
    from models import Agent, TaskProgress, SubmittedForm, AgentSession
    
    print("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    database_ready = True
    db_dependency = get_db
    
except Exception as e:
    print(f"❌ Database setup failed: {e}")
    import traceback
    traceback.print_exc()
    database_ready = False
    db_dependency = get_mock_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up...")
    yield
    print("🛑 Shutting down...")

# Create app with lifespan
app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add static files
try:
    os.makedirs("static/task_images", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Static files configured")
except Exception as e:
    print(f"❌ Static files setup failed: {e}")

@app.get("/")
def root():
    return {
        "message": "Client Records Data Entry System API v2.0",
        "status": "running",
        "database_ready": database_ready,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "Railway",
        "database_ready": database_ready,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

# ADD CORE ENDPOINTS:
@app.get("/debug")
def debug_info():
    """Debug endpoint"""
    return {
        "database_ready": database_ready,
        "python_version": sys.version,
        "files": os.listdir(".") if os.path.exists(".") else []
    }

@app.get("/api/admin/statistics")
async def get_admin_statistics(db = Depends(db_dependency)):
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
        
        # If database is ready, try to get real stats
        total_agents = db.query(Agent).count()
        total_tasks = db.query(TaskProgress).count()
        completed_tasks = db.query(TaskProgress).filter(TaskProgress.status == 'completed').count()
        
        return {
            "total_agents": total_agents,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": 0,
            "in_progress_tasks": 0
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
async def list_agents(db = Depends(db_dependency)):
    """List all agents"""
    try:
        if not database_ready:
            return []
        
        agents = db.query(Agent).all()
        return [{"agent_id": agent.agent_id, "name": agent.name} for agent in agents]
    except Exception as e:
        print(f"❌ Error listing agents: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
