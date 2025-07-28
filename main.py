from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes.agent_routes import router as agent_router

# Database setup
from app.database import Base, engine
from app.models import Agent, TaskProgress, SubmittedForm, AgentSession

import os
PORT = int(os.environ.get("PORT", 8000))
ENVIRONMENT = os.environ.get("RAILWAY_ENVIRONMENT", "development")

# Database URL for Railway (they provide PostgreSQL)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./database.db")

app = FastAPI(
    title="Crime Records Data Entry System",
    version="2.0.0",
    description="Enhanced system with ZIP upload and Excel export"
)

# Update CORS for your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://agent-task-system.com",
        "https://www.agent-task-system.com",
        "http://localhost:3000",  # For development
        "http://127.0.0.1:3000"   # For development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on startup
print("🔧 Creating database tables...")
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    # Verify tables were created
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"📋 Available tables: {', '.join(tables)}")
    
except Exception as e:
    print(f"❌ Error creating database tables: {e}")

# Create directories if they don't exist
os.makedirs("static/task_images/crime_records_wide", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Crime Records Data Entry System", 
    version="2.0.0",
    description="Enhanced system with ZIP upload and Excel export"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(agent_router)

# ✅ ADD THESE ROUTES TO SERVE HTML FILES
@app.get("/admin.html")
async def serve_admin_panel():
    """Serve the admin dashboard"""
    admin_file = "../admin.html"  # Go up one level from backend
    if os.path.exists(admin_file):
        return FileResponse(admin_file)
    else:
        # Try current directory
        if os.path.exists("admin.html"):
            return FileResponse("admin.html")
        # Try parent directory
        elif os.path.exists("../admin.html"):
            return FileResponse("../admin.html")
        else:
            return {"error": "Admin panel not found. Please ensure admin.html is in the project root or backend directory."}

@app.get("/agent.html")
async def serve_agent_panel():
    """Serve the agent interface"""
    agent_file = "../agent.html"  # Go up one level from backend
    if os.path.exists(agent_file):
        return FileResponse(agent_file)
    else:
        # Try current directory
        if os.path.exists("agent.html"):
            return FileResponse("agent.html")
        # Try parent directory
        elif os.path.exists("../agent.html"):
            return FileResponse("../agent.html")
        else:
            return {"error": "Agent panel not found. Please ensure agent.html is in the project root or backend directory."}

@app.get("/")
def root():
    return {
        "message": "Crime Records Data Entry System API v2.0", 
        "panels": {
            "admin_dashboard": "http://localhost:8000/admin.html",
            "agent_interface": "http://localhost:8000/agent.html"
        },
        "features": [
            "Agent Registration with Auto-Generated Credentials",
            "ZIP File Upload for Bulk Image Assignment", 
            "Excel Export with Filtering",
            "Auto-Progression After Form Submission",
            "Real-time Progress Tracking"
        ],
        "api_docs": "http://localhost:8000/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)