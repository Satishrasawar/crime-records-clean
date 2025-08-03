import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize FastAPI app first
app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0",
    description="Enhanced system for agent-task-system.com"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL: Health endpoint (must work first)
@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy", 
        "platform": "Railway",
        "message": "Service is running",
        "imports_loaded": "database" in sys.modules
    }

# Root endpoint
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Client Records Data Entry System API v2.0",
        "status": "running",
        "platform": "Railway",
        "health_check": "/health"
    }

# Try to import and setup database
try:
    print("📦 Importing database modules...")
    from database import Base, engine
    from models import Agent, TaskProgress, SubmittedForm, AgentSession
    
    print("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    database_ready = True
except Exception as e:
    print(f"❌ Database setup failed: {e}")
    database_ready = False

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

# Create directories
try:
    os.makedirs("static/task_images", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("✅ Static files configured")
except Exception as e:
    print(f"❌ Static files setup failed: {e}")

# Serve HTML files
@app.get("/admin.html")
async def serve_admin_panel():
    """Serve admin dashboard"""
    try:
        if os.path.exists("admin.html"):
            return FileResponse("admin.html")
        return {"error": "Admin panel not found", "files": os.listdir(".")}
    except Exception as e:
        return {"error": f"Could not serve admin panel: {e}"}

@app.get("/agent.html")
async def serve_agent_panel():
    """Serve agent interface"""
    try:
        if os.path.exists("agent.html"):
            return FileResponse("agent.html")
        return {"error": "Agent panel not found", "files": os.listdir(".")}
    except Exception as e:
        return {"error": f"Could not serve agent panel: {e}"}

# Debug endpoint
@app.get("/debug")
def debug_info():
    """Debug endpoint to check system status"""
    import sys
    return {
        "files": os.listdir("."),
        "python_path": sys.path,
        "modules": list(sys.modules.keys()),
        "database_ready": database_ready,
        "routes_ready": routes_ready,
        "port": os.environ.get("PORT", "not set")
    }

# Status endpoint
@app.get("/status")
def system_status():
    """System status endpoint"""
    return {
        "database": "ready" if database_ready else "failed",
        "routes": "ready" if routes_ready else "failed",
        "health": "ok"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

