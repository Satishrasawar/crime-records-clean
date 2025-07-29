import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import your local modules (ensure these imports work)
try:
    from agent_routes import router as agent_router
    from database import Base, engine
    from models import Agent, TaskProgress, SubmittedForm, AgentSession
except ImportError as e:
    print(f"Import error: {e}")
    # Create a minimal app if imports fail
    pass

# Create directories if they don't exist
os.makedirs("static/task_images", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Crime Records Data Entry System", 
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

# Create database tables on startup
print("🔧 Creating database tables...")
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
except Exception as e:
    print(f"❌ Error creating database tables: {e}")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# Include routers
try:
    app.include_router(agent_router)
    print("✅ Agent routes included")
except Exception as e:
    print(f"Warning: Could not include agent routes: {e}")

# CRITICAL: Health endpoint (must work for Railway)
@app.get("/health")
def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy", 
        "platform": "Railway",
        "message": "Service is running"
    }

# Root endpoint
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Crime Records Data Entry System API v2.0",
        "status": "running",
        "platform": "Railway",
        "health_check": "/health"
    }

# Serve HTML files
@app.get("/admin.html")
async def serve_admin_panel():
    """Serve admin dashboard"""
    try:
        if os.path.exists("admin.html"):
            return FileResponse("admin.html")
        return {"error": "Admin panel not found"}
    except Exception as e:
        return {"error": f"Could not serve admin panel: {e}"}

@app.get("/agent.html")
async def serve_agent_panel():
    """Serve agent interface"""
    try:
        if os.path.exists("agent.html"):
            return FileResponse("agent.html")
        return {"error": "Agent panel not found"}
    except Exception as e:
        return {"error": f"Could not serve agent panel: {e}"}

# Debug info
@app.get("/debug")
def debug_info():
    """Debug endpoint to check what files exist"""
    import os
    files = os.listdir(".")
    return {
        "files": files,
        "python_version": os.sys.version,
        "environment": dict(os.environ)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
