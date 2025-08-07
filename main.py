import os
import sys
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio


# ADD DATABASE SECTION:
database_ready = False

try:
    print("📦 Importing database modules...")
    from database import Base, engine, get_db
    from models import Agent, TaskProgress, SubmittedForm, AgentSession
    
    print("🔧 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    
    database_ready = True
    
except Exception as e:
    print(f"❌ Database setup failed: {e}")
    import traceback
    traceback.print_exc()
    database_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up...")
    yield
    print("🛑 Shutting down...")

# Modify app creation:
app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0",
    lifespan=lifespan  # ADD THIS
)
app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


