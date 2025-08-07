import os
import sys
from datetime import datetime
from fastapi import FastAPI

app = FastAPI(
    title="Client Records Data Entry System", 
    version="2.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Client Records Data Entry System API v2.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "Railway",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
