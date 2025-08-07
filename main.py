# MINIMAL WORKING FASTAPI FOR RAILWAY - TEST THIS FIRST

import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello Railway", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
