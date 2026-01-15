from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import api_router
from app.api.endpoints.yolo import model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application.
    Manages model loading on startup and cleanup on shutdown.
    """
    # Startup: Load models
    print("Starting up: Loading models...")
    try:
        model_loader.load_all_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Warning: Error loading models: {e}")
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    print("Shutting down: Unloading models...")
    model_loader.unload_all_models()
    print("Models unloaded successfully")


app = FastAPI(
    title="ONNX TensorRT API",
    description="API for model inference using ONNX and TensorRT",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ONNX TensorRT API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
